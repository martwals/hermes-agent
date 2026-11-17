"""SUP-53 / DD-DENNIS-009 — the gateway owns the timeout verdict on synchronous turns.

Rows T1-T9 of the design's test plan. What each one is FOR:

* T1-T3 the deadline answers in the API's own JSON envelope, carrying the session id, while the
  turn keeps running (it is not cancelled) — the whole point of the ticket.
* T4    a turn that finishes after its caller was answered is accounted for, not silent.
* T5-T8 the durable claim turns a duplicate into a 409 with a poll pointer instead of a second
  paid turn, on the same store `/v1/runs` uses.
* T9    pins the streaming path's existing keepalive so the ticket's "unverified" claim cannot
  regress.
"""

import asyncio
import json
import time
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import api_server as api_server_module
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore

API_KEY = "sk-sup53"
SESSION = "sup53-deadline-session"
USER_TURN = {"role": "user", "content": "give me four verdicts"}
AUTH = {"Authorization": f"Bearer {API_KEY}", "X-Hermes-Session-Id": SESSION}


def _make_adapter(deadline: float = 0.0, api_key: str = API_KEY) -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra=extra))
    # The deadline is a float here so a test costs milliseconds, not the configured 840 s.
    adapter._sync_turn_deadline_seconds = deadline
    previous = getattr(adapter, "_run_idempotency_store", None)
    if previous is not None:
        previous.close()
    adapter._run_idempotency_store = RunIdempotencyStore(":memory:")
    return adapter


def _create_app(adapter: APIServerAdapter) -> web.Application:
    middlewares = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=middlewares)
    app["api_server_adapter"] = adapter
    app.router.add_get("/health/detailed", adapter._handle_health_detailed)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


def _reply(text: str = "late answer"):
    return (
        {"final_response": text, "messages": [], "api_calls": 1},
        {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    )


class _SlowTurn:
    """A run_agent double that blocks until released, recording whether it finished."""

    def __init__(self):
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.calls = 0
        self.finished = 0

    async def __call__(self, **_kwargs):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        self.finished += 1
        return _reply()


# ---------------------------------------------------------------------------
# T1 / T2 / T3 — the deadline, its off switch, and turn survival
# ---------------------------------------------------------------------------


class TestSyncTurnDeadline:
    @pytest.mark.asyncio
    async def test_expired_deadline_answers_gateway_timeout_with_the_session_id(self):
        """T1: a turn past the deadline is answered by the gateway, in JSON, naming the session."""
        adapter = _make_adapter(deadline=0.05)
        turn = _SlowTurn()
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with patch.object(adapter, "_run_agent", turn):
                response = await cli.post(
                    "/v1/chat/completions", json={"messages": [USER_TURN]}, headers=dict(AUTH))
                body = await response.json()
                turn.release.set()
                await asyncio.sleep(0.05)

        assert response.status == 504
        assert body["error"]["code"] == "gateway_timeout"
        assert body["error"]["type"] == "gateway_timeout"
        assert body["hermes"]["session_id"] == SESSION
        assert body["hermes"]["poll"] == {"method": "GET", "path": f"/api/sessions/{SESSION}/messages"}
        assert body["hermes"]["deadline_seconds"] == pytest.approx(0.05, abs=1e-6)
        assert response.headers["X-Hermes-Session-Id"] == SESSION

    @pytest.mark.asyncio
    async def test_deadline_disabled_returns_the_answer_as_before(self):
        """T2 (control): ``0`` disables the deadline and restores the awaited-response behaviour."""
        adapter = _make_adapter(deadline=0)
        turn = _SlowTurn()
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with patch.object(adapter, "_run_agent", turn):
                pending = asyncio.create_task(
                    cli.post("/v1/chat/completions", json={"messages": [USER_TURN]}, headers=dict(AUTH)))
                await turn.started.wait()
                turn.release.set()
                response = await pending
                body = await response.json()

        assert response.status == 200
        assert body["choices"][0]["message"]["content"] == "late answer"

    @pytest.mark.asyncio
    async def test_expired_turn_is_not_cancelled_and_still_completes(self):
        """T3: the caller gets 504 and the turn runs on — that is what makes the poll useful."""
        adapter = _make_adapter(deadline=0.05)
        turn = _SlowTurn()
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with patch.object(adapter, "_run_agent", turn):
                response = await cli.post(
                    "/v1/chat/completions", json={"messages": [USER_TURN]}, headers=dict(AUTH))
                assert response.status == 504
                assert turn.finished == 0, "the turn must still be running when the 504 is written"
                turn.release.set()
                for _ in range(100):
                    if turn.finished:
                        break
                    await asyncio.sleep(0.01)

        assert turn.finished == 1, "the detached turn must run to completion, not be cancelled"


# ---------------------------------------------------------------------------
# T4 — a late completion is accounted for
# ---------------------------------------------------------------------------


class TestLateCompletionAccounting:
    @pytest.mark.asyncio
    async def test_late_completion_is_logged_and_counted(self, caplog):
        """T4: one WARNING per late completion plus the counter on /health/detailed."""
        adapter = _make_adapter(deadline=0.05)
        turn = _SlowTurn()
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with patch.object(adapter, "_run_agent", turn):
                with caplog.at_level("WARNING", logger="gateway.platforms.api_server"):
                    response = await cli.post(
                        "/v1/chat/completions", json={"messages": [USER_TURN]}, headers=dict(AUTH))
                    assert response.status == 504
                    turn.release.set()
                    for _ in range(100):
                        if adapter._late_completions_today:
                            break
                        await asyncio.sleep(0.01)
                    health = await (await cli.get("/health/detailed", headers=dict(AUTH))).json()

        assert adapter._late_completions_today == 1
        assert health["api_server"]["late_completions_today"] == 1
        assert health["api_server"]["last_late_completion_at"] is not None
        late_records = [r for r in caplog.records if "late_completion" in r.getMessage()]
        assert len(late_records) == 1
        assert SESSION in late_records[0].getMessage()
        assert "output_chars=" in late_records[0].getMessage()


# ---------------------------------------------------------------------------
# T5-T8 — the durable duplicate-turn claim
# ---------------------------------------------------------------------------


class TestDuplicateTurnClaim:
    @pytest.mark.asyncio
    async def test_live_duplicate_is_refused_and_runs_one_turn(self):
        """T5: same key + same body while the first turn is live → 409, one paid turn."""
        adapter = _make_adapter(deadline=0)
        turn = _SlowTurn()
        headers = {**AUTH, "Idempotency-Key": "sup53-key-live"}
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with patch.object(adapter, "_run_agent", turn):
                first = asyncio.create_task(
                    cli.post("/v1/chat/completions", json={"messages": [USER_TURN]}, headers=headers))
                await turn.started.wait()
                duplicate = await cli.post(
                    "/v1/chat/completions", json={"messages": [USER_TURN]}, headers=headers)
                duplicate_body = await duplicate.json()
                turn.release.set()
                first_response = await first

        assert duplicate.status == 409
        assert duplicate_body["error"]["code"] == "duplicate_turn_in_progress"
        assert duplicate_body["hermes"]["session_id"] == SESSION
        assert duplicate_body["hermes"]["poll"]["path"] == f"/api/sessions/{SESSION}/messages"
        assert turn.calls == 1, "the duplicate must not start a second turn"
        assert first_response.status == 200

    @pytest.mark.asyncio
    async def test_same_key_different_body_is_a_conflict(self):
        """T6: parity with /v1/runs — a reused key with a different payload is a conflict."""
        adapter = _make_adapter(deadline=0)
        turn = _SlowTurn()
        turn.release.set()
        headers = {**AUTH, "Idempotency-Key": "sup53-key-conflict"}
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with patch.object(adapter, "_run_agent", turn):
                first = await cli.post(
                    "/v1/chat/completions", json={"messages": [USER_TURN]}, headers=headers)
                second = await cli.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "a different question"}]},
                    headers=headers)
                second_body = await second.json()

        assert first.status == 200
        assert second.status == 409
        assert second_body["error"]["code"] == "idempotency_key_conflict"
        assert turn.calls == 1

    @pytest.mark.asyncio
    async def test_completed_body_replays_inside_the_window(self):
        """T6b: a sequential retry inside the memory TTL still gets the original body back."""
        adapter = _make_adapter(deadline=0)
        turn = _SlowTurn()
        turn.release.set()
        headers = {**AUTH, "Idempotency-Key": "sup53-key-replay"}
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with patch.object(adapter, "_run_agent", turn):
                first = await cli.post(
                    "/v1/chat/completions", json={"messages": [USER_TURN]}, headers=headers)
                second = await cli.post(
                    "/v1/chat/completions", json={"messages": [USER_TURN]}, headers=headers)
                second_body = await second.json()

        assert first.status == second.status == 200
        assert second_body["choices"][0]["message"]["content"] == "late answer"
        assert turn.calls == 1, "the replay must not run a second turn"

    @pytest.mark.asyncio
    async def test_no_idempotency_key_never_refuses_a_second_request(self):
        """T7 (control): headerless clients keep today's behaviour — two requests, two turns."""
        adapter = _make_adapter(deadline=0)
        turn = _SlowTurn()
        turn.release.set()
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with patch.object(adapter, "_run_agent", turn):
                first = await cli.post("/v1/chat/completions", json={"messages": [USER_TURN]}, headers=dict(AUTH))
                second = await cli.post("/v1/chat/completions", json={"messages": [USER_TURN]}, headers=dict(AUTH))

        assert first.status == second.status == 200
        assert turn.calls == 2

    @pytest.mark.asyncio
    async def test_malformed_idempotency_key_is_rejected(self):
        """T8: same bounds and code /v1/runs enforces."""
        adapter = _make_adapter(deadline=0)
        turn = _SlowTurn()
        turn.release.set()
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with patch.object(adapter, "_run_agent", turn):
                response = await cli.post(
                    "/v1/chat/completions", json={"messages": [USER_TURN]},
                    headers={**AUTH, "Idempotency-Key": "k" * 300})
                body = await response.json()

        assert response.status == 400
        assert body["error"]["code"] == "invalid_idempotency_key"
        assert turn.calls == 0


# ---------------------------------------------------------------------------
# T9 — the streaming path's keepalive is pinned, not assumed
# ---------------------------------------------------------------------------


class TestStreamingKeepalive:
    @pytest.mark.asyncio
    async def test_idle_stream_writes_keepalive_frames(self, monkeypatch):
        """T9: a silent turn emits ``: keepalive`` comments, so an idle-read timeout cannot fire."""
        monkeypatch.setattr(api_server_module, "CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS", 0.02)
        adapter = _make_adapter(deadline=0)
        turn = _SlowTurn()
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with patch.object(adapter, "_run_agent", turn):
                pending = asyncio.create_task(cli.post(
                    "/v1/chat/completions", json={"messages": [USER_TURN], "stream": True}, headers=dict(AUTH)))
                await turn.started.wait()
                # The SSE writer's idle poll is 0.5 s, so it needs at least one full poll
                # window before the keepalive threshold can be crossed.
                await asyncio.sleep(1.2)
                turn.release.set()
                response = await pending
                text = await response.text()

        assert response.status == 200
        assert ": keepalive" in text


# ---------------------------------------------------------------------------
# T10-T13 — review findings: settlement ownership, unknown outcomes, encoding
# ---------------------------------------------------------------------------


def _only_claim_row(adapter):
    rows = adapter._run_idempotency_store._conn.execute(
        "SELECT scope, idempotency_key, fingerprint, run_id, status_json, owner_pid "
        "FROM run_idempotency").fetchall()
    assert len(rows) == 1, rows
    return rows[0]


class TestReviewDisposition:
    @pytest.mark.asyncio
    async def test_cancelled_request_still_settles_the_claim(self):
        """Finding 1: a request cancelled before the deadline must not strand the claim row."""
        import contextlib

        adapter = _make_adapter(deadline=30)
        store = adapter._run_idempotency_store
        claim = {"id": "sync_cancel_row", "scope": "sc", "key": "k", "session_id": SESSION,
                 "route": "chat_completions"}
        store.reserve("sc", "k", "fp", claim["id"],
                      {"status": "running", "session_id": SESSION, "route": "chat_completions"},
                      owner_pid=0, owner_started=0)
        release = asyncio.Event()
        started = asyncio.Event()

        async def compute():
            started.set()
            await release.wait()
            return _reply("cancelled but finished")

        running = asyncio.create_task(adapter._run_sync_turn_under_deadline(
            compute, session_id=SESSION, claim=claim, log_label="t10"))
        await started.wait()
        running.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await running
        release.set()
        row = None
        for _ in range(200):
            row = _only_claim_row(adapter)
            if "running" not in row[4]:
                break
            await asyncio.sleep(0.01)

        assert '"completed"' in row[4], row[4]

    @pytest.mark.asyncio
    async def test_dead_owner_claim_reports_unknown_not_completed(self):
        """Finding 2: an owner that died mid-turn must not be reported as a finished turn."""
        adapter = _make_adapter(deadline=0)
        turn = _SlowTurn()
        turn.release.set()
        headers = {**AUTH, "Idempotency-Key": "sup53-dead-owner"}
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            with patch.object(adapter, "_run_agent", turn):
                first = await cli.post("/v1/chat/completions",
                                       json={"messages": [USER_TURN]}, headers=headers)
                assert first.status == 200
                scope, key, fingerprint, claim_id, _status, _pid = _only_claim_row(adapter)
                adapter._run_idempotency_store._conn.execute(
                    "UPDATE run_idempotency SET status_json=?, owner_pid=?, owner_started=0 "
                    "WHERE run_id=?",
                    ('{"status":"running","session_id":"' + SESSION + '"}', 999999, claim_id))
                adapter._run_idempotency_store._conn.commit()
                # Drop the in-process replay slot so the DURABLE claim is what answers.
                api_server_module._idem_cache._store.clear()
                second = await cli.post("/v1/chat/completions",
                                        json={"messages": [USER_TURN]}, headers=headers)
                body = await second.json()

        assert second.status == 409
        assert body["error"]["code"] == "duplicate_turn_interrupted"
        assert "NEW Idempotency-Key" in body["error"]["message"]
        assert '"interrupted"' in _only_claim_row(adapter)[4]

    @pytest.mark.asyncio
    async def test_late_failed_turn_is_not_counted_as_a_completion(self):
        """Finding 4: a turn that ends flagged failed is settled failed and not counted."""
        adapter = _make_adapter(deadline=0.05)
        release = asyncio.Event()
        started = asyncio.Event()

        async def compute():
            started.set()
            await release.wait()
            return {"final_response": "", "failed": True, "messages": []}, {"total_tokens": 0}

        claim = {"id": "sync_failed_row", "scope": "sc", "key": "k2", "session_id": SESSION,
                 "route": "chat_completions"}
        adapter._run_idempotency_store.reserve("sc", "k2", "fp2", claim["id"],
                                               {"status": "running", "session_id": SESSION},
                                               owner_pid=0, owner_started=0)
        with pytest.raises(Exception):
            await adapter._run_sync_turn_under_deadline(
                compute, session_id=SESSION, claim=claim, log_label="t12")
        release.set()
        row = None
        for _ in range(200):
            row = _only_claim_row(adapter)
            if "running" not in row[4]:
                break
            await asyncio.sleep(0.01)

        assert '"failed"' in row[4], row[4]
        assert adapter._late_completions_today == 0

    @pytest.mark.asyncio
    async def test_poll_path_encodes_the_session_id_as_one_segment(self):
        """Finding 6: an accepted-but-URL-unsafe session id cannot change the poll target."""
        adapter = _make_adapter(deadline=0)
        response = adapter._sync_turn_deadline_response("a b?c#d", 840)
        body = json.loads(response.text)

        assert body["hermes"]["poll"]["path"] == "/api/sessions/a%20b%3Fc%23d/messages"

