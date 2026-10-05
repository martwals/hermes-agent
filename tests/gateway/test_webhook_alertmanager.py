"""Tests for Alertmanager-format array payloads (SUP-29).

VictoriaMetrics vmalert sends the Alertmanager webhook format — a JSON array of
alert objects — rather than the single-object payloads Grafana uses.  The
webhook handler must accept both without raising AttributeError.
"""

import asyncio
import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(routes=None, **kwargs):
    extra = {"host": "0.0.0.0", "port": 0, "routes": routes or {}}
    extra.update(kwargs)
    return WebhookAdapter(PlatformConfig(enabled=True, extra=extra))


def _make_config(routes=None, **kwargs):
    extra = {"host": "0.0.0.0", "port": 0, "routes": routes or {}, **kwargs}
    return PlatformConfig(enabled=True, extra=extra)


# A representative vmalert / Alertmanager array payload (2 alerts in a group)
ALERTMANAGER_ARRAY_PAYLOAD = {
    "version": "0",
    "status": "firing",
    "groupLabels": {"alertname": "SystemdUnitFailed"},
    "commonLabels": {"alertname": "SystemdUnitFailed"},
    "commonAnnotations": {},
    "externalURL": "",
    "alerts": [
        {
            "status": "firing",
            "labels": {
                "alertname": "SystemdUnitFailed",
                "instance": "hardy:9100",
                "job": "node-exporter",
                "service": "docker",
            },
            "annotations": {
                "description": "Service docker is in failed state",
                "summary": "Systemd unit docker.service failed",
            },
            "startsAt": "2026-10-01T10:00:00Z",
            "endsAt": "0001-01-01T00:00:00Z",
            "generatorURL": "",
            "fingerprint": "abc123",
        },
        {
            "status": "firing",
            "labels": {
                "alertname": "HighMemoryUsage",
                "instance": "hardy:9100",
                "job": "node-exporter",
            },
            "annotations": {
                "description": "Memory usage above 90%",
                "summary": "High memory usage on hardy",
            },
            "startsAt": "2026-10-01T10:05:00Z",
            "endsAt": "0001-01-01T00:00:00Z",
            "generatorURL": "",
            "fingerprint": "def456",
        },
    ],
}

# Single-object Grafana-style payload (should still work)
GRAFANA_SINGLE_PAYLOAD = {
    "title": "Test Alert",
    "message": "Something happened",
    "status": "alerting",
    "event_type": "grafana-alert",
}


# ===================================================================
# Test: _safe_payload helper
# ===================================================================

class TestSafePayload:

    def test_dict_passthrough(self):
        d = {"key": "val"}
        assert WebhookAdapter._safe_payload(d) is d

    def test_alertmanager_first(self):
        arr = [{"alert": "one"}, {"alert": "two"}]
        result = WebhookAdapter._safe_payload(arr)
        assert result == {"alert": "one"}

    def test_empty_list(self):
        assert WebhookAdapter._safe_payload([]) == {}

    def test_non_dict_first_element(self):
        assert WebhookAdapter._safe_payload(["string"]) == {}

    def test_non_iterable(self):
        assert WebhookAdapter._safe_payload(42) == {}


# ===================================================================
# Test: Single-object Grafana payload still works (regression)
# ===================================================================

class TestGrafanaSinglePayload:

    @pytest.mark.asyncio
    async def test_grafana_single_object_still_dispatches(self):
        """A single-object Grafana payload should still produce 202 and dispatch."""
        routes = {
            "grafana": {
                "secret": _INSECURE_NO_AUTH,
                "events": ["grafana-alert"],
                "prompt": "Grafana alert: {title}\n{message}",
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)
        app = web.Application(client_max_size=1048576)
        app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
        client = TestClient(TestServer(app))

        async with client:
            resp = await client.post(
                "/webhooks/grafana",
                data=json.dumps(GRAFANA_SINGLE_PAYLOAD),
            )
            assert resp.status == 202
            body = await resp.json()
            assert body["status"] == "accepted"


# ===================================================================
# Test: Alertmanager array payload does not crash (SUP-29)
# ===================================================================

class TestAlertmanagerArrayPayload:

    @pytest.mark.asyncio
    async def test_array_payload_no_crash(self):
        """POSTing an Alertmanager array payload must NOT raise AttributeError
        and should return 202 Accepted — the full array is passed to the agent."""
        routes = {
            "vmalert": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "vmalert fired:\n{__raw__}",
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)
        app = web.Application(client_max_size=1048576)
        app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
        client = TestClient(TestServer(app))

        async with client:
            resp = await client.post(
                "/webhooks/vmalert",
                data=json.dumps(ALERTMANAGER_ARRAY_PAYLOAD),
            )
            assert resp.status == 202
            body = await resp.json()
            assert body["status"] == "accepted"

    @pytest.mark.asyncio
    async def test_array_payload_dispatches_message(self):
        """The full array payload (not just the first element) must reach the agent
        so the rendered prompt contains all firing alerts."""
        routes = {
            "vmalert": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "vmalert fired:\n{__raw__}",
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)
        app = web.Application(client_max_size=1048576)
        app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
        client = TestClient(TestServer(app))

        async with client:
            with patch.object(adapter, 'handle_message') as mock_handle:
                mock_handle.return_value = asyncio.Future()
                mock_handle.return_value.set_result(None)

                resp = await client.post(
                    "/webhooks/vmalert",
                    data=json.dumps(ALERTMANAGER_ARRAY_PAYLOAD),
                )
                assert resp.status == 202

                # handle_message should have been called once (one POST = one run)
                assert mock_handle.called
                event: MessageEvent = mock_handle.call_args[0][0]
                # The raw_message should be the ORIGINAL array, not the first element
                assert event.raw_message == ALERTMANAGER_ARRAY_PAYLOAD
                # The rendered prompt should contain the full JSON
                prompt_text = event.text
                assert "alerts" in prompt_text
                assert "SystemdUnitFailed" in prompt_text
                assert "HighMemoryUsage" in prompt_text

    @pytest.mark.asyncio
    async def test_empty_array_returns_202(self):
        """An empty alerts array [] should not crash — return 202."""
        routes = {
            "vmalert": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "vmalert:\n{__raw__}",
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)
        app = web.Application(client_max_size=1048576)
        app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
        client = TestClient(TestServer(app))

        async with client:
            resp = await client.post(
                "/webhooks/vmalert",
                data=json.dumps({"alerts": []}),
            )
            assert resp.status == 202


# ===================================================================
# Test: Mixed single-element array (realistic vmalert for 1 alert)
# ===================================================================

class TestAlertmanagerSingleElementArray:

    @pytest.mark.asyncio
    async def test_single_element_array(self):
        """vmalert may send an array with a single alert — still an array."""
        single_alert = {
            "alerts": [
                {
                    "status": "firing",
                    "labels": {"alertname": "HostDown", "instance": "node1:9100"},
                    "annotations": {"summary": "Host is down"},
                    "fingerprint": "abc",
                }
            ]
        }
        routes = {
            "vmalert": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "vmalert:\n{__raw__}",
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)
        app = web.Application(client_max_size=1048576)
        app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
        client = TestClient(TestServer(app))

        async with client:
            with patch.object(adapter, 'handle_message') as mock_handle:
                mock_handle.return_value = asyncio.Future()
                mock_handle.return_value.set_result(None)

                resp = await client.post(
                    "/webhooks/vmalert",
                    data=json.dumps(single_alert),
                )
                assert resp.status == 202
                event: MessageEvent = mock_handle.call_args[0][0]
                assert event.raw_message == single_alert
                assert "HostDown" in event.text
