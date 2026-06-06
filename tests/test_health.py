from __future__ import annotations

from urllib.error import URLError

import pytest

from warply.runtime.health import wait_for_http_ready


class _FakeResponse:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_wait_for_http_ready_returns_on_first_success(monkeypatch):
    calls: list[str] = []

    def fake_urlopen(request, timeout=5):
        calls.append(request.full_url)
        if len(calls) == 2:
            return _FakeResponse()
        raise URLError("connection refused")

    monkeypatch.setattr("warply.runtime.health.urlopen", fake_urlopen)
    monkeypatch.setattr("warply.runtime.health.time.sleep", lambda _: None)

    wait_for_http_ready("http://router.example:8000", timeout=30, interval=1)

    assert calls == [
        "http://router.example:8000/health",
        "http://router.example:8000/v1/models",
    ]


def test_wait_for_http_ready_times_out(monkeypatch):
    def fake_urlopen(request, timeout=5):
        raise URLError("connection refused")

    monkeypatch.setattr("warply.runtime.health.urlopen", fake_urlopen)
    monkeypatch.setattr("warply.runtime.health.time.sleep", lambda _: None)
    monkeypatch.setattr("warply.runtime.health.time.monotonic", iter([0.0, 0.0, 11.0]).__next__)

    with pytest.raises(TimeoutError, match="timed out waiting"):
        wait_for_http_ready("http://router.example:8000", timeout=10, interval=5)
