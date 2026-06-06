from __future__ import annotations

import time
from urllib.error import URLError
from urllib.request import Request, urlopen

_DEFAULT_PATHS = ("/health", "/v1/models")


def wait_for_http_ready(
    base_url: str,
    *,
    timeout: float = 600.0,
    interval: float = 5.0,
    paths: tuple[str, ...] = _DEFAULT_PATHS,
) -> None:
    """Poll common HTTP health paths until one succeeds or timeout expires."""
    normalized = base_url.rstrip("/")
    deadline = time.monotonic() + timeout
    last_error: str | None = None

    while time.monotonic() < deadline:
        for path in paths:
            url = f"{normalized}{path}"
            try:
                request = Request(url, method="GET")
                with urlopen(request, timeout=5) as response:
                    if 200 <= response.status < 300:
                        return
                last_error = f"{url} returned HTTP {response.status}"
            except URLError as exc:
                last_error = f"{url}: {exc.reason}"

        time.sleep(interval)

    raise TimeoutError(
        f"timed out waiting for {normalized} to become ready"
        + (f" (last error: {last_error})" if last_error else "")
    )
