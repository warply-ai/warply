from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("WARPLY_INTEGRATION") != "1",
    reason="Lambda/SkyPilot integration tests require WARPLY_INTEGRATION=1.",
)


def test_lambda_sglang_e2e_placeholder():
    pytest.skip("Real SkyPilot Lambda + SGLang/NIXL E2E wiring is documented but not implemented.")
