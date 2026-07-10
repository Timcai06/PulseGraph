import os
import tempfile

from fastapi.testclient import TestClient

from app.security.local_boundary import TRUSTED_EXECUTION_HEADER, TRUSTED_EXECUTION_VALUE

# Keep run persistence out of the repo during tests; must be set before app import.
os.environ.setdefault("PULSEGRAPH_RUNS_DIR", tempfile.mkdtemp(prefix="pulsegraph-test-runs-"))

_test_client_init = TestClient.__init__


def _trusted_test_client_init(self, *args, **kwargs):
    headers = dict(kwargs.pop("headers", {}) or {})
    headers.setdefault(TRUSTED_EXECUTION_HEADER, TRUSTED_EXECUTION_VALUE)
    _test_client_init(self, *args, headers=headers, **kwargs)


TestClient.__init__ = _trusted_test_client_init
