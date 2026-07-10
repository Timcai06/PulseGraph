from fastapi.testclient import TestClient

from app.main import app
from app.security.local_boundary import TRUSTED_EXECUTION_HEADER


client = TestClient(app)


def test_rejects_non_local_browser_origin() -> None:
    response = client.get("/health", headers={"Origin": "https://example.com"})

    assert response.status_code == 403
    assert response.json()["detail"] == "PulseGraph rejected a non-local browser origin."


def test_accepts_loopback_browser_origin() -> None:
    response = client.get("/health", headers={"Origin": "http://127.0.0.1:5174"})

    assert response.status_code == 200


def test_rejects_non_loopback_client() -> None:
    remote_client = TestClient(app, client=("192.0.2.10", 50000))

    response = remote_client.get("/health")

    assert response.status_code == 403
    assert response.json()["detail"] == "PulseGraph only accepts local connections."


def test_python_execution_requires_explicit_trust_header() -> None:
    response = client.post(
        "/api/inspect/resource/preview",
        headers={TRUSTED_EXECUTION_HEADER: ""},
        files={"files": ("resource.py", b"class TrainingResource: pass", "text/x-python")},
        data={"entry_file": "resource.py"},
    )

    assert response.status_code == 428
    assert "executes local Python" in response.json()["detail"]
