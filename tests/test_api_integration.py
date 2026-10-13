"""Integration tests for the Accident-Intel API (all mocked — no GPU needed)."""
import io


# ---------------------------------------------------------------------------
# 1. Health
# ---------------------------------------------------------------------------

def test_health(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# 2. Upload video
# ---------------------------------------------------------------------------

def test_upload_video(test_client):
    dummy_content = b"\x00"  # 1-byte dummy file
    files = {"file": ("test.mp4", io.BytesIO(dummy_content), "video/mp4")}
    response = test_client.post("/api/accidents/upload", files=files)
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert "status" in data


# ---------------------------------------------------------------------------
# 3. Get status
# ---------------------------------------------------------------------------

def test_get_status(test_client):
    # Create an accident first
    files = {"file": ("test.mp4", io.BytesIO(b"\x00"), "video/mp4")}
    create_resp = test_client.post("/api/accidents/upload", files=files)
    assert create_resp.status_code == 201
    accident_id = create_resp.json()["id"]

    response = test_client.get(f"/api/accidents/{accident_id}/status")
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "status" in data
    assert "error" in data


# ---------------------------------------------------------------------------
# 4. Trigger analyze
# ---------------------------------------------------------------------------

def test_trigger_analyze(test_client):
    files = {"file": ("test.mp4", io.BytesIO(b"\x00"), "video/mp4")}
    create_resp = test_client.post("/api/accidents/upload", files=files)
    assert create_resp.status_code == 201
    accident_id = create_resp.json()["id"]

    response = test_client.post(f"/api/accidents/{accident_id}/analyze")
    # Either 200 (started) or 400 (no valid video — 1-byte file). Both are acceptable.
    assert response.status_code in (200, 400)
    if response.status_code == 200:
        assert "status" in response.json()


# ---------------------------------------------------------------------------
# 5. List accidents
# ---------------------------------------------------------------------------

def test_list_accidents(test_client):
    response = test_client.get("/api/accidents/")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "limit" in data


# ---------------------------------------------------------------------------
# 6. Delete accident
# ---------------------------------------------------------------------------

def test_delete_accident(test_client):
    files = {"file": ("test.mp4", io.BytesIO(b"\x00"), "video/mp4")}
    create_resp = test_client.post("/api/accidents/upload", files=files)
    assert create_resp.status_code == 201
    accident_id = create_resp.json()["id"]

    response = test_client.delete(f"/api/accidents/{accident_id}")
    assert response.status_code == 204

    # Confirm it's gone
    get_resp = test_client.get(f"/api/accidents/{accident_id}")
    assert get_resp.status_code == 404


# ---------------------------------------------------------------------------
# 7. VLM alias
# ---------------------------------------------------------------------------

def test_vlm_alias(test_client):
    files = {"file": ("test.mp4", io.BytesIO(b"\x00"), "video/mp4")}
    create_resp = test_client.post("/api/accidents/upload", files=files)
    accident_id = create_resp.json()["id"]

    response = test_client.get(f"/api/accidents/{accident_id}/vlm")
    # 200 if VLM result is available, 404 if not yet — never 405
    assert response.status_code in (200, 404)


# ---------------------------------------------------------------------------
# 8. Liability alias
# ---------------------------------------------------------------------------

def test_liability_alias(test_client):
    files = {"file": ("test.mp4", io.BytesIO(b"\x00"), "video/mp4")}
    create_resp = test_client.post("/api/accidents/upload", files=files)
    accident_id = create_resp.json()["id"]

    response = test_client.get(f"/api/accidents/{accident_id}/liability")
    # 200 (report=null) or 404 — never 405
    assert response.status_code in (200, 404)
