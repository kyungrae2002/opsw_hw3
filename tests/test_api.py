from fastapi.testclient import TestClient

from app.main import MAX_CHARS, app

client = TestClient(app)


def test_index_page():
    res = client.get("/")
    assert res.status_code == 200
    assert "ReviewMood" in res.text


def test_docs_page():
    assert client.get("/docs").status_code == 200


def test_health_reports_model_and_version():
    data = client.get("/api/health").json()
    assert data["status"] == "ok"
    assert data["model"]["loaded"] is True
    assert data["version"] and data["commit"]


def test_analyze_positive():
    res = client.post("/api/analyze", json={"text": "Absolutely love it. Best purchase I made this year!"})
    assert res.status_code == 200
    data = res.json()
    assert data["label"] == "positive"
    assert 0.5 <= data["score"] <= 1
    assert abs(data["scores"]["positive"] + data["scores"]["negative"] - 1) < 1e-6
    # 무중단 배포 검증용 메타데이터
    assert data["version"] and data["commit"] and data["model"]


def test_analyze_negative():
    res = client.post("/api/analyze", json={"text": "Broke after one day. Awful product, do not buy."})
    assert res.status_code == 200
    assert res.json()["label"] == "negative"


def test_long_text_is_truncated_not_rejected():
    res = client.post("/api/analyze", json={"text": "nice " * 900})
    assert res.status_code == 200
    assert res.json()["truncated"] is True


def test_empty_text_rejected():
    res = client.post("/api/analyze", json={"text": "   "})
    assert res.status_code == 400


def test_too_long_text_rejected():
    res = client.post("/api/analyze", json={"text": "a" * (MAX_CHARS + 1)})
    assert res.status_code == 413


def test_missing_field_rejected():
    assert client.post("/api/analyze", json={}).status_code == 422


def test_wrong_type_rejected():
    assert client.post("/api/analyze", json={"text": 123}).status_code == 422


def test_non_json_rejected():
    res = client.post("/api/analyze", content=b"hello", headers={"Content-Type": "text/plain"})
    assert res.status_code == 422
