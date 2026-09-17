import pytest
from fastapi.testclient import TestClient

from ai_newsletter.config import Settings
from ai_newsletter.models import Article, NewsletterIssue
from ai_newsletter.store import NewsletterStore
from ai_newsletter.web import create_app


@pytest.fixture
def client(tmp_path):
    store = NewsletterStore(tmp_path / "web_test.db")
    settings = Settings(db_path=tmp_path / "web_test.db", admin_key="test_secret")

    # Add sample issue
    article = Article(
        title="OpenAI announces GPT-4o",
        title_ko="오픈AI, GPT-4o 발표",
        title_en="OpenAI Announces GPT-4o",
        url="https://openai.com",
        source="OpenAI",
        category="frontier_models",
        summary_ko="새로운 플래그십 AI 모델입니다.",
        summary_en="New flagship AI model.",
        priority_score=95.0,
    )
    issue = NewsletterIssue(issue_date="2026-09-17", articles=[article])
    store.save_issue(issue)

    app = create_app(store=store, settings=settings)
    return TestClient(app)


def test_home_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "AI 뉴스레터" in res.text
    assert "GPT-4o" in res.text


def test_issue_page(client):
    res = client.get("/issues/2026-09-17")
    assert res.status_code == 200
    assert "2026-09-17" in res.text


def test_health_check(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["latest_issue_date"] == "2026-09-17"


def test_admin_verify(client):
    # No auth
    res1 = client.get("/api/admin/verify")
    assert res1.json()["admin"] is False

    # With auth
    res2 = client.get("/api/admin/verify", headers={"X-Admin-Key": "test_secret"})
    assert res2.json()["admin"] is True


def test_mail_settings_api(client):
    # Unauthorized
    res1 = client.get("/api/mail-settings")
    assert res1.status_code == 401

    # Authorized
    res2 = client.get("/api/mail-settings", headers={"X-Admin-Key": "test_secret"})
    assert res2.status_code == 200
    assert "settings" in res2.json()


def test_default_port_and_env_override(monkeypatch):
    from ai_newsletter.config import load_settings
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("WEB_PORT", raising=False)
    s = load_settings()
    assert s.port == 8001

    monkeypatch.setenv("PORT", "8888")
    s2 = load_settings()
    assert s2.port == 8888


