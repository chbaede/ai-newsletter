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


def test_mail_settings_modal_rendered(client):
    res = client.get("/")
    assert res.status_code == 200
    # Must have the mail settings trigger buttons
    assert "data-open-mail-settings" in res.text
    # Must have the modal backdrop container
    assert 'id="mailSettingsModal"' in res.text
    assert "data-mail-settings-form" in res.text
    # Must NOT have static bottom settings panel
    assert 'data-view-panel="settings"' not in res.text


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


# ── Hardened Admin Auth & Health Endpoint Tests ───────────────────────────────


def test_admin_auth_configured_with_correct_key(tmp_path):
    """1. ADMIN_KEY configured + correct key allows admin access via header or bearer token."""
    store = NewsletterStore(tmp_path / "auth1.db")
    settings = Settings(db_path=tmp_path / "auth1.db", admin_key="secret123", app_env="production")
    app = create_app(store=store, settings=settings)
    c = TestClient(app)

    # Via X-Admin-Key header
    res_hdr = c.get("/api/admin/verify", headers={"X-Admin-Key": "secret123"})
    assert res_hdr.status_code == 200
    assert res_hdr.json() == {"ok": True, "admin": True}

    res_mail = c.get("/api/mail-settings", headers={"X-Admin-Key": "secret123"})
    assert res_mail.status_code == 200

    # Via Authorization: Bearer token
    res_bearer = c.get("/api/admin/verify", headers={"Authorization": "Bearer secret123"})
    assert res_bearer.status_code == 200
    assert res_bearer.json() == {"ok": True, "admin": True}


def test_admin_auth_configured_with_incorrect_key(tmp_path):
    """2. ADMIN_KEY configured + incorrect key rejects access."""
    store = NewsletterStore(tmp_path / "auth2.db")
    settings = Settings(db_path=tmp_path / "auth2.db", admin_key="secret123", app_env="production")
    app = create_app(store=store, settings=settings)
    c = TestClient(app)

    res = c.get("/api/admin/verify", headers={"X-Admin-Key": "wrong_password"})
    assert res.status_code == 200
    assert res.json() == {"ok": True, "admin": False}

    res_mail = c.get("/api/mail-settings", headers={"X-Admin-Key": "wrong_password"})
    assert res_mail.status_code == 401
    assert res_mail.json()["ok"] is False

    res_post = c.post("/api/collect", headers={"X-Admin-Key": "wrong_password"})
    assert res_post.status_code == 401


def test_admin_auth_configured_with_missing_key(tmp_path):
    """3. ADMIN_KEY configured + missing key rejects access."""
    store = NewsletterStore(tmp_path / "auth3.db")
    settings = Settings(db_path=tmp_path / "auth3.db", admin_key="secret123", app_env="production")
    app = create_app(store=store, settings=settings)
    c = TestClient(app)

    res = c.get("/api/admin/verify")
    assert res.status_code == 200
    assert res.json() == {"ok": True, "admin": False}

    res_mail = c.get("/api/mail-settings")
    assert res_mail.status_code == 401

    res_health = c.get("/api/sources/health")
    assert res_health.status_code == 401


def test_admin_auth_production_missing_key_fails_closed(tmp_path):
    """4. Production configuration + missing ADMIN_KEY fails closed (admin endpoints unauthorized)."""
    store = NewsletterStore(tmp_path / "auth4.db")
    settings = Settings(db_path=tmp_path / "auth4.db", admin_key=None, app_env="production", allow_anonymous_admin=False)
    app = create_app(store=store, settings=settings)
    c = TestClient(app)

    # verify endpoint
    res_verify = c.get("/api/admin/verify")
    assert res_verify.status_code == 200
    assert res_verify.json() == {"ok": True, "admin": False}

    # admin endpoints fail closed
    assert c.get("/api/mail-settings").status_code == 401
    assert c.post("/api/mail-settings", json={"smtp_host": "example.com"}).status_code == 401
    assert c.post("/api/collect").status_code == 401
    assert c.post("/api/issues/2026-09-17/send").status_code == 401
    assert c.get("/api/sources/health").status_code == 401


def test_admin_auth_development_test_configuration(tmp_path):
    """5. Development/test configuration with allow_anonymous_admin=True permits unauthenticated admin access."""
    store = NewsletterStore(tmp_path / "auth5.db")
    settings = Settings(db_path=tmp_path / "auth5.db", admin_key=None, app_env="development", allow_anonymous_admin=True)
    app = create_app(store=store, settings=settings)
    c = TestClient(app)

    res_verify = c.get("/api/admin/verify")
    assert res_verify.status_code == 200
    assert res_verify.json() == {"ok": True, "admin": True}

    res_mail = c.get("/api/mail-settings")
    assert res_mail.status_code == 200


def test_health_check_healthy_database(tmp_path):
    """6. Healthy database returns status 200, status=healthy, and latest issue info."""
    store = NewsletterStore(tmp_path / "health_ok.db")
    settings = Settings(db_path=tmp_path / "health_ok.db")
    article = Article(
        title="Sample News",
        url="https://example.com/1",
        source="TechNews",
        category="frontier_models",
        summary_ko="샘플 기사입니다.",
    )
    store.save_issue(NewsletterIssue(issue_date="2026-09-17", articles=[article]))
    app = create_app(store=store, settings=settings)
    c = TestClient(app)

    res = c.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["database"] == "ok"
    assert data["latest_issue_date"] == "2026-09-17"
    assert data["articles_count"] == 1


def test_health_check_database_failure_and_degraded_conditions(tmp_path):
    """7. Database failure returns 503 / unhealthy, and missing tables return degraded."""
    import sqlite3

    # Degraded condition: database missing required table
    degraded_db = tmp_path / "degraded.db"
    conn = sqlite3.connect(str(degraded_db))
    conn.execute("create table schema_version (version integer primary key);")
    conn.execute("create table issues (id integer primary key);")
    # articles and events tables are intentionally missing
    conn.commit()
    conn.close()

    # Create store bypassing init migration to test degraded state
    store_deg = object.__new__(NewsletterStore)
    store_deg.db_path = degraded_db
    app_deg = create_app(store=store_deg, settings=Settings(db_path=degraded_db))
    c_deg = TestClient(app_deg)

    res_deg = c_deg.get("/api/health")
    assert res_deg.status_code == 200
    data_deg = res_deg.json()
    assert data_deg["status"] == "degraded"
    assert data_deg["database"] == "degraded"
    assert "Missing tables" in data_deg["detail"]

    # Unhealthy condition: database file path is invalid or unopenable directory
    bad_db_path = tmp_path / "nonexistent_dir" / "cannot_create.db"
    store_bad = object.__new__(NewsletterStore)
    store_bad.db_path = bad_db_path
    app_bad = create_app(store=store_bad, settings=Settings(db_path=bad_db_path))
    c_bad = TestClient(app_bad)

    res_bad = c_bad.get("/api/health")
    assert res_bad.status_code == 503
    data_bad = res_bad.json()
    assert data_bad["status"] == "unhealthy"
    assert data_bad["database"] == "error"
    assert data_bad["latest_issue_date"] is None


def test_main_hub_links_and_priority_filter_attributes(client):
    res = client.get("/issues/2026-09-17")
    assert res.status_code == 200
    html = res.text

    # Main Hub link present
    assert 'https://main.yocto.co.kr/' in html
    assert 'class="hub-link-action"' in html
    assert 'class="rail-hub-btn"' in html

    # Priority filter metric cards present
    assert 'data-priority-filter="all"' in html
    assert 'data-priority-filter="critical"' in html
    assert 'data-priority-filter="high"' in html

    # Priority attributes on cards and table rows
    assert 'data-priority=' in html


def test_card_metadata_unification_and_table_filters(client):
    res = client.get("/issues/2026-09-17")
    assert res.status_code == 200
    html = res.text

    # Unified card meta box present
    assert 'class="card-meta-box"' in html
    assert '출처:' in html
    assert '핵심 분야:' in html
    assert '증거:' in html
    assert '신뢰도:' in html

    # Table filter toolbar elements present
    assert 'id="tableFilterCategory"' in html
    assert 'id="tableFilterSource"' in html
    assert 'id="tableFilterConfidence"' in html
    assert 'id="tablePageSize"' in html
    assert 'id="tableFilterReset"' in html

    # Sortable headers present
    assert 'data-sort="priority"' in html
    assert 'data-sort="category"' in html
    assert 'data-sort="confidence"' in html
    assert 'data-sort="source"' in html

    # Pagination elements present
    assert 'id="tablePagination"' in html
    assert 'id="tablePaginationControls"' in html





