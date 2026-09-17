from datetime import datetime, timezone

from ai_newsletter.collector import (
    are_titles_similar,
    canonicalize_url,
    clean_title_for_comparison,
    collect_from_entries,
    extract_title_tokens,
    normalize_title,
)
from ai_newsletter.models import Article, FeedEntry


def test_clean_title():
    t1 = "[단독] OpenAI, 차세대 AI 모델 GPT-5 개발 착수 - AI타임스"
    cleaned = clean_title_for_comparison(t1)
    assert "단독" not in cleaned
    assert "ai타임스" not in cleaned
    assert "openai" in cleaned


def test_are_titles_similar():
    t1 = "OpenAI announces GPT-4o with voice and vision"
    t2 = "OpenAI launches GPT-4o with voice and vision features"
    assert are_titles_similar(t1, t2) is True

    t3 = "NVIDIA reveals Blackwell B200 GPU architecture"
    assert are_titles_similar(t1, t3) is False


def test_canonicalize_url():
    url = "https://techcrunch.com/2026/09/15/openai-gpt4o/?utm_source=twitter&utm_campaign=ai&ref=feed"
    clean = canonicalize_url(url)
    assert "utm_source" not in clean
    assert "utm_campaign" not in clean
    assert "ref" not in clean
    assert clean == "https://techcrunch.com/2026/09/15/openai-gpt4o"


def test_collect_from_entries_deduplication():
    entries = [
        FeedEntry(
            title="OpenAI launches GPT-4o multimodal model",
            url="https://techcrunch.com/openai-gpt4o?utm_source=rss",
            source="TechCrunch",
            bucket="frontier_models",
        ),
        FeedEntry(
            title="OpenAI launches GPT-4o multimodal model",
            url="https://techcrunch.com/openai-gpt4o?utm_source=twitter",
            source="TechCrunch",
            bucket="frontier_models",
        ),
        FeedEntry(
            title="NVIDIA announces new Blackwell AI chips",
            url="https://theverge.com/nvidia-blackwell",
            source="The Verge",
            bucket="hardware_infra",
        ),
    ]

    articles, warnings, suppressed = collect_from_entries(entries)
    assert len(articles) == 2, f"Expected 2 articles after dedupe, got {len(articles)}"
    assert suppressed == 1


# ── Production Feed Health & Resilience Tests ─────────────────────────────────

import httpx
import pytest

from ai_newsletter.collector import check_feed_health, fetch_source_feed
from ai_newsletter.config import Settings
from ai_newsletter.sources import SourceFeed


def test_fetch_source_feed_malformed_rss(monkeypatch):
    """Malformed XML/RSS returns empty entries and parsing error without crashing."""
    feed = SourceFeed(name="Bad XML Feed", bucket="frontier_models", url="https://example.com/bad.xml")
    settings = Settings(request_timeout_seconds=2)

    class MockResponse:
        status_code = 200
        content = b"<<<invalid xml>>>>>"
        def raise_for_status(self):
            pass

    class MockClient:
        def get(self, *args, **kwargs):
            return MockResponse()

    res = fetch_source_feed(feed, settings, MockClient())
    assert len(res.entries) == 0
    assert res.error is not None
    assert "파싱 실패" in res.error
    assert res.http_status == 200


def test_fetch_source_feed_empty_feed(monkeypatch):
    """Empty but valid RSS feed returns empty entries without error."""
    feed = SourceFeed(name="Empty Feed", bucket="frontier_models", url="https://example.com/empty.xml")
    settings = Settings(request_timeout_seconds=2)

    class MockResponse:
        status_code = 200
        content = b'<?xml version="1.0"?><rss version="2.0"><channel><title>Empty</title></channel></rss>'
        def raise_for_status(self):
            pass

    class MockClient:
        def get(self, *args, **kwargs):
            return MockResponse()

    res = fetch_source_feed(feed, settings, MockClient())
    assert len(res.entries) == 0
    assert res.error is None
    assert res.http_status == 200


def test_fetch_source_feed_timeout(monkeypatch):
    """Network timeout returns error and does not raise exception."""
    feed = SourceFeed(name="Timeout Feed", bucket="frontier_models", url="https://example.com/timeout.xml")
    settings = Settings(request_timeout_seconds=1)

    class MockClient:
        def get(self, *args, **kwargs):
            raise httpx.ReadTimeout("Read timed out")

    res = fetch_source_feed(feed, settings, MockClient())
    assert len(res.entries) == 0
    assert "Read timed out" in res.error
    assert res.http_status is None


def test_fetch_source_feed_http_error(monkeypatch):
    """HTTP 404 or 500 error returns appropriate status and message."""
    feed = SourceFeed(name="404 Feed", bucket="frontier_models", url="https://example.com/404.xml")
    settings = Settings(request_timeout_seconds=2)

    req = httpx.Request("GET", "https://example.com/404.xml")
    resp = httpx.Response(404, request=req)

    class MockClient:
        def get(self, *args, **kwargs):
            raise httpx.HTTPStatusError("Client error '404 Not Found'", request=req, response=resp)

    res = fetch_source_feed(feed, settings, MockClient())
    assert len(res.entries) == 0
    assert res.http_status == 404
    assert "HTTP 404" in res.error


def test_check_feed_health_classification(monkeypatch):
    """check_feed_health correctly assigns healthy, degraded, failing, disabled states."""
    f_healthy = SourceFeed(name="Good Feed", bucket="frontier_models", url="https://example.com/good.xml", evidence_level="primary")
    f_disabled = SourceFeed(name="Disabled Feed", bucket="frontier_models", url="https://example.com/off.xml", enabled=False, evidence_level="independent")
    f_error = SourceFeed(name="Error Feed", bucket="frontier_models", url="https://example.com/err.xml", evidence_level="industry_media")

    settings = Settings(request_timeout_seconds=2)

    def mock_fetch(feed, settings, client):
        from ai_newsletter.collector import FeedFetchResult
        if feed.id == "good_feed":
            return FeedFetchResult(entries=[FeedEntry(title="T", url="U", source="Good Feed", bucket="frontier_models")], response_time=0.2)
        elif feed.id == "disabled_feed":
            return FeedFetchResult(entries=[], response_time=0.0)
        else:
            return FeedFetchResult(entries=[], error="Connection refused", response_time=0.5)

    monkeypatch.setattr("ai_newsletter.collector.fetch_source_feed", mock_fetch)

    results = check_feed_health([f_healthy, f_disabled, f_error], settings=settings)
    r_map = {r["id"]: r for r in results}

    assert r_map["good_feed"]["health_state"] == "healthy"
    assert r_map["good_feed"]["source_role"] == "primary"
    assert r_map["disabled_feed"]["health_state"] == "disabled"
    assert r_map["disabled_feed"]["source_role"] == "independent"
    assert r_map["error_feed"]["health_state"] == "failing"
    assert r_map["error_feed"]["consecutive_failures"] == 1


def test_partial_source_failure_isolation(monkeypatch):
    """When some feeds fail and others succeed, collect_and_store continues and collects from healthy feeds."""
    from ai_newsletter.collector import collect_and_store
    from ai_newsletter.store import NewsletterStore

    f1 = SourceFeed(name="OpenAI News", bucket="frontier_models", url="https://openai.com/feed", evidence_level="primary")
    f2 = SourceFeed(name="Failing Feed", bucket="frontier_models", url="https://fail.com/feed", evidence_level="industry_media")

    def mock_fetch(feed, settings, client):
        from ai_newsletter.collector import FeedFetchResult
        if feed.id == "openai_news":
            return FeedFetchResult(entries=[FeedEntry(title="OpenAI GPT-5 Launch", url="https://openai.com/gpt5", source="OpenAI News", bucket="frontier_models")])
        else:
            return FeedFetchResult(entries=[], error="500 Internal Server Error", http_status=500)

    monkeypatch.setattr("ai_newsletter.collector.fetch_source_feed", mock_fetch)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        store = NewsletterStore(tmp.name)
        issue = collect_and_store(store=store, sources=[f1, f2], target_date="2026-09-17")

        assert len(issue.articles) == 1
        assert issue.articles[0].title == "OpenAI GPT-5 Launch"
        assert issue.metrics["feeds_ok"] == 1
        assert issue.metrics["feeds_failed"] == 1
        assert issue.metrics["primary_feeds"] == 1
        assert issue.metrics["industry_media_feeds"] == 1
        assert issue.metrics["successful_feeds"] == 1
        assert issue.metrics["failed_feeds"] == 1
        assert any("Failing Feed" in w for w in issue.warnings)


def test_all_feeds_failing_does_not_crash(monkeypatch):
    """When all feeds fail, collect_and_store completes gracefully with 0 articles and warnings."""
    from ai_newsletter.collector import collect_and_store
    from ai_newsletter.store import NewsletterStore

    f1 = SourceFeed(name="Fail1", bucket="frontier_models", url="https://f1.com/feed")
    f2 = SourceFeed(name="Fail2", bucket="frontier_models", url="https://f2.com/feed")

    def mock_fetch(feed, settings, client):
        from ai_newsletter.collector import FeedFetchResult
        return FeedFetchResult(entries=[], error="Network unreachable")

    monkeypatch.setattr("ai_newsletter.collector.fetch_source_feed", mock_fetch)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        store = NewsletterStore(tmp.name)
        issue = collect_and_store(store=store, sources=[f1, f2], target_date="2026-09-17")

        assert len(issue.articles) == 0
        assert issue.metrics["feeds_ok"] == 0
        assert issue.metrics["feeds_failed"] == 2
        assert len(issue.warnings) == 2


