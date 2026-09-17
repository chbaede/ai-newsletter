from datetime import datetime, timezone
import pytest

from ai_newsletter.models import Article, Event, EventArticle, NewsletterIssue
from ai_newsletter.store import NewsletterStore


@pytest.fixture
def test_store(tmp_path):
    db_file = tmp_path / "test_newsletter.db"
    return NewsletterStore(db_file)


def test_store_save_and_retrieve_issue(test_store):
    articles = [
        Article(
            title="OpenAI announces GPT-4o flagship model",
            title_ko="오픈AI, 플래그십 모델 GPT-4o 공개",
            title_en="OpenAI Announces GPT-4o Flagship Model",
            url="https://openai.com/index/gpt-4o",
            source="OpenAI",
            category="frontier_models",
            summary_ko="오픈AI가 음성과 비전을 지원하는 GPT-4o를 공개했습니다.",
            summary_en="OpenAI announced GPT-4o with native multimodal features.",
            why_it_matters_ko="멀티모달 인터랙션의 지연시간을 획기적으로 개선했습니다.",
            why_it_matters_en="Dramatically reduces latency for multimodal interactions.",
            evidence_quality_score=96.0,
            priority_score=92.0,
            tags=["OpenAI", "LLM"],
        )
    ]
    events = [
        Event(
            event_id="ev_test1",
            title="OpenAI GPT-4o Launch",
            category="frontier_models",
            importance=92.0,
        )
    ]
    event_articles = [
        EventArticle(event_id="ev_test1", article_id=articles[0].article_id, relationship="primary")
    ]

    issue = NewsletterIssue(
        issue_date="2026-09-17",
        articles=articles,
        events=events,
        warnings=["Test note"],
    )

    test_store.save_issue(issue, event_articles=event_articles)

    retrieved = test_store.get_issue("2026-09-17")
    assert retrieved is not None
    assert retrieved.issue_date == "2026-09-17"
    assert len(retrieved.articles) == 1
    assert retrieved.articles[0].title_ko == "오픈AI, 플래그십 모델 GPT-4o 공개"
    assert retrieved.articles[0].title_en == "OpenAI Announces GPT-4o Flagship Model"
    assert retrieved.articles[0].summary_ko == "오픈AI가 음성과 비전을 지원하는 GPT-4o를 공개했습니다."
    assert retrieved.articles[0].summary_en == "OpenAI announced GPT-4o with native multimodal features."
    assert retrieved.articles[0].why_it_matters_ko == "멀티모달 인터랙션의 지연시간을 획기적으로 개선했습니다."
    assert retrieved.articles[0].why_it_matters_en == "Dramatically reduces latency for multimodal interactions."
    assert retrieved.articles[0].evidence_quality_score == 96.0
    assert len(retrieved.events) == 1
    assert retrieved.events[0].event_id == "ev_test1"


def test_store_mail_settings(test_store):
    test_store.update_mail_settings({"smtp_host": "smtp.gmail.com", "smtp_port": 587})
    settings = test_store.mail_settings()
    assert settings["smtp_host"] == "smtp.gmail.com"
    assert settings["smtp_port"] == "587"


def test_scheduler_runs(test_store):
    assert test_store.has_daily_run_completed("2026-09-17") is False
    acquired = test_store.acquire_daily_run("2026-09-17")
    assert acquired is True

    # Re-acquire while running should fail
    assert test_store.acquire_daily_run("2026-09-17") is False

    test_store.finish_daily_run("2026-09-17", metrics={"articles": 10})
    assert test_store.has_daily_run_completed("2026-09-17") is True


# ── Tests for recent_articles(days) time-based filtering ──────────────────────

from datetime import timedelta


def _make_store_article(title: str, published_at: datetime | None = None, collected_at: datetime | None = None) -> Article:
    return Article(
        title=title,
        url=f"https://example.com/{title.lower().replace(' ', '-')}",
        source="Test Source",
        category="frontier_models",
        summary_ko="요약",
        published_at=published_at,
        collected_at=collected_at,
    )


def test_recent_articles_within_3_days_included(test_store):
    """1. Article within 3 days -> included."""
    now = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
    a1 = _make_store_article("Article 2 Days Old", published_at=now - timedelta(days=2))
    issue = NewsletterIssue(issue_date="2026-09-15", articles=[a1])
    test_store.save_issue(issue)

    recent = test_store.recent_articles(days=3, now_dt=now)
    assert len(recent) == 1
    assert recent[0].title == "Article 2 Days Old"


def test_recent_articles_older_than_3_days_excluded(test_store):
    """2. Article older than 3 days -> excluded."""
    now = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
    a1 = _make_store_article("Article 4 Days Old", published_at=now - timedelta(days=4))
    issue = NewsletterIssue(issue_date="2026-09-13", articles=[a1])
    test_store.save_issue(issue)

    recent = test_store.recent_articles(days=3, now_dt=now)
    assert len(recent) == 0


def test_recent_articles_cutoff_boundary_deterministic(test_store):
    """3. Article exactly around the cutoff -> deterministic behavior (inclusive >= cutoff)."""
    now = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
    exact_cutoff = now - timedelta(days=3)
    just_before = exact_cutoff - timedelta(seconds=1)
    just_after = exact_cutoff + timedelta(seconds=1)

    a_exact = _make_store_article("Exact Cutoff", published_at=exact_cutoff)
    a_old = _make_store_article("Just Old", published_at=just_before)
    a_new = _make_store_article("Just New", published_at=just_after)

    issue = NewsletterIssue(issue_date="2026-09-14", articles=[a_old, a_exact, a_new])
    test_store.save_issue(issue)

    recent = test_store.recent_articles(days=3, now_dt=now)
    titles = [a.title for a in recent]
    assert "Just New" in titles
    assert "Exact Cutoff" in titles
    assert "Just Old" not in titles


def test_recent_articles_days_7_includes_older_than_3_days(test_store):
    """4. days=7 includes an article excluded by days=3."""
    now = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
    a_5days = _make_store_article("Article 5 Days Old", published_at=now - timedelta(days=5))
    issue = NewsletterIssue(issue_date="2026-09-12", articles=[a_5days])
    test_store.save_issue(issue)

    assert len(test_store.recent_articles(days=3, now_dt=now)) == 0
    assert len(test_store.recent_articles(days=7, now_dt=now)) == 1
    assert test_store.recent_articles(days=7, now_dt=now)[0].title == "Article 5 Days Old"


def test_recent_articles_published_at_missing_fallback_collected_at(test_store):
    """5. published_at missing -> fallback behavior to collected_at."""
    now = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
    a_fallback_recent = _make_store_article("Fallback Recent", published_at=None, collected_at=now - timedelta(days=1))
    a_fallback_old = _make_store_article("Fallback Old", published_at=None, collected_at=now - timedelta(days=5))

    issue = NewsletterIssue(issue_date="2026-09-16", articles=[a_fallback_recent, a_fallback_old])
    test_store.save_issue(issue)

    recent = test_store.recent_articles(days=3, now_dt=now)
    titles = [a.title for a in recent]
    assert "Fallback Recent" in titles
    assert "Fallback Old" not in titles


def test_recent_articles_timezone_aware_timestamps(test_store):
    """6. Timezone-aware timestamps across different offsets."""
    from datetime import timezone as tz
    # KST is UTC+9
    kst = tz(timedelta(hours=9))
    now_utc = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
    # Published 1 day before in KST
    pub_kst = datetime(2026, 9, 16, 21, 0, 0, tzinfo=kst)  # equal to 2026-09-16 12:00 UTC (1 day old)

    a_kst = _make_store_article("KST Article", published_at=pub_kst)
    issue = NewsletterIssue(issue_date="2026-09-16", articles=[a_kst])
    test_store.save_issue(issue)

    recent = test_store.recent_articles(days=3, now_dt=now_utc)
    assert len(recent) == 1
    assert recent[0].title == "KST Article"


def test_recent_articles_invalid_or_non_positive_days(test_store):
    """7. Invalid/non-positive days -> returns empty list."""
    now = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
    a1 = _make_store_article("Recent Article", published_at=now - timedelta(days=1))
    issue = NewsletterIssue(issue_date="2026-09-16", articles=[a1])
    test_store.save_issue(issue)

    assert test_store.recent_articles(days=0, now_dt=now) == []
    assert test_store.recent_articles(days=-3, now_dt=now) == []
    assert test_store.recent_articles(days="invalid", now_dt=now) == []  # type: ignore

