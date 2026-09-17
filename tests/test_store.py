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


# ── Tests for Event Evidence Persistence and Article Metadata Reconstruction ──


def test_event_evidence_persistence_and_article_metadata_reconstruction(test_store):
    """1-4. Save event + articles, reload from SQLite, verify event evidence & consistency."""
    ev = Event(
        event_id="ev_meta_test",
        title="Anthropic Claude 3.5 Launch",
        category="frontier_models",
        importance=95.0,
        source_count=3,
        independent_source_count=2,
        has_official_source=True,
        has_regulatory_source=False,
        has_major_media_source=True,
        official_source_url="https://anthropic.com/news/claude-3-5",
        official_source_name="Anthropic",
        reference_source_name="Reuters",
        related_sources=["Anthropic", "Reuters", "TechCrunch"],
        evidence_sources=["Anthropic", "Reuters", "TechCrunch"],
        primary_sources=["Anthropic"],
        independent_sources=["Reuters", "TechCrunch"],
        evidence_diversity=2,
        verification_status="primary_and_independent",
        confidence_score=94.0,
        confidence_label="High",
        confidence_explanation_ko="1차 출처 및 독립 언론 보도로 검증된 이벤트입니다.",
        confidence_explanation_en="Verified by primary and independent reporting.",
    )
    art1 = Article(
        title="Anthropic announces Claude 3.5 Sonnet",
        url="https://anthropic.com/news/claude-3-5",
        source="Anthropic",
        category="frontier_models",
        summary_ko="앤스로픽이 클로드 3.5 소넷을 출시했습니다.",
        event_id="ev_meta_test",
        evidence_level="primary",
        is_primary_source=True,
    )
    art2 = Article(
        title="Reuters: Anthropic debuts Claude 3.5 Sonnet",
        url="https://reuters.com/technology/anthropic-claude-3-5",
        source="Reuters",
        category="frontier_models",
        summary_ko="로이터가 클로드 3.5 출시 소식을 전했습니다.",
        event_id="ev_meta_test",
        evidence_level="independent",
        is_independent_source=True,
    )

    issue = NewsletterIssue(
        issue_date="2026-09-17",
        articles=[art1, art2],
        events=[ev],
    )
    test_store.save_issue(issue)

    # Reload issue from SQLite
    reloaded = test_store.get_issue("2026-09-17")
    assert reloaded is not None
    assert len(reloaded.events) == 1
    assert len(reloaded.articles) == 2

    # 3. Verify event evidence is still available on the reloaded Event
    rev = reloaded.events[0]
    assert rev.event_id == "ev_meta_test"
    assert rev.source_count == 3
    assert rev.independent_source_count == 2
    assert rev.has_official_source is True
    assert rev.official_source_name == "Anthropic"
    assert rev.evidence_sources == ["Anthropic", "Reuters", "TechCrunch"]
    assert rev.primary_sources == ["Anthropic"]
    assert rev.independent_sources == ["Reuters", "TechCrunch"]
    assert rev.evidence_diversity == 2
    assert rev.verification_status == "primary_and_independent"
    assert rev.confidence_score == 94.0
    assert rev.confidence_label == "High"
    assert rev.confidence_explanation_ko == "1차 출처 및 독립 언론 보도로 검증된 이벤트입니다."
    assert rev.confidence_explanation_en == "Verified by primary and independent reporting."

    # 4. Verify article values are consistent with the event
    for a in reloaded.articles:
        assert a.event_id == "ev_meta_test"
        assert a.event_title == rev.title
        assert a.event_source_count == rev.source_count
        assert a.event_independent_source_count == rev.independent_source_count
        assert a.event_has_official_source == rev.has_official_source
        assert a.event_official_source_name == rev.official_source_name
        assert a.event_official_source_url == rev.official_source_url
        assert a.event_reference_source_name == rev.reference_source_name
        assert a.event_related_sources == rev.related_sources
        assert a.event_evidence_sources == rev.evidence_sources
        assert a.event_primary_sources == rev.primary_sources
        assert a.event_independent_sources == rev.independent_sources
        assert a.event_evidence_diversity == rev.evidence_diversity
        assert a.event_verification_status == rev.verification_status
        assert a.event_confidence_score == rev.confidence_score
        assert a.event_confidence_label == rev.confidence_label
        assert a.event_confidence_explanation_ko == rev.confidence_explanation_ko
        assert a.event_confidence_explanation_en == rev.confidence_explanation_en


def test_legacy_database_without_new_evidence_columns(tmp_path):
    """5. Test loading from a legacy database where events table lacks the newer evidence columns."""
    import sqlite3
    db_file = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        """
        create table schema_version (version integer primary key);
        """
    )
    conn.execute("insert into schema_version values (1)")
    conn.execute(
        """
        create table issues (
            id integer primary key autoincrement,
            issue_date text unique not null,
            title text not null default '',
            warnings text not null default '[]',
            metrics text not null default '{}',
            created_at text not null,
            sent_at text
        );
        """
    )
    conn.execute(
        """
        create table articles (
            id integer primary key autoincrement,
            issue_id integer not null references issues(id) on delete cascade,
            position integer not null default 0,
            title text not null,
            url text not null,
            source text not null,
            category text not null,
            published_at text,
            summary_ko text not null,
            summary_en text,
            why_it_matters_ko text,
            excerpt text,
            tags text not null default '[]',
            score real not null default 0,
            discovered_via text,
            source_id text,
            source_authority real,
            source_type text,
            publisher text,
            article_id text not null,
            canonical_url text,
            original_url text,
            content text,
            content_source_type text,
            key_points text not null default '[]',
            summary_model text,
            summary_version integer,
            summary_created_at text,
            primary_category text,
            secondary_categories text not null default '[]',
            topics text not null default '[]',
            entities text not null default '[]',
            source_score real,
            relevance_score real,
            impact_score real,
            novelty_score real,
            recency_score real,
            priority_score real,
            event_id text,
            event_title text,
            related_article_ids text not null default '[]',
            is_official integer not null default 0,
            is_reference integer not null default 0,
            is_primary_source integer not null default 0,
            collected_at text
        );
        """
    )
    conn.execute(
        """
        create table events (
            event_id text primary key,
            issue_id integer not null references issues(id) on delete cascade,
            title text not null,
            category text not null default 'big',
            importance real not null default 0.0,
            primary_article_id text,
            created_at text not null,
            source_count integer not null default 1,
            independent_source_count integer not null default 1,
            has_official_source integer not null default 0,
            has_regulatory_source integer not null default 0,
            has_major_media_source integer not null default 0,
            official_source_url text,
            official_source_name text,
            reference_source_name text,
            related_sources text not null default '[]'
        );
        """
    )
    conn.execute("insert into issues (id, issue_date, title, created_at) values (1, '2026-09-17', 'Legacy Issue', '2026-09-17T06:00:00')")
    conn.execute(
        """
        insert into events (event_id, issue_id, title, category, importance, created_at, source_count, independent_source_count)
        values ('ev_legacy', 1, 'Legacy Event', 'big', 80.0, '2026-09-17T06:00:00', 1, 0)
        """
    )
    conn.execute(
        """
        insert into articles (issue_id, position, title, url, source, category, summary_ko, article_id, event_id, collected_at)
        values (1, 0, 'Legacy Article', 'https://legacy.com/1', 'Legacy Source', 'big', '레거시 요약', 'art_legacy', 'ev_legacy', '2026-09-17T06:00:00')
        """
    )
    conn.commit()
    conn.close()

    # Now open with NewsletterStore (which runs migrations/handles missing columns safely)
    store = NewsletterStore(db_file)
    issue = store.get_issue("2026-09-17")
    assert issue is not None
    assert len(issue.events) == 1
    assert len(issue.articles) == 1
    ev = issue.events[0]
    assert ev.event_id == "ev_legacy"
    assert ev.evidence_sources == []
    assert ev.verification_status == "insufficient_evidence"

    art = issue.articles[0]
    assert art.event_id == "ev_legacy"
    assert art.event_evidence_sources == []
    assert art.event_verification_status == "insufficient_evidence"


def test_eventless_articles_persistence_and_loading(test_store):
    """6. Test articles without an event_id load cleanly with default event values."""
    art = Article(
        title="Standalone Article without Event",
        url="https://example.com/standalone",
        source="Standalone Tech",
        category="startups",
        summary_ko="이벤트에 묶이지 않은 단독 기사입니다.",
        event_id=None,
    )
    issue = NewsletterIssue(
        issue_date="2026-09-17",
        articles=[art],
        events=[],
    )
    test_store.save_issue(issue)

    reloaded = test_store.get_issue("2026-09-17")
    assert reloaded is not None
    assert len(reloaded.articles) == 1
    assert len(reloaded.events) == 0

    reloaded_art = reloaded.articles[0]
    assert reloaded_art.event_id is None
    assert reloaded_art.event_verification_status == "insufficient_evidence"
    assert reloaded_art.event_confidence_score == 0.0
    assert reloaded_art.event_confidence_label == "Low"


