"""
Tests for event evidence computation and verification_status logic.

Covers:
- primary_only: one official publisher, no independent
- independently_reported: primary + independent
- multi_source: two or more independent publishers
- same publisher multiple articles (must count as ONE source)
- discovery_only: Google News / aggregator articles only
- insufficient_evidence: community only or empty publisher
- unrelated articles (should NOT cluster into one event)
- multilingual articles about the same event
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ai_newsletter.clustering import (
    VALID_VERIFICATION_STATUSES,
    VERIFICATION_STATUS_DISCOVERY_ONLY,
    VERIFICATION_STATUS_INDEPENDENTLY_REPORTED,
    VERIFICATION_STATUS_INSUFFICIENT,
    VERIFICATION_STATUS_MULTI_SOURCE,
    VERIFICATION_STATUS_PRIMARY_ONLY,
    EventEvidence,
    _normalize_publisher,
    cluster_articles,
    compute_event_evidence,
)
from ai_newsletter.models import Article, Event


# ── helpers ────────────────────────────────────────────────────────────────────

def make_article(
    title: str,
    url: str,
    publisher: str,
    source: str | None = None,
    evidence_level: str = "industry_media",
    source_type: str = "media",
    is_primary_source: bool = False,
    is_independent_source: bool = False,
    is_discovery_source: bool = False,
    tags: list[str] | None = None,
    published_at: datetime | None = None,
) -> Article:
    return Article(
        title=title,
        url=url,
        source=source or publisher,
        publisher=publisher,
        evidence_level=evidence_level,
        source_type=source_type,
        is_primary_source=is_primary_source,
        is_independent_source=is_independent_source,
        is_discovery_source=is_discovery_source,
        is_official=(source_type == "official"),
        tags=tags or [],
        published_at=published_at,
    )


NOW = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)


# ── _normalize_publisher ───────────────────────────────────────────────────────

def test_normalize_publisher_strips_suffixes():
    assert _normalize_publisher("Reuters Technology") == "reuters"
    assert _normalize_publisher("Reuters News") == "reuters"
    assert _normalize_publisher("REUTERS") == "reuters"
    assert _normalize_publisher("reuters.com") == "reuters"
    assert _normalize_publisher("OpenAI Newsroom") == "openai"
    assert _normalize_publisher("OpenAI Blog") == "openai"
    assert _normalize_publisher("Bloomberg") == "bloomberg"
    assert _normalize_publisher("Bloomberg Finance") == "bloomberg"


def test_normalize_publisher_preserves_distinct_entities():
    assert _normalize_publisher("Reuters") != _normalize_publisher("Bloomberg")
    assert _normalize_publisher("OpenAI") != _normalize_publisher("Anthropic")
    assert _normalize_publisher("TechCrunch") != _normalize_publisher("The Verge")


# ── compute_event_evidence: primary_only ───────────────────────────────────────

def test_evidence_primary_only():
    """One official publisher, no independent → primary_only."""
    articles = [
        make_article(
            "OpenAI launches GPT-5",
            "https://openai.com/blog/gpt-5",
            publisher="OpenAI",
            evidence_level="primary",
            source_type="official",
            is_primary_source=True,
        )
    ]
    ev = compute_event_evidence(articles)
    assert ev.verification_status == VERIFICATION_STATUS_PRIMARY_ONLY
    assert "openai" in ev.primary_sources
    assert ev.independent_sources == []
    assert ev.evidence_diversity == 1


def test_evidence_primary_only_with_same_publisher_repost():
    """
    OpenAI post + OpenAI repost = still ONE primary publisher.
    Must NOT become multi_source or independently_reported.
    """
    articles = [
        make_article(
            "OpenAI announces GPT-5",
            "https://openai.com/blog/gpt-5",
            publisher="OpenAI Newsroom",
            evidence_level="primary",
            source_type="official",
            is_primary_source=True,
        ),
        make_article(
            "OpenAI GPT-5 launch recap",
            "https://openai.com/blog/gpt-5-recap",
            publisher="OpenAI Blog",   # same publisher, different suffix
            evidence_level="primary",
            source_type="official",
            is_primary_source=True,
        ),
        make_article(
            "OpenAI GPT-5 social announcement",
            "https://twitter.com/openai/gpt5",
            publisher="OpenAI",
            evidence_level="primary",
            source_type="official",
            is_primary_source=True,
        ),
    ]
    ev = compute_event_evidence(articles)
    # All three normalize to "openai" → one primary publisher
    assert ev.verification_status == VERIFICATION_STATUS_PRIMARY_ONLY
    assert len(ev.primary_sources) == 1
    assert ev.independent_sources == []
    assert ev.evidence_diversity == 1


# ── compute_event_evidence: independently_reported ────────────────────────────

def test_evidence_independently_reported():
    """Primary (OpenAI) + independent (Reuters) → independently_reported."""
    articles = [
        make_article(
            "OpenAI launches GPT-5",
            "https://openai.com/blog/gpt-5",
            publisher="OpenAI",
            evidence_level="primary",
            source_type="official",
            is_primary_source=True,
        ),
        make_article(
            "OpenAI unveils GPT-5 in major AI leap",
            "https://reuters.com/technology/openai-gpt5",
            publisher="Reuters",
            evidence_level="independent",
            source_type="media",
            is_independent_source=True,
        ),
    ]
    ev = compute_event_evidence(articles)
    assert ev.verification_status == VERIFICATION_STATUS_INDEPENDENTLY_REPORTED
    assert "openai" in ev.primary_sources
    assert "reuters" in ev.independent_sources
    assert ev.evidence_diversity >= 2


def test_evidence_independently_reported_reuters_follow_up():
    """
    Reuters article + Reuters follow-up = still ONE independent publisher.
    Must remain independently_reported (primary + one distinct independent).
    """
    articles = [
        make_article(
            "OpenAI launches GPT-5",
            "https://openai.com/blog/gpt-5",
            publisher="OpenAI",
            evidence_level="primary",
            source_type="official",
            is_primary_source=True,
        ),
        make_article(
            "OpenAI GPT-5 — what it means for the industry",
            "https://reuters.com/technology/openai-gpt5-analysis",
            publisher="Reuters Technology",
            evidence_level="independent",
            source_type="media",
            is_independent_source=True,
        ),
        make_article(
            "GPT-5 launch follow-up: reactions",
            "https://reuters.com/technology/gpt5-reactions",
            publisher="Reuters News",
            evidence_level="independent",
            source_type="media",
            is_independent_source=True,
        ),
    ]
    ev = compute_event_evidence(articles)
    # Reuters Technology and Reuters News both normalize to "reuters"
    assert ev.verification_status == VERIFICATION_STATUS_INDEPENDENTLY_REPORTED
    assert len(ev.independent_sources) == 1, (
        f"Expected 1 independent source (Reuters), got {ev.independent_sources}"
    )


# ── compute_event_evidence: multi_source ──────────────────────────────────────

def test_evidence_multi_source_two_independent_publishers():
    """Reuters + Bloomberg = two distinct independent publishers → multi_source."""
    articles = [
        make_article(
            "OpenAI launches GPT-5",
            "https://openai.com/blog/gpt-5",
            publisher="OpenAI",
            evidence_level="primary",
            source_type="official",
            is_primary_source=True,
        ),
        make_article(
            "OpenAI unveils GPT-5 — biggest model yet",
            "https://reuters.com/technology/openai-gpt5",
            publisher="Reuters",
            evidence_level="independent",
            source_type="media",
            is_independent_source=True,
        ),
        make_article(
            "OpenAI GPT-5 shakes up AI landscape",
            "https://bloomberg.com/news/openai-gpt5",
            publisher="Bloomberg",
            evidence_level="independent",
            source_type="media",
            is_independent_source=True,
        ),
        make_article(
            "TechCrunch on OpenAI GPT-5",
            "https://techcrunch.com/openai-gpt5",
            publisher="TechCrunch",
            evidence_level="industry_media",
            source_type="media",
        ),
    ]
    ev = compute_event_evidence(articles)
    assert ev.verification_status == VERIFICATION_STATUS_MULTI_SOURCE
    assert len(ev.independent_sources) >= 2
    assert "reuters" in ev.independent_sources
    assert "bloomberg" in ev.independent_sources


def test_evidence_multi_source_no_primary():
    """Two independent publishers, no primary → multi_source."""
    articles = [
        make_article(
            "Reports: OpenAI preparing GPT-5 launch",
            "https://reuters.com/technology/openai-gpt5-prep",
            publisher="Reuters",
            evidence_level="independent",
            source_type="media",
            is_independent_source=True,
        ),
        make_article(
            "Bloomberg: OpenAI set to unveil GPT-5",
            "https://bloomberg.com/news/openai-gpt5-upcoming",
            publisher="Bloomberg",
            evidence_level="independent",
            source_type="media",
            is_independent_source=True,
        ),
    ]
    ev = compute_event_evidence(articles)
    assert ev.verification_status == VERIFICATION_STATUS_MULTI_SOURCE
    assert ev.primary_sources == []
    assert len(ev.independent_sources) == 2


# ── compute_event_evidence: discovery_only ────────────────────────────────────

def test_evidence_discovery_only():
    """Articles all from Google News discovery → discovery_only."""
    articles = [
        make_article(
            "OpenAI GPT-5 news - Google News",
            "https://news.google.com/articles/xxx",
            publisher="Google News",
            evidence_level="discovery",
            source_type="aggregator",
            is_discovery_source=True,
        ),
        make_article(
            "OpenAI GPT-5 update - Google News",
            "https://news.google.com/articles/yyy",
            publisher="Google News",
            evidence_level="discovery",
            source_type="aggregator",
            is_discovery_source=True,
        ),
    ]
    ev = compute_event_evidence(articles)
    assert ev.verification_status == VERIFICATION_STATUS_DISCOVERY_ONLY
    assert ev.primary_sources == []
    assert ev.independent_sources == []
    assert ev.evidence_diversity == 0


# ── compute_event_evidence: insufficient_evidence ─────────────────────────────

def test_evidence_insufficient_community_only():
    """Community discussion only → insufficient_evidence."""
    articles = [
        make_article(
            "Is GPT-5 coming? Discussion thread",
            "https://news.ycombinator.com/item?id=12345",
            publisher="Hacker News",
            evidence_level="community",
            source_type="community",
        ),
    ]
    ev = compute_event_evidence(articles)
    assert ev.verification_status == VERIFICATION_STATUS_INSUFFICIENT
    assert ev.evidence_diversity == 0


def test_evidence_insufficient_empty():
    """Empty cluster → insufficient_evidence with empty lists."""
    ev = compute_event_evidence([])
    assert ev.verification_status == VERIFICATION_STATUS_INSUFFICIENT
    assert ev.evidence_sources == []
    assert ev.primary_sources == []
    assert ev.independent_sources == []
    assert ev.evidence_diversity == 0


# ── cluster_articles: verification_status propagated ─────────────────────────

def test_cluster_articles_sets_verification_status():
    """cluster_articles() must set verification_status on Event objects."""
    a1 = Article(
        title="OpenAI announces GPT-4o with omni modal intelligence",
        url="https://openai.com/index/hello-gpt-4o",
        source="OpenAI Newsroom",
        publisher="OpenAI",
        category="frontier_models",
        is_official=True,
        source_type="official",
        evidence_level="primary",
        is_primary_source=True,
        score=95.0,
        tags=["OpenAI"],
    )
    a2 = Article(
        title="OpenAI launches GPT-4o multimodal model for all users",
        url="https://techcrunch.com/openai-gpt4o",
        source="TechCrunch",
        publisher="TechCrunch",
        category="frontier_models",
        is_official=False,
        source_type="media",
        evidence_level="industry_media",
        score=85.0,
        tags=["OpenAI"],
    )

    events, _, _ = cluster_articles([a1, a2])
    assert len(events) == 1
    event = events[0]
    assert event.verification_status in VALID_VERIFICATION_STATUSES
    assert event.primary_sources == ["openai"]
    assert event.evidence_diversity >= 1


def test_cluster_articles_multi_source_verification():
    """Clustering OpenAI + Reuters + Bloomberg gives multi_source."""
    a_openai = Article(
        title="OpenAI announces GPT-4o multimodal intelligence model",
        url="https://openai.com/blog/gpt-4o",
        source="OpenAI",
        publisher="OpenAI",
        category="frontier_models",
        source_type="official",
        evidence_level="primary",
        is_primary_source=True,
        is_official=True,
        score=95.0,
        tags=["OpenAI", "GPT-4o"],
    )
    a_reuters = Article(
        title="OpenAI launches GPT-4o multimodal AI model",
        url="https://reuters.com/technology/openai-gpt4o",
        source="Reuters Technology AI",
        publisher="Reuters",
        category="frontier_models",
        source_type="media",
        evidence_level="independent",
        is_independent_source=True,
        score=80.0,
        tags=["OpenAI", "GPT-4o"],
    )
    a_bloomberg = Article(
        title="OpenAI debuts GPT-4o, its latest multimodal AI model",
        url="https://bloomberg.com/news/openai-gpt4o-launch",
        source="Bloomberg Technology AI",
        publisher="Bloomberg",
        category="frontier_models",
        source_type="media",
        evidence_level="independent",
        is_independent_source=True,
        score=80.0,
        tags=["OpenAI", "GPT-4o"],
    )

    events, _, _ = cluster_articles([a_openai, a_reuters, a_bloomberg])
    # All three should cluster together
    assert len(events) == 1
    event = events[0]
    assert event.verification_status == VERIFICATION_STATUS_MULTI_SOURCE
    assert len(event.independent_sources) == 2
    assert "reuters" in event.independent_sources
    assert "bloomberg" in event.independent_sources
    assert "openai" in event.primary_sources


def test_cluster_articles_unrelated_no_merge():
    """Unrelated articles must NOT be merged into one event."""
    a1 = Article(
        title="OpenAI announces GPT-4o with omni modal intelligence",
        url="https://openai.com/index/hello-gpt-4o",
        source="OpenAI Newsroom",
        publisher="OpenAI",
        category="frontier_models",
        is_official=True,
        source_type="official",
        evidence_level="primary",
        is_primary_source=True,
        score=95.0,
        tags=["OpenAI"],
    )
    a2 = Article(
        title="NVIDIA announces Blackwell B200 shipments for AI servers",
        url="https://nvidianews.nvidia.com/blackwell",
        source="NVIDIA",
        publisher="NVIDIA",
        category="hardware_infra",
        is_official=True,
        source_type="official",
        evidence_level="primary",
        is_primary_source=True,
        score=92.0,
        tags=["NVIDIA"],
    )

    events, _, _ = cluster_articles([a1, a2])
    assert len(events) == 2, f"Expected 2 separate events, got {len(events)}"
    for event in events:
        assert event.verification_status == VERIFICATION_STATUS_PRIMARY_ONLY


# ── multilingual articles about the same event ────────────────────────────────

def test_evidence_multilingual_same_publisher():
    """
    Korean article from ZDNet Korea and English article from ZDNet Korea
    about the same event must count as ONE publisher, not two.
    """
    articles = [
        make_article(
            "OpenAI, GPT-5 공개… 추론 능력 획기적 향상",
            "https://zdnet.co.kr/view/?no=12345",
            publisher="ZDNet Korea",
            evidence_level="industry_media",
        ),
        make_article(
            "OpenAI GPT-5: what the Korean coverage says",
            "https://zdnet.co.kr/view-en/?no=12345",
            publisher="ZDNet Korea",
            evidence_level="industry_media",
        ),
    ]
    ev = compute_event_evidence(articles)
    # Both normalize to "zdnet korea"
    assert len(ev.evidence_sources) == 1
    assert ev.evidence_diversity <= 1


def test_evidence_verification_status_is_always_valid():
    """All possible evidence combinations must produce a valid verification_status."""
    combos = [
        [],
        [make_article("x", "https://a.com", "OpenAI", evidence_level="primary",
                      source_type="official", is_primary_source=True)],
        [make_article("x", "https://b.com", "Reuters", evidence_level="independent",
                      is_independent_source=True)],
        [make_article("x", "https://c.com", "Google News", evidence_level="discovery",
                      is_discovery_source=True)],
        [make_article("x", "https://d.com", "Hacker News", evidence_level="community")],
        [make_article("x", "https://e.com", "TechCrunch", evidence_level="industry_media")],
    ]
    for articles in combos:
        ev = compute_event_evidence(articles)
        assert ev.verification_status in VALID_VERIFICATION_STATUSES, (
            f"Invalid status '{ev.verification_status}' for articles: "
            f"{[a.publisher for a in articles]}"
        )


def test_event_evidence_fields_exist():
    """Verify Event dataclass has all required evidence metadata fields."""
    event = Event(
        event_id="ev_test",
        title="Test Event",
    )
    assert hasattr(event, "evidence_sources")
    assert hasattr(event, "primary_sources")
    assert hasattr(event, "independent_sources")
    assert hasattr(event, "evidence_diversity")
    assert hasattr(event, "verification_status")
    assert hasattr(event, "confidence_score")
    assert hasattr(event, "confidence_label")
    assert hasattr(event, "confidence_explanation_ko")
    assert hasattr(event, "confidence_explanation_en")
    assert isinstance(event.evidence_sources, list)
    assert isinstance(event.primary_sources, list)
    assert isinstance(event.independent_sources, list)
    assert isinstance(event.evidence_diversity, int)
    assert isinstance(event.verification_status, str)
    assert isinstance(event.confidence_score, float)
    assert isinstance(event.confidence_label, str)
    assert isinstance(event.confidence_explanation_ko, str)
    assert isinstance(event.confidence_explanation_en, str)


# ── Confidence scoring tests ──────────────────────────────────────────────────

from ai_newsletter.clustering import (
    CONFIDENCE_LABEL_HIGH,
    CONFIDENCE_LABEL_HIGH_KO,
    CONFIDENCE_LABEL_LOW,
    CONFIDENCE_LABEL_LOW_KO,
    CONFIDENCE_LABEL_MEDIUM,
    CONFIDENCE_LABEL_MEDIUM_KO,
    compute_event_confidence,
)


def test_confidence_primary_only_official():
    """Primary announcement only -> Medium / High depending on authority."""
    articles = [
        make_article(
            "OpenAI announces GPT-5",
            "https://openai.com/blog/gpt-5",
            publisher="OpenAI",
            evidence_level="primary",
            source_type="official",
            is_primary_source=True,
        )
    ]
    ev = compute_event_evidence(articles)
    conf = compute_event_confidence(ev, articles)
    assert conf.confidence_label in {CONFIDENCE_LABEL_MEDIUM, CONFIDENCE_LABEL_HIGH}
    assert "Openai primary announcement" in conf.explanation_en or "OpenAI" in conf.explanation_en
    assert "Openai 공식 발표" in conf.explanation_ko or "OpenAI" in conf.explanation_ko
    assert "verified" not in conf.explanation_en.lower()
    assert "fact" not in conf.explanation_en.lower()


def test_confidence_primary_plus_independent():
    """Primary + independent reporting -> High confidence."""
    articles = [
        make_article(
            "OpenAI launches GPT-5",
            "https://openai.com/blog/gpt-5",
            publisher="OpenAI",
            evidence_level="primary",
            source_type="official",
            is_primary_source=True,
        ),
        make_article(
            "OpenAI launches GPT-5 in major milestone",
            "https://reuters.com/tech/openai-gpt5",
            publisher="Reuters",
            evidence_level="independent",
            source_type="media",
            is_independent_source=True,
        ),
    ]
    ev = compute_event_evidence(articles)
    conf = compute_event_confidence(ev, articles)
    assert conf.confidence_label == CONFIDENCE_LABEL_HIGH
    assert conf.label_ko == CONFIDENCE_LABEL_HIGH_KO
    assert conf.confidence_score >= 70.0
    assert "Openai primary announcement" in conf.explanation_en
    assert "independent Reuters reporting" in conf.explanation_en
    assert "Openai 공식 발표" in conf.explanation_ko
    assert "Reuters 독립 취재" in conf.explanation_ko


def test_confidence_multiple_independent_publishers():
    """Multiple independent publishers -> High confidence."""
    articles = [
        make_article(
            "OpenAI prepares GPT-5 launch",
            "https://reuters.com/tech/openai-gpt5",
            publisher="Reuters",
            evidence_level="independent",
            source_type="media",
            is_independent_source=True,
        ),
        make_article(
            "Bloomberg: OpenAI to unveil GPT-5 soon",
            "https://bloomberg.com/tech/openai-gpt5",
            publisher="Bloomberg",
            evidence_level="independent",
            source_type="media",
            is_independent_source=True,
        ),
    ]
    ev = compute_event_evidence(articles)
    conf = compute_event_confidence(ev, articles)
    assert conf.confidence_label == CONFIDENCE_LABEL_HIGH
    assert conf.label_ko == CONFIDENCE_LABEL_HIGH_KO
    assert conf.confidence_score >= 75.0
    assert "independent" in conf.explanation_en.lower()
    assert "독립 취재" in conf.explanation_ko


def test_confidence_research_preprint():
    """Research publication / preprint -> Medium confidence."""
    articles = [
        make_article(
            "Scaling Laws for Next-Gen LLMs",
            "https://arxiv.org/abs/2609.12345",
            publisher="arXiv",
            evidence_level="research",
            source_type="research",
        )
    ]
    ev = compute_event_evidence(articles)
    conf = compute_event_confidence(ev, articles)
    assert conf.confidence_label == CONFIDENCE_LABEL_MEDIUM
    assert conf.label_ko == CONFIDENCE_LABEL_MEDIUM_KO
    assert "research publication" in conf.explanation_en.lower()
    assert "연구기관 발표 기반" in conf.explanation_ko


def test_confidence_community_only():
    """Community discussion only -> Low confidence."""
    articles = [
        make_article(
            "Rumors about GPT-5 on Hacker News",
            "https://news.ycombinator.com/item?id=9999",
            publisher="Hacker News",
            evidence_level="community",
            source_type="community",
        )
    ]
    ev = compute_event_evidence(articles)
    conf = compute_event_confidence(ev, articles)
    assert conf.confidence_label == CONFIDENCE_LABEL_LOW
    assert conf.label_ko == CONFIDENCE_LABEL_LOW_KO
    assert conf.confidence_score < 44.0
    assert "community" in conf.explanation_en.lower() or "insufficient" in conf.explanation_en.lower()
    assert "커뮤니티" in conf.explanation_ko or "출처 없음" in conf.explanation_ko


def test_confidence_discovery_only():
    """Google News discovery only -> Low confidence."""
    articles = [
        make_article(
            "OpenAI GPT-5 update - Google News",
            "https://news.google.com/articles/123",
            publisher="Google News",
            evidence_level="discovery",
            source_type="aggregator",
            is_discovery_source=True,
        )
    ]
    ev = compute_event_evidence(articles)
    conf = compute_event_confidence(ev, articles)
    assert conf.confidence_label == CONFIDENCE_LABEL_LOW
    assert conf.label_ko == CONFIDENCE_LABEL_LOW_KO
    assert conf.confidence_score < 44.0
    assert "aggregator" in conf.explanation_en.lower()
    assert "검색 집계 결과만 확인됨" in conf.explanation_ko


def test_confidence_duplicated_same_publisher():
    """Multiple articles from same publisher do not inflate independent count or confidence."""
    articles = [
        make_article(
            "Reuters report part 1",
            "https://reuters.com/tech/p1",
            publisher="Reuters Technology",
            evidence_level="independent",
            is_independent_source=True,
        ),
        make_article(
            "Reuters report follow-up",
            "https://reuters.com/tech/p2",
            publisher="Reuters News",
            evidence_level="independent",
            is_independent_source=True,
        ),
    ]
    ev = compute_event_evidence(articles)
    conf = compute_event_confidence(ev, articles)
    # Both normalize to "reuters" -> len(independent_sources) == 1
    assert len(ev.independent_sources) == 1
    assert conf.confidence_score < 85.0  # Not boosted to multi_source base 85


def test_confidence_no_publisher_or_empty():
    """Edge case: article with empty publisher."""
    articles = [
        make_article(
            "Anonymous leak post",
            "https://example.com/post",
            publisher="",
            source="",
            evidence_level="community",
        )
    ]
    ev = compute_event_evidence(articles)
    conf = compute_event_confidence(ev, articles)
    assert conf.confidence_label == CONFIDENCE_LABEL_LOW
    assert conf.confidence_score <= 20.0
    assert "insufficient" in conf.explanation_en.lower()

