from ai_newsletter.models import Article, NewsletterIssue
from ai_newsletter.presentation import (
    build_intelligence_sections,
    compute_issue_metrics,
    display_primary_category,
    display_title_en,
    display_title_ko,
    region_counts,
    topic_counts,
)


def test_bilingual_display_helpers():
    article = Article(
        title="Original English Title",
        title_ko="한글 제목",
        title_en="English Title",
        url="https://example.com",
        source="Test Source",
        category="frontier_models",
    )
    assert display_title_ko(article) == "한글 제목"
    assert display_title_en(article) == "English Title"

    assert display_primary_category(article, lang="ko") == "프론티어 모델 & LLM"
    assert display_primary_category(article, lang="en") == "Frontier Models & LLMs"


def test_build_intelligence_sections():
    a1 = Article(
        title="OpenAI o1 release",
        url="https://openai.com",
        source="OpenAI",
        category="frontier_models",
        primary_category="frontier_models",
        priority_score=90.0,
    )
    a2 = Article(
        title="NVIDIA Blackwell shipments",
        url="https://nvidia.com",
        source="NVIDIA",
        category="hardware_infra",
        primary_category="hardware_infra",
        priority_score=85.0,
    )
    issue = NewsletterIssue(issue_date="2026-09-17", articles=[a1, a2])

    sections_ko = build_intelligence_sections(issue, lang="ko")
    sec_frontier = next(s for s in sections_ko if s["key"] == "frontier_models")
    assert len(sec_frontier["articles"]) == 1
    assert sec_frontier["label"] == "프론티어 모델 & LLM"

    sections_en = build_intelligence_sections(issue, lang="en")
    sec_frontier_en = next(s for s in sections_en if s["key"] == "frontier_models")
    assert sec_frontier_en["label"] == "Frontier Models & LLMs"


def test_compute_issue_metrics():
    a1 = Article(title="A1", url="u1", source="s1", priority_score=85.0)  # critical
    a2 = Article(title="A2", url="u2", source="s2", priority_score=65.0)  # high
    a3 = Article(title="A3", url="u3", source="s3", priority_score=40.0)  # watch

    metrics = compute_issue_metrics([a1, a2, a3])
    assert metrics["total"] == 3
    assert metrics["critical"] == 1
    assert metrics["high"] == 1


def test_display_source_transparency_primary_announcement():
    from ai_newsletter.presentation import display_source_transparency, prepare_article_view

    article = Article(
        title="OpenAI announces GPT-5",
        url="https://openai.com/blog/gpt-5",
        source="OpenAI",
        publisher="OpenAI",
        is_official=True,
        source_type="official",
        evidence_level="primary",
        is_primary_source=True,
        event_verification_status="primary_only",
        event_confidence_label="High",
        event_confidence_explanation_ko="OpenAI 공식 발표",
        event_confidence_explanation_en="OpenAI primary announcement",
        event_related_sources=["OpenAI"],
        event_primary_sources=["openai"],
    )
    t = display_source_transparency(article)
    assert t.evidence_badge.key == "primary"
    assert t.evidence_badge.label_en == "Primary"
    assert t.evidence_badge.label_ko == "1차 출처"
    assert t.sources_display == "OpenAI"
    assert t.evidence_label_en == "Primary announcement"
    assert t.evidence_label_ko == "1차 공식 발표"
    assert t.confidence_label == "High"
    assert t.confidence_label_ko == "높음"
    assert t.is_multi_source is False

    view = prepare_article_view(article)
    assert view.transparency.evidence_badge.key == "primary"


def test_display_source_transparency_multi_source():
    from ai_newsletter.presentation import display_source_transparency

    article = Article(
        title="OpenAI launches GPT-5",
        url="https://reuters.com/tech/gpt5",
        source="Reuters Technology",
        publisher="Reuters",
        source_type="media",
        evidence_level="independent",
        is_independent_source=True,
        event_verification_status="independently_reported",
        event_confidence_label="High",
        event_confidence_explanation_ko="OpenAI 공식 발표 + Reuters 독립 취재",
        event_confidence_explanation_en="OpenAI primary announcement + independent Reuters reporting",
        event_related_sources=["OpenAI", "Reuters", "TechCrunch"],
        event_primary_sources=["openai"],
        event_independent_sources=["reuters"],
    )
    t = display_source_transparency(article)
    assert t.evidence_badge.key == "independent"
    assert t.evidence_badge.label_en == "Independent"
    assert t.evidence_badge.label_ko == "독립 취재"
    assert "OpenAI" in t.sources_display
    assert "Reuters" in t.sources_display
    assert t.evidence_label_en == "Primary + independent reporting"
    assert t.evidence_label_ko == "1차 출처 + 독립 언론 취재"
    assert t.confidence_label == "High"
    assert t.confidence_label_ko == "높음"
    assert t.is_multi_source is True
    assert "openai" in t.primary_sources
    assert "reuters" in t.independent_sources


def test_display_source_transparency_all_evidence_levels():
    from ai_newsletter.presentation import EVIDENCE_BADGES, display_source_transparency

    levels = ["primary", "research", "independent", "industry_media", "community", "discovery"]
    for lvl in levels:
        art = Article(
            title=f"Article {lvl}",
            url=f"https://example.com/{lvl}",
            source="Test Publisher",
            evidence_level=lvl,
        )
        t = display_source_transparency(art)
        assert t.evidence_badge.key == lvl
        assert t.evidence_badge.label_en == EVIDENCE_BADGES[lvl].label_en
        assert t.evidence_badge.label_ko == EVIDENCE_BADGES[lvl].label_ko


# ── Tests for Evidence Taxonomy Source Filtering Alignment ────────────────────

def test_source_type_filters_evidence_taxonomy_alignment():
    """Verify SOURCE_TYPE_FILTERS conforms to the evidence taxonomy with correct Korean and English labels."""
    from ai_newsletter.presentation import SOURCE_TYPE_FILTERS

    keys = [f.key for f in SOURCE_TYPE_FILTERS]
    assert keys == ["all", "primary", "research", "independent", "industry_media", "community", "discovery"]

    labels_ko = {f.key: f.label_ko for f in SOURCE_TYPE_FILTERS}
    assert labels_ko == {
        "all": "전체",
        "primary": "1차 출처",
        "research": "연구",
        "independent": "독립 취재",
        "industry_media": "산업 미디어",
        "community": "커뮤니티",
        "discovery": "검색 집계",
    }

    labels_en = {f.key: f.label_en for f in SOURCE_TYPE_FILTERS}
    assert labels_en == {
        "all": "All",
        "primary": "Primary",
        "research": "Research",
        "independent": "Independent",
        "industry_media": "Industry Media",
        "community": "Community",
        "discovery": "Discovery",
    }


def test_canonical_source_type_and_counts():
    """Verify canonical_source_type resolves correctly for each evidence level and legacy source types."""
    from ai_newsletter.presentation import canonical_source_type, source_type_counts, prepare_article_view

    a_prim = Article(title="A1", url="u1", source="OpenAI", evidence_level="primary", is_primary_source=True)
    a_res = Article(title="A2", url="u2", source="arXiv", evidence_level="research")
    a_ind = Article(title="A3", url="u3", source="Reuters", evidence_level="independent", is_independent_source=True)
    a_med = Article(title="A4", url="u4", source="TechCrunch", evidence_level="industry_media")
    a_com = Article(title="A5", url="u5", source="Hacker News", evidence_level="community")
    a_disc = Article(title="A6", url="u6", source="Google News", evidence_level="discovery", is_discovery_source=True)

    # Legacy articles without explicit evidence_level
    a_leg_off = Article(title="A7", url="u7", source="OpenAI", source_type="official")
    a_leg_reg = Article(title="A8", url="u8", source="NIST", source_type="regulator")

    assert canonical_source_type(a_prim) == "primary"
    assert canonical_source_type(a_res) == "research"
    assert canonical_source_type(a_ind) == "independent"
    assert canonical_source_type(a_med) == "industry_media"
    assert canonical_source_type(a_com) == "community"
    assert canonical_source_type(a_disc) == "discovery"
    assert canonical_source_type(a_leg_off) == "primary"
    assert canonical_source_type(a_leg_reg) == "primary"

    articles = [a_prim, a_res, a_ind, a_med, a_com, a_disc]
    counts = source_type_counts(articles)
    assert counts["all"] == 6
    assert counts["primary"] == 1
    assert counts["research"] == 1
    assert counts["independent"] == 1
    assert counts["industry_media"] == 1
    assert counts["community"] == 1
    assert counts["discovery"] == 1

    # Check article view has matching data attribute
    v_disc = prepare_article_view(a_disc)
    assert v_disc.source_type == "discovery"


def test_display_topics_str_and_clean_key_points():
    from ai_newsletter.presentation import display_key_points, display_topics_str, prepare_article_view

    article = Article(
        title="Meta releases Llama 3.3",
        url="https://theinformation.com/articles/meta-llama-33",
        source="The Information",
        publisher="The Information",
        category="frontier_models",
        topics=["Funding & M&A", "Open Weights"],
        key_points=[
            "주요 기업/모델 (Entities): Meta AI",
            "출처 (Source): The Information",
            "핵심 분야 (Topics): Funding & M&A",
            "발행처: The Information",
            "실제 핵심 사실: 새로운 70B 오픈 가중치 모델 공개",
        ],
    )

    topics_str = display_topics_str(article)
    assert "Funding & M&A" in topics_str

    cleaned_points = display_key_points(article)
    # Redundant source and topics points should be filtered out
    assert not any("출처" in pt for pt in cleaned_points)
    assert not any("발행처" in pt for pt in cleaned_points)
    assert not any("핵심 분야" in pt for pt in cleaned_points)
    # Substantive facts and entities are kept
    assert any("Meta AI" in pt for pt in cleaned_points)
    assert any("새로운 70B" in pt for pt in cleaned_points)

    view = prepare_article_view(article)
    assert view.topics_display == topics_str
    assert view.transparency.sources_display == "The Information"




