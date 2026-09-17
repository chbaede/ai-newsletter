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

