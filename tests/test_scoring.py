from ai_newsletter.models import Article
from ai_newsletter.scoring import (
    compute_impact_score,
    compute_multi_dimensional_scores,
    compute_novelty_score,
    compute_recency_score,
    compute_relevance_score,
    compute_source_score,
)


def test_source_scoring():
    a_official = Article(
        title="OpenAI Launch",
        url="https://openai.com",
        source="OpenAI",
        source_type="official",
        source_authority=95,
    )
    assert compute_source_score(a_official) >= 90.0

    a_aggregator = Article(
        title="Google News Snippet",
        url="https://news.google.com",
        source="Google News",
        source_type="aggregator",
        source_authority=50,
    )
    assert compute_source_score(a_aggregator) <= 60.0


def test_relevance_scoring():
    a_high = Article(
        title="Anthropic introduces Claude 3.5 Sonnet autonomous agent",
        url="https://example.com",
        source="TechCrunch",
        category="frontier_models",
        topics=["LLM", "Agents"],
    )
    score_high = compute_relevance_score(a_high)
    assert score_high >= 70.0


def test_impact_scoring():
    a_big = Article(
        title="NVIDIA achieves state-of-the-art benchmark with $10 billion datacenter cluster deployment",
        url="https://example.com",
        source="Reuters",
        category="hardware_infra",
        excerpt="The multi-billion investment enables unprecedented scale.",
    )
    score_impact = compute_impact_score(a_big)
    assert score_impact >= 60.0


def test_multi_dimensional_scores():
    article = Article(
        title="OpenAI announces GPT-4o with multimodal reasoning and API access",
        url="https://openai.com",
        source="OpenAI",
        category="frontier_models",
        source_type="official",
        source_authority=95,
        topics=["LLM", "Reasoning"],
        excerpt="The new flagship model outperforms prior generation models.",
    )
    res = compute_multi_dimensional_scores(article)
    assert res.priority_score >= 75.0
    assert res.explanation.summary_ko
    assert res.explanation.summary_en

