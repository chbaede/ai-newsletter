from ai_newsletter.models import Article
from ai_newsletter.scoring import (
    compute_evidence_quality_score,
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
    assert res.evidence_quality_score >= 95.0
    assert res.source_authority_score == res.source_score
    assert res.explanation.summary_ko
    assert res.explanation.summary_en
    assert "공식 발표 출처" in res.explanation.summary_ko
    assert "Primary source" in res.explanation.summary_en


def test_evidence_quality_scoring_levels():
    # Primary official announcement
    a_primary = Article(
        title="Anthropic announces Claude 3.5 Sonnet",
        url="https://anthropic.com/news/claude-3-5",
        source="Anthropic",
        source_type="official",
        is_primary_source=True,
        evidence_level="primary",
    )
    assert compute_evidence_quality_score(a_primary) >= 95.0

    # Independent investigative reporting
    a_independent = Article(
        title="Inside the AI chip race",
        url="https://reuters.com/technology/ai-chips",
        source="Reuters",
        is_independent_source=True,
        evidence_level="independent",
    )
    assert compute_evidence_quality_score(a_independent) >= 92.0

    # Research: preprint (arXiv) vs institutional lab
    a_arxiv = Article(
        title="Chain-of-Thought Reasoning in Latent Space",
        url="https://arxiv.org/abs/2401.12345",
        source="arXiv",
        source_type="research",
        evidence_level="research",
    )
    assert 85.0 <= compute_evidence_quality_score(a_arxiv) <= 90.0

    a_stanford = Article(
        title="Stanford AI Index Report",
        url="https://hai.stanford.edu/news/report",
        source="Stanford HAI",
        source_type="research",
        evidence_level="research",
    )
    assert 92.0 <= compute_evidence_quality_score(a_stanford) <= 96.0

    # Industry media
    a_media = Article(
        title="AI startup raises funding",
        url="https://techcrunch.com/article",
        source="TechCrunch",
        evidence_level="industry_media",
    )
    assert 75.0 <= compute_evidence_quality_score(a_media) <= 80.0

    # Community discussion
    a_comm = Article(
        title="Discussion on local LLM performance",
        url="https://news.ycombinator.com/item?id=123456",
        source="Hacker News",
        source_type="community",
        evidence_level="community",
    )
    assert 50.0 <= compute_evidence_quality_score(a_comm) <= 60.0

    # Discovery / Aggregator
    a_disc = Article(
        title="Google News AI round up",
        url="https://news.google.com/articles/xyz",
        source="Google News",
        is_discovery_source=True,
        evidence_level="discovery",
    )
    assert compute_evidence_quality_score(a_disc) <= 42.0


def test_google_news_cannot_gain_high_evidence_quality_via_high_authority_publisher():
    """Google News must NOT receive high evidence quality even if the publisher is OpenAI."""
    a_gnews_openai = Article(
        title="OpenAI announces next gen reasoning model",
        url="https://news.google.com/articles/openai-launch",
        source="Google News",
        publisher="OpenAI",
        source_type="aggregator",
        source_authority=95,
        is_discovery_source=True,
        evidence_level="discovery",
    )
    # Source authority reflects OpenAI/prominence, but evidence quality MUST stay low discovery tier
    assert compute_evidence_quality_score(a_gnews_openai) <= 42.0
    # Confirm they are separate concepts
    assert compute_source_score(a_gnews_openai) > compute_evidence_quality_score(a_gnews_openai)


def test_neutral_factual_evidence_explanations():
    cases = [
        ("primary", "공식 발표 출처", "Primary source"),
        ("independent", "독립 취재 출처", "Independent reporting"),
        ("research", "공인 학술·연구기관 출처", "Research source"),
        ("industry_media", "산업 전문 보도 출처", "Industry media coverage"),
        ("community", "커뮤니티 논의 출처", "Community discussion source"),
        ("discovery", "검색 집계 출처", "Discovery source"),
    ]

    for lvl, exp_ko, exp_en in cases:
        art = Article(
            title=f"Sample title for {lvl}",
            url=f"https://example.com/{lvl}",
            source="Sample Source",
            evidence_level=lvl,
            is_primary_source=(lvl == "primary"),
            is_independent_source=(lvl == "independent"),
            is_discovery_source=(lvl == "discovery"),
        )
        res = compute_multi_dimensional_scores(art)
        assert exp_ko in res.explanation.reasons_ko
        assert exp_en in res.explanation.reasons_en
        # Check no subjective praise words
        for phrase in ["가장 신뢰", "최고의 출처", "most trustworthy", "best source"]:
            assert phrase not in res.explanation.summary_ko
            assert phrase not in res.explanation.summary_en


