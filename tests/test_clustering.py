from ai_newsletter.clustering import cluster_articles, select_primary_article
from ai_newsletter.models import Article


def test_cluster_articles_empty():
    events, event_articles, updated = cluster_articles([])
    assert events == []
    assert event_articles == []
    assert updated == []


def test_cluster_articles_grouping():
    a1 = Article(
        title="OpenAI announces GPT-4o with omni modal intelligence",
        url="https://openai.com/index/hello-gpt-4o",
        source="OpenAI Newsroom",
        category="frontier_models",
        is_official=True,
        source_type="official",
        score=95.0,
        tags=["OpenAI"],
    )
    a2 = Article(
        title="OpenAI launches GPT-4o multimodal model for all users",
        url="https://techcrunch.com/openai-gpt4o",
        source="TechCrunch",
        category="frontier_models",
        is_official=False,
        source_type="media",
        score=85.0,
        tags=["OpenAI"],
    )
    a3 = Article(
        title="NVIDIA announces Blackwell B200 shipments for AI servers",
        url="https://nvidianews.nvidia.com/blackwell",
        source="NVIDIA",
        category="hardware_infra",
        is_official=True,
        source_type="official",
        score=92.0,
        tags=["NVIDIA"],
    )

    events, event_articles, updated = cluster_articles([a1, a2, a3])
    assert len(events) == 2, f"Expected 2 events, got {len(events)}"

    # Check that a1 was chosen as primary for OpenAI event
    openai_event = next(e for e in events if "GPT-4o" in e.title or "gpt-4o" in e.title.lower())
    assert openai_event.source_count == 2
    assert openai_event.has_official_source is True
    assert openai_event.official_source_url == "https://openai.com/index/hello-gpt-4o"


def test_select_primary_article():
    official = Article(
        title="Official Announcement",
        url="https://official.com",
        source="OpenAI",
        is_official=True,
        source_type="official",
        score=90.0,
    )
    media = Article(
        title="Media Reporting",
        url="https://media.com",
        source="TechCrunch",
        is_official=False,
        source_type="media",
        score=85.0,
    )
    chosen = select_primary_article([media, official])
    assert chosen == official


# ── Regression Tests for Layered Event Clustering ────────────────────────────


def test_clustering_false_negative_cross_publisher_same_event():
    """False negative: Same event, slightly different wording, different publishers clusters together."""
    a1 = Article(
        title="OpenAI and Anthropic announce new AI safety initiative",
        url="https://techcrunch.com/openai-anthropic-safety",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI", "Anthropic"],
    )
    a2 = Article(
        title="OpenAI announces partnership with Anthropic on AI safety",
        url="https://reuters.com/openai-anthropic-partnership",
        source="Reuters",
        category="frontier_models",
        tags=["OpenAI", "Anthropic"],
    )

    events, event_articles, updated = cluster_articles([a1, a2])
    assert len(events) == 1, f"Expected 1 joint event, got {len(events)}"
    assert events[0].source_count == 2
    assert "safety" in events[0].title.lower() or "ai" in events[0].title.lower()


def test_clustering_same_company_different_events_remain_separate():
    """Same company, different event: Must NOT be merged merely because they share a company name."""
    a1 = Article(
        title="OpenAI releases GPT-4o flagship model with multimodal capabilities",
        url="https://openai.com/gpt-4o",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI"],
    )
    a2 = Article(
        title="OpenAI opens new corporate office in London for European expansion",
        url="https://theverge.com/openai-london-office",
        source="The Verge",
        category="frontier_models",
        tags=["OpenAI"],
    )

    events, event_articles, updated = cluster_articles([a1, a2])
    assert len(events) == 2, f"Expected 2 separate events, got {len(events)}"


def test_clustering_same_model_different_events_remain_separate():
    """Same model, different event: Release vs Lawsuit clashing actions must remain separate."""
    a1 = Article(
        title="OpenAI launches GPT-4o voice mode for mobile subscribers",
        url="https://techcrunch.com/openai-gpt4o-voice",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI", "GPT-4o"],
    )
    a2 = Article(
        title="OpenAI sued over GPT-4o training data copyright infringement",
        url="https://reuters.com/openai-gpt4o-lawsuit",
        source="Reuters",
        category="frontier_models",
        tags=["OpenAI", "GPT-4o"],
    )

    events, event_articles, updated = cluster_articles([a1, a2])
    assert len(events) == 2, f"Expected 2 separate events for release vs lawsuit, got {len(events)}"


def test_clustering_different_models_same_broader_event():
    """Different models, same broader event: Joint safety framework/testing clusters together."""
    a1 = Article(
        title="US AI Safety Institute to evaluate OpenAI GPT-4o and Anthropic Claude 3 models",
        url="https://techcrunch.com/us-aisi-testing",
        source="TechCrunch",
        category="frontier_models",
        tags=["US NIST AISI", "OpenAI", "Anthropic"],
    )
    a2 = Article(
        title="US AISI announces joint safety evaluations for frontier models with Anthropic and OpenAI",
        url="https://reuters.com/us-aisi-safety-evaluation",
        source="Reuters",
        category="frontier_models",
        tags=["US NIST AISI", "OpenAI", "Anthropic"],
    )

    events, event_articles, updated = cluster_articles([a1, a2])
    assert len(events) == 1, f"Expected 1 broader event cluster, got {len(events)}"
    assert events[0].source_count == 2


def test_clustering_multilingual_korean_english_coverage():
    """Multilingual coverage: Korean and English reporting of the same event cluster together."""
    a_en = Article(
        title="OpenAI announces GPT-4o flagship model with omni modal intelligence",
        title_ko="오픈AI, 플래그십 AI 모델 GPT-4o 발표",
        url="https://openai.com/gpt-4o",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI", "GPT-4o"],
    )
    a_ko = Article(
        title="오픈AI, 음성 및 비전 지원하는 플래그십 AI 모델 GPT-4o 공개",
        title_en="OpenAI launches flagship AI model GPT-4o with voice and vision",
        url="https://zdnet.co.kr/news/gpt4o-launch",
        source="ZDNet Korea",
        category="frontier_models",
        tags=["OpenAI", "GPT-4o"],
    )

    events, event_articles, updated = cluster_articles([a_en, a_ko])
    assert len(events) == 1, f"Expected 1 bilingual cluster, got {len(events)}"
    assert events[0].source_count == 2


