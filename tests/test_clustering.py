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

