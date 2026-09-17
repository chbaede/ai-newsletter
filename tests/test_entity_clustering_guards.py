from ai_newsletter.clustering import calculate_event_similarity
from ai_newsletter.entity_registry import find_entities_in_text, resolve_canonical_entity
from ai_newsletter.models import Article


def test_entity_resolution():
    e1 = resolve_canonical_entity("chatgpt")
    assert e1 is not None
    assert e1.canonical_name == "OpenAI"

    e2 = resolve_canonical_entity("claude 3.5")
    assert e2 is not None
    assert e2.canonical_name == "Anthropic"

    e3 = resolve_canonical_entity("엔비디아")
    assert e3 is not None
    assert e3.canonical_name == "NVIDIA"

    e4 = resolve_canonical_entity("하이퍼클로바x")
    assert e4 is not None
    assert e4.canonical_name == "Naver"


def test_find_entities_in_text():
    text = "Anthropic challenges OpenAI with Claude 3.5 Sonnet release while NVIDIA powers compute."
    found = find_entities_in_text(text)
    assert "Anthropic" in found
    assert "OpenAI" in found
    assert "NVIDIA" in found


def test_competitor_guard_prevents_clustering():
    a1 = Article(
        title="OpenAI releases new ChatGPT voice mode",
        url="https://example.com/openai-voice",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI"],
    )
    a2 = Article(
        title="Anthropic introduces Claude 3.5 Sonnet updates",
        url="https://example.com/anthropic-claude",
        source="VentureBeat",
        category="frontier_models",
        tags=["Anthropic"],
    )

    sim = calculate_event_similarity(a1, a2)
    assert sim == 0.0, "Competitors should not cluster together"


def test_model_version_guard_prevents_clustering():
    a1 = Article(
        title="OpenAI announces GPT-4o with multimodal features",
        url="https://example.com/gpt4o",
        source="TechCrunch",
        category="frontier_models",
    )
    a2 = Article(
        title="OpenAI announces o1 reasoning model with chain of thought",
        url="https://example.com/o1",
        source="The Verge",
        category="frontier_models",
    )

    sim = calculate_event_similarity(a1, a2)
    assert sim == 0.0, "Different models should not cluster together"


def test_same_event_syndication_clusters_together():
    a1 = Article(
        title="OpenAI unveils ChatGPT Search to challenge Google",
        url="https://techcrunch.com/openai-search",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI"],
    )
    a2 = Article(
        title="OpenAI launches ChatGPT Search engine integration",
        url="https://theverge.com/openai-search",
        source="The Verge",
        category="frontier_models",
        tags=["OpenAI"],
    )

    sim = calculate_event_similarity(a1, a2)
    assert sim >= 0.60, f"Same event from multiple outlets should cluster (sim={sim})"

