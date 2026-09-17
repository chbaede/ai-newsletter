from ai_newsletter.taxonomy import (
    CATEGORY_LABELS_EN,
    CATEGORY_LABELS_KO,
    PRIMARY_CATEGORIES,
    classify_taxonomy,
    extract_topics,
)


def test_primary_categories():
    assert "frontier_models" in PRIMARY_CATEGORIES
    assert "hardware_infra" in PRIMARY_CATEGORIES
    assert "agents_coding" in PRIMARY_CATEGORIES
    assert "regulation_policy" in PRIMARY_CATEGORIES

    for cat in PRIMARY_CATEGORIES:
        assert cat in CATEGORY_LABELS_KO
        assert cat in CATEGORY_LABELS_EN


def test_extract_topics():
    topics = extract_topics("OpenAI releases new reasoning model o1 with chain of thought capabilities")
    assert "Reasoning" in topics

    chip_topics = extract_topics("NVIDIA launches Blackwell B200 GPU with HBM3e memory for AI datacenters")
    assert "GPU/Semiconductor" in chip_topics
    assert "Datacenter & Energy" in chip_topics


def test_classify_taxonomy():
    # Frontier model
    res1 = classify_taxonomy(
        title="OpenAI announces GPT-4o with multimodal reasoning",
        excerpt="The new flagship model integrates voice, vision, and text natively.",
        source="OpenAI",
        source_type="official",
    )
    assert res1.primary_category == "frontier_models"
    assert "OpenAI" in res1.entities

    # AI Hardware
    res2 = classify_taxonomy(
        title="NVIDIA announces mass production of Blackwell B200 GPUs",
        excerpt="Datacenter demand for AI accelerators surges across cloud providers.",
        source="TechCrunch",
        source_type="media",
    )
    assert res2.primary_category == "hardware_infra"
    assert "NVIDIA" in res2.entities

    # AI Agent
    res3 = classify_taxonomy(
        title="Cognition announces Devin 2.0 autonomous SWE agent",
        excerpt="New coding assistant scores 45% on SWE-bench benchmark.",
        source="VentureBeat",
        source_type="media",
    )
    assert res3.primary_category == "agents_coding"
    assert "Cognition" in res3.entities

    # Regulation
    res4 = classify_taxonomy(
        title="EU AI Act enforcement begins for general purpose AI models",
        excerpt="Strict transparency and risk assessments required under new compliance framework.",
        source="EU AI Office",
        source_type="regulator",
    )
    assert res4.primary_category == "regulation_policy"
    assert "EU AI Office" in res4.entities

