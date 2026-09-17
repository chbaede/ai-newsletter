from ai_newsletter.sources import (
    AUTHORITY_HIERARCHY,
    KNOWN_PUBLISHER_AUTHORITY,
    SOURCE_CATALOG,
    classify_source_type,
    get_enabled_sources,
    get_korean_sources,
    get_media_sources,
    get_primary_sources,
    get_research_sources,
    get_source,
    source_authority,
    source_metadata,
)


def test_source_catalog_integrity():
    assert len(SOURCE_CATALOG) > 10
    ids = set()
    for feed in SOURCE_CATALOG:
        assert feed.id, f"Feed missing id: {feed.name}"
        assert feed.id not in ids, f"Duplicate id: {feed.id}"
        ids.add(feed.id)
        assert feed.name
        assert feed.bucket
        assert feed.url.startswith("http")
        assert feed.source_type in AUTHORITY_HIERARCHY
        assert 0 <= feed.authority_score <= 100
        assert feed.region in {"global", "us", "korea", "europe", "asia"}
        assert feed.language in {"en", "ko"}


def test_source_groups():
    primary = get_primary_sources()
    research = get_research_sources()
    media = get_media_sources()
    korean = get_korean_sources()
    enabled = get_enabled_sources()

    assert len(primary) >= 5
    assert len(research) >= 3
    assert len(media) >= 4
    assert len(korean) >= 4
    assert len(enabled) == len(SOURCE_CATALOG)


def test_authority_scoring():
    assert source_authority("OpenAI Newsroom", "official") >= 90
    assert source_authority("NIST AISI", "regulator") == 100
    assert source_authority("AI타임스", "korean_media") >= 85
    assert source_authority("Unknown Blog", "media") == 85


def test_classify_source_type():
    assert classify_source_type("NIST AI Safety Institute") == "regulator"
    assert classify_source_type("OpenAI Research") == "official"
    assert classify_source_type("arXiv Computer Science") == "research"
    assert classify_source_type("TechCrunch") == "media"


def test_get_source():
    feed = get_source("openai_news")
    assert feed is not None
    assert feed.name == "OpenAI Newsroom"
    assert feed.bucket == "frontier_models"

