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


def test_sourcefeed_default_behavior():
    from ai_newsletter.sources import SourceFeed
    feed = SourceFeed(name="Minimal Feed", bucket="frontier_models", url="https://example.com/rss")
    assert feed.source_type == "media"
    assert feed.authority_score == 85
    assert feed.evidence_level == "industry_media"
    assert feed.is_primary_source is False
    assert feed.is_independent_source is False
    assert feed.is_discovery_source is False
    assert feed.enabled is True


def test_sourcefeed_backward_compatibility():
    from ai_newsletter.sources import SourceFeed, is_primary_source, is_independent_source, is_discovery_source, get_source_role

    # Official / Regulator without evidence_level specified
    f_official = SourceFeed(name="Official Source", bucket="frontier_models", url="https://example.com", source_type="official")
    assert f_official.evidence_level == "primary"
    assert f_official.is_primary_source is True
    assert is_primary_source(f_official) is True
    assert get_source_role(f_official) == "primary"

    f_reg = SourceFeed(name="Regulator Source", bucket="regulation_policy", url="https://example.com", source_type="regulator")
    assert f_reg.evidence_level == "primary"
    assert f_reg.is_primary_source is True

    # Research without evidence_level specified
    f_research = SourceFeed(name="Research Lab", bucket="research_breakthroughs", url="https://example.com", source_type="research")
    assert f_research.evidence_level == "research"
    assert f_research.is_primary_source is False
    assert f_research.is_independent_source is False

    # Aggregator without evidence_level specified
    f_agg = SourceFeed(name="Aggregator Source", bucket="frontier_models", url="https://example.com", source_type="aggregator")
    assert f_agg.evidence_level == "discovery"
    assert f_agg.is_discovery_source is True
    assert is_discovery_source(f_agg) is True

    # Community without evidence_level specified
    f_comm = SourceFeed(name="Discussion Forum", bucket="open_source", url="https://example.com", source_type="community")
    assert f_comm.evidence_level == "community"
    assert f_comm.is_primary_source is False


def test_primary_source_classification():
    from ai_newsletter.sources import is_primary_source, get_source_role, source_evidence_level

    # Registered feeds
    for feed_id in ["openai_news", "anthropic_news", "deepmind_blog", "nvidia_news", "eu_ai_office", "us_nist_aisi", "msit_kr_ai"]:
        feed = get_source(feed_id)
        assert feed is not None
        assert feed.evidence_level == "primary"
        assert feed.is_primary_source is True
        assert is_primary_source(feed) is True
        assert get_source_role(feed) == "primary"

    # String resolutions
    assert is_primary_source("OpenAI") is True
    assert is_primary_source("Anthropic") is True
    assert is_primary_source("Google DeepMind") is True
    assert is_primary_source("NVIDIA") is True
    assert is_primary_source("Meta AI") is True
    assert is_primary_source("European Commission") is True
    assert is_primary_source("NIST") is True
    assert is_primary_source("과기정통부") is True
    assert source_evidence_level("OpenAI") == "primary"


def test_research_source_classification():
    from ai_newsletter.sources import is_primary_source, is_independent_source, is_discovery_source, get_source_role, source_evidence_level

    for feed_id in ["arxiv_ai", "arxiv_cl", "stanford_hai", "mit_csail"]:
        feed = get_source(feed_id)
        assert feed is not None
        assert feed.evidence_level == "research"
        assert feed.is_primary_source is False
        assert feed.is_independent_source is False
        assert feed.is_discovery_source is False
        assert get_source_role(feed) == "research"

    assert source_evidence_level("arXiv") == "research"
    assert source_evidence_level("Stanford HAI") == "research"
    assert source_evidence_level("MIT CSAIL") == "research"
    assert source_evidence_level("BAIR") == "research"


def test_independent_source_classification():
    from ai_newsletter.sources import (
        is_independent_source,
        is_primary_source,
        get_source_role,
        source_evidence_level,
        get_independent_sources,
    )

    mit_feed = get_source("mit_tech_review")
    assert mit_feed is not None
    assert mit_feed.evidence_level == "independent"
    assert mit_feed.is_independent_source is True
    assert is_independent_source(mit_feed) is True
    assert is_primary_source(mit_feed) is False
    assert get_source_role(mit_feed) == "independent"

    # String lookups for independent journalism
    assert is_independent_source("Reuters") is True
    assert is_independent_source("Bloomberg") is True
    assert is_independent_source("Financial Times") is True
    assert is_independent_source("The Information") is True
    assert is_independent_source("MIT Technology Review") is True
    assert source_evidence_level("Reuters") == "independent"

    # Independent sources list
    independents = get_independent_sources()
    assert any(f.id == "mit_tech_review" for f in independents)


def test_discovery_source_classification_and_google_news_rule():
    from ai_newsletter.sources import (
        SourceFeed,
        is_discovery_source,
        is_primary_source,
        get_source_role,
        get_discovery_sources,
        source_evidence_level,
    )

    # Actual aggregator feeds
    gnews_feed = get_source("gnews_global_ai")
    assert gnews_feed is not None
    assert gnews_feed.evidence_level == "discovery"
    assert gnews_feed.is_discovery_source is True
    assert is_discovery_source(gnews_feed) is True

    gnews_kr = get_source("gnews_kr_ai")
    assert gnews_kr is not None
    assert gnews_kr.evidence_level == "discovery"
    assert gnews_kr.is_discovery_source is True

    # Important Design Rule: discovery_method = "google_news" / "search" does NOT make a source "discovery"
    openai_feed = get_source("openai_news")
    assert openai_feed is not None
    assert openai_feed.discovery_method == "search"
    assert openai_feed.evidence_level == "primary"
    assert is_discovery_source(openai_feed) is False
    assert is_primary_source(openai_feed) is True

    stanford_feed = get_source("stanford_hai")
    assert stanford_feed is not None
    assert stanford_feed.discovery_method == "search"
    assert stanford_feed.evidence_level == "research"
    assert is_discovery_source(stanford_feed) is False

    custom_google_news_discovery_method = SourceFeed(
        name="Anthropic Research Feed",
        bucket="frontier_models",
        url="https://news.google.com/rss/search?q=anthropic",
        discovery_method="google_news",
        source_type="official",
    )
    assert custom_google_news_discovery_method.evidence_level == "primary"
    assert custom_google_news_discovery_method.is_discovery_source is False
    assert custom_google_news_discovery_method.is_primary_source is True

    # Helper function check
    assert is_discovery_source("Google News") is True
    assert is_discovery_source("OpenAI") is False

    discovery_feeds = get_discovery_sources()
    assert len(discovery_feeds) == 2
    assert all(f.is_discovery_source for f in discovery_feeds)


def test_source_catalog_evidence_levels():
    from ai_newsletter.sources import EVIDENCE_LEVELS

    for feed in SOURCE_CATALOG:
        assert feed.evidence_level in EVIDENCE_LEVELS, f"Invalid evidence_level on {feed.id}: {feed.evidence_level}"
        if feed.evidence_level == "primary":
            assert feed.is_primary_source is True
        else:
            assert feed.is_primary_source is False

        if feed.evidence_level == "independent":
            assert feed.is_independent_source is True
        else:
            assert feed.is_independent_source is False

        if feed.evidence_level == "discovery":
            assert feed.is_discovery_source is True
        else:
            assert feed.is_discovery_source is False


