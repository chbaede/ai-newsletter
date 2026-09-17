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
    anthropic_feed = get_source("anthropic_news")
    assert anthropic_feed is not None
    assert anthropic_feed.discovery_method == "search"
    assert anthropic_feed.evidence_level == "primary"
    assert is_discovery_source(anthropic_feed) is False
    assert is_primary_source(anthropic_feed) is True

    openai_feed = get_source("openai_news")
    assert openai_feed is not None
    assert openai_feed.discovery_method == "rss"
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


def test_no_duplicate_source_ids_or_canonical_urls():
    from ai_newsletter.collector import canonicalize_url

    ids = [f.id for f in SOURCE_CATALOG]
    assert len(ids) == len(set(ids)), f"Duplicate source IDs found: {[x for x in ids if ids.count(x) > 1]}"

    urls = [canonicalize_url(f.url) for f in SOURCE_CATALOG]
    assert len(urls) == len(set(urls)), f"Duplicate canonical feed URLs found: {[x for x in urls if urls.count(x) > 1]}"


def test_new_primary_and_research_sources_metadata():
    # Verify new primary sources
    new_primary_ids = ["mistral_ai", "aws_ml_blog", "apple_ml_research"]
    for sid in new_primary_ids:
        feed = get_source(sid)
        assert feed is not None, f"Missing primary source: {sid}"
        assert feed.url.startswith("http"), f"Invalid URL for {sid}: {feed.url}"
        assert feed.evidence_level == "primary"
        assert feed.is_primary_source is True
        assert feed.is_independent_source is False, f"Company blog {sid} must NOT be marked independent"
        assert feed.is_discovery_source is False
        assert feed.catalog_group == "primary"

    # Verify new research sources
    new_research_ids = ["bair_blog", "google_research", "ibm_research"]
    for sid in new_research_ids:
        feed = get_source(sid)
        assert feed is not None, f"Missing research source: {sid}"
        assert feed.url.startswith("http"), f"Invalid URL for {sid}: {feed.url}"
        assert feed.evidence_level == "research"
        assert feed.is_primary_source is False
        assert feed.is_independent_source is False
        assert feed.is_discovery_source is False
        assert feed.catalog_group == "research"


def test_company_blogs_never_marked_independent():
    company_blog_ids = [
        "openai_news",
        "anthropic_news",
        "deepmind_blog",
        "meta_ai_blog",
        "microsoft_ai",
        "nvidia_news",
        "mistral_ai",
        "huggingface_blog",
        "aws_ml_blog",
        "apple_ml_research",
    ]
    for sid in company_blog_ids:
        feed = get_source(sid)
        assert feed is not None
        assert feed.is_primary_source is True
        assert feed.is_independent_source is False, f"{sid} should NOT be independent"


def test_feed_health_for_new_sources():
    from ai_newsletter.collector import check_feed_health

    new_source_ids = [
        "openai_news",
        "deepmind_blog",
        "meta_ai_blog",
        "nvidia_news",
        "mistral_ai",
        "aws_ml_blog",
        "apple_ml_research",
        "bair_blog",
        "google_research",
        "ibm_research",
    ]
    new_feeds = [get_source(sid) for sid in new_source_ids if get_source(sid)]
    assert len(new_feeds) == len(new_source_ids)

    health_results = check_feed_health(new_feeds)
    for res in health_results:
        assert res["status"] == "ok", f"Feed {res['id']} failed health check: {res['error']}"
        assert res["entries_count"] > 0, f"Feed {res['id']} returned 0 entries"


# ────────────────────────────────────────────────────────────────
# Phase 4: Independent journalism & industry media sources
# ────────────────────────────────────────────────────────────────


def test_independent_journalism_feeds_classification():
    """Reuters, Bloomberg, FT, WSJ, The Information must be evidence_level=independent."""
    independent_ids = [
        "reuters_ai",
        "bloomberg_ai",
        "ft_tech_ai",
        "wsj_tech",
        "the_information_ai",
    ]
    for sid in independent_ids:
        feed = get_source(sid)
        assert feed is not None, f"Missing independent journalism feed: {sid}"
        assert feed.evidence_level == "independent", (
            f"{sid}: expected evidence_level='independent', got '{feed.evidence_level}'"
        )
        assert feed.is_independent_source is True, f"{sid} must be is_independent_source=True"
        assert feed.is_primary_source is False, f"{sid} must NOT be is_primary_source"
        assert feed.is_discovery_source is False, f"{sid} must NOT be is_discovery_source"
        assert feed.url.startswith("http"), f"{sid} has invalid URL: {feed.url}"
        assert feed.catalog_group == "media"


def test_industry_media_new_feeds_classification():
    """Wired and Ars Technica must be evidence_level=industry_media (NOT independent)."""
    industry_ids = ["wired_ai", "arstechnica_ai"]
    for sid in industry_ids:
        feed = get_source(sid)
        assert feed is not None, f"Missing industry media feed: {sid}"
        assert feed.evidence_level == "industry_media", (
            f"{sid}: expected 'industry_media', got '{feed.evidence_level}'"
        )
        assert feed.is_independent_source is False, (
            f"{sid}: industry_media feed must NOT be marked independent"
        )
        assert feed.is_primary_source is False
        assert feed.catalog_group == "media"


def test_independent_vs_primary_are_separate_evidence():
    """
    A Reuters article about an OpenAI announcement is independent evidence.
    The OpenAI announcement itself is primary evidence.
    These must be treated as separate, non-interchangeable evidence items.
    """
    from ai_newsletter.models import Article

    # Primary source: OpenAI's own announcement
    openai_article = Article(
        title="OpenAI launches GPT-5",
        url="https://openai.com/blog/gpt-5",
        source="OpenAI",
        source_type="official",
        evidence_level="primary",
        is_primary_source=True,
        is_independent_source=False,
    )
    assert openai_article.evidence_level == "primary"
    assert openai_article.is_primary_source is True
    assert openai_article.is_independent_source is False

    # Independent source: Reuters reporting the same event
    reuters_article = Article(
        title="OpenAI launches GPT-5 — Reuters",
        url="https://reuters.com/technology/openai-gpt5",
        source="Reuters Technology AI",
        publisher="Reuters",
        source_type="media",
        evidence_level="independent",
        is_primary_source=False,
        is_independent_source=True,
    )
    assert reuters_article.evidence_level == "independent"
    assert reuters_article.is_independent_source is True
    assert reuters_article.is_primary_source is False

    # They have different evidence_level — not interchangeable
    assert openai_article.evidence_level != reuters_article.evidence_level


def test_source_authority_includes_new_publishers():
    """KNOWN_PUBLISHER_AUTHORITY must include independent journalism publishers."""
    from ai_newsletter.sources import KNOWN_PUBLISHER_AUTHORITY, source_authority

    for pub in ("reuters", "bloomberg", "financial times", "wsj", "ft"):
        assert pub in KNOWN_PUBLISHER_AUTHORITY, f"Missing authority entry for: {pub}"
        assert KNOWN_PUBLISHER_AUTHORITY[pub] >= 90

    # source_authority() lookup by name
    assert source_authority("Reuters") >= 90
    assert source_authority("Bloomberg") >= 90
    assert source_authority("Financial Times") >= 90
    assert source_authority("WSJ") >= 90


def test_source_evidence_level_for_independent_journalism():
    """source_evidence_level() must recognise WSJ / FT by name string."""
    from ai_newsletter.sources import source_evidence_level

    assert source_evidence_level("Wall Street Journal") == "independent"
    assert source_evidence_level("wsj") == "independent"
    assert source_evidence_level("financial times") == "independent"
    assert source_evidence_level("reuters") == "independent"
    assert source_evidence_level("bloomberg") == "independent"


def test_no_duplicate_independent_journalism_ids():
    """Sanity: no two independent journalism feeds share an ID or URL."""
    from ai_newsletter.collector import canonicalize_url

    independent_ids = [
        "reuters_ai",
        "bloomberg_ai",
        "ft_tech_ai",
        "wsj_tech",
        "the_information_ai",
        "mit_tech_review",
    ]
    feeds = [get_source(sid) for sid in independent_ids]
    assert all(f is not None for f in feeds), "One or more independent journalism feeds missing"

    urls = [canonicalize_url(f.url) for f in feeds]
    assert len(urls) == len(set(urls)), f"Duplicate canonical URLs: {urls}"

    ids = [f.id for f in feeds]
    assert len(ids) == len(set(ids)), f"Duplicate IDs: {ids}"


def test_independent_journalism_feeds_in_get_independent_sources():
    """get_independent_sources() must include all new independent journalism feeds."""
    from ai_newsletter.sources import get_independent_sources

    independents = {f.id for f in get_independent_sources()}
    expected = {"reuters_ai", "bloomberg_ai", "ft_tech_ai", "wsj_tech", "the_information_ai", "mit_tech_review"}
    missing = expected - independents
    assert not missing, f"Missing from get_independent_sources(): {missing}"


def test_industry_media_not_in_independent_sources():
    """Wired and Ars Technica must NOT appear in get_independent_sources()."""
    from ai_newsletter.sources import get_independent_sources

    independents = {f.id for f in get_independent_sources()}
    assert "wired_ai" not in independents, "Wired should NOT be in independent sources"
    assert "arstechnica_ai" not in independents, "Ars Technica should NOT be in independent sources"


def test_total_feed_count():
    """Verify total catalog has grown to 39 feeds."""
    assert len(SOURCE_CATALOG) == 39, (
        f"Expected 39 feeds in SOURCE_CATALOG, got {len(SOURCE_CATALOG)}"
    )


# ── Tests for Evidence Origin vs. Discovery Mechanism Separation ──────────────

from ai_newsletter.collector import parse_feed_entry
from ai_newsletter.models import Article
from ai_newsletter.sources import SourceFeed


def test_scenario_1_direct_openai_feed():
    """1. Direct OpenAI RSS feed: publisher=OpenAI, evidence_level=primary, discovery_method=rss."""
    openai_feed = get_source("openai_news")
    assert openai_feed is not None
    assert openai_feed.discovery_method == "rss"
    assert openai_feed.evidence_level == "primary"
    assert openai_feed.is_primary_source is True
    assert openai_feed.is_discovery_source is False

    mock_entry = {
        "title": "OpenAI announces GPT-5 frontier model",
        "link": "https://openai.com/news/gpt-5",
        "published": "Thu, 17 Sep 2026 10:00:00 GMT",
        "summary": "Official announcement of GPT-5.",
    }
    parsed = parse_feed_entry(mock_entry, openai_feed)
    assert parsed is not None
    assert parsed.publisher == "OpenAI Newsroom"
    assert parsed.evidence_level == "primary"
    assert parsed.is_primary_source is True
    assert parsed.is_discovery_source is False
    assert parsed.discovered_via == "openai_news"


def test_scenario_2_primary_discovered_via_google_news():
    """2. Primary publisher (Anthropic/OpenAI) discovered via Google News:
    publisher=Anthropic, evidence_level=primary, discovery_method=search/google_news,
    is_primary_source=True, is_discovery_source=False.
    """
    anthropic_feed = get_source("anthropic_news")
    assert anthropic_feed is not None
    assert anthropic_feed.discovery_method == "search"
    assert anthropic_feed.evidence_level == "primary"
    assert anthropic_feed.is_primary_source is True
    assert anthropic_feed.is_discovery_source is False

    mock_entry = {
        "title": "Anthropic introduces Claude 3.5 Sonnet upgrade - Anthropic",
        "link": "https://news.google.com/rss/articles/anthropic-claude",
        "published": "Thu, 17 Sep 2026 10:00:00 GMT",
        "summary": "Anthropic releases Claude 3.5 Sonnet update.",
        "source": {"title": "Anthropic"},
    }
    parsed = parse_feed_entry(mock_entry, anthropic_feed)
    assert parsed is not None
    assert parsed.publisher == "Anthropic"
    assert parsed.evidence_level == "primary"
    assert parsed.is_primary_source is True
    assert parsed.is_discovery_source is False
    assert parsed.discovered_via == "anthropic_news"


def test_scenario_3_independent_media_discovered_via_google_news():
    """3. Independent media (Reuters) discovered via Google News:
    publisher=Reuters, evidence_level=independent, discovery_method=search,
    is_independent_source=True, is_discovery_source=False.
    """
    reuters_feed = get_source("reuters_ai")
    assert reuters_feed is not None
    assert reuters_feed.discovery_method == "search"
    assert reuters_feed.evidence_level == "independent"
    assert reuters_feed.is_independent_source is True
    assert reuters_feed.is_discovery_source is False

    mock_entry = {
        "title": "OpenAI in talks for new funding round - Reuters",
        "link": "https://news.google.com/rss/articles/reuters-openai-funding",
        "published": "Thu, 17 Sep 2026 11:00:00 GMT",
        "summary": "Reuters report on OpenAI funding.",
        "source": {"title": "Reuters"},
    }
    parsed = parse_feed_entry(mock_entry, reuters_feed)
    assert parsed is not None
    assert parsed.publisher == "Reuters"
    assert parsed.evidence_level == "independent"
    assert parsed.is_independent_source is True
    assert parsed.is_discovery_source is False
    assert parsed.discovered_via == "reuters_ai"


def test_scenario_4_unknown_publisher_discovered_via_google_news():
    """4. Unknown publisher discovered via Google News query:
    publisher extracted from entry, evidence_level defaults to feed/industry_media,
    is_discovery_source=False (the article origin has a named publisher).
    """
    custom_search_feed = SourceFeed(
        name="AI Trends Monitor",
        bucket="frontier_models",
        url="https://news.google.com/rss/search?q=AI+Trends",
        source_type="media",
        discovery_method="search",
        evidence_level="industry_media",
    )
    mock_entry = {
        "title": "Small AI startup announces breakthrough - Silicon Valley Herald",
        "link": "https://news.google.com/rss/articles/startup-breakthrough",
        "source": {"title": "Silicon Valley Herald"},
    }
    parsed = parse_feed_entry(mock_entry, custom_search_feed)
    assert parsed is not None
    assert parsed.publisher == "Silicon Valley Herald"
    assert parsed.evidence_level == "industry_media"
    assert parsed.is_primary_source is False
    assert parsed.is_discovery_source is False


def test_scenario_5_actual_google_news_aggregator_article():
    """5. Actual Google News aggregator article without identifiable publisher:
    evidence_level=discovery, is_discovery_source=True.
    """
    gnews_feed = get_source("gnews_global_ai")
    assert gnews_feed is not None
    assert gnews_feed.evidence_level == "discovery"
    assert gnews_feed.is_discovery_source is True
    assert gnews_feed.is_primary_source is False

    mock_entry = {
        "title": "AI Model Index and Overview",
        "link": "https://news.google.com/topics/ai",
        "summary": "Aggregated news overview from Google News.",
    }
    parsed = parse_feed_entry(mock_entry, gnews_feed)
    assert parsed is not None
    assert parsed.publisher == "Google News Global AI"
    assert parsed.evidence_level == "discovery"
    assert parsed.is_discovery_source is True
    assert parsed.is_primary_source is False


