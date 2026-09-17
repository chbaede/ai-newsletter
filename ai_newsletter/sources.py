from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus


EVIDENCE_LEVELS = (
    "primary",
    "research",
    "independent",
    "industry_media",
    "community",
    "discovery",
)


SECTION_LABELS = {
    "big": "주요 AI 뉴스",
    "frontier_models": "프론티어 모델 & LLM",
    "agents_coding": "AI 에이전트 & 코딩",
    "hardware_infra": "AI 반도체 & 인프라",
    "open_source": "오픈소스 & 가중치",
    "research_breakthroughs": "연구 & 브레이크스루",
    "multimodal_media": "멀티모달 & 미디어",
    "enterprise_app": "엔터프라이즈 & 응용",
    "regulation_policy": "정책, 규제 & 안전",
    "industry_vc": "투자, 시장 & 스타트업",
    "conference": "학회 & 컨퍼런스",
    "reference": "참고자료 & 가이드",
}

SECTION_LABELS_EN = {
    "big": "Top AI Stories",
    "frontier_models": "Frontier Models & LLMs",
    "agents_coding": "AI Agents & Coding Tools",
    "hardware_infra": "AI Chips & Infrastructure",
    "open_source": "Open Source & Weights",
    "research_breakthroughs": "Research & Breakthroughs",
    "multimodal_media": "Multimodal & Media",
    "enterprise_app": "Enterprise & Applications",
    "regulation_policy": "Policy, Safety & Regulation",
    "industry_vc": "Industry, VC & Business",
    "conference": "Conferences & Events",
    "reference": "Reference & Resources",
}

SECTION_ORDER = [
    "big",
    "frontier_models",
    "agents_coding",
    "hardware_infra",
    "open_source",
    "research_breakthroughs",
    "multimodal_media",
    "enterprise_app",
    "regulation_policy",
    "industry_vc",
    "conference",
    "reference",
]

AUTHORITY_HIERARCHY: dict[str, int] = {
    "regulator": 100,
    "research": 95,
    "official": 90,
    "media": 85,
    "korean_media": 85,
    "open_source": 85,
    "community": 75,
    "aggregator": 50,
}

KNOWN_PUBLISHER_AUTHORITY: dict[str, int] = {
    # Frontier Labs & Official
    "openai": 95,
    "anthropic": 95,
    "google deepmind": 95,
    "deepmind": 95,
    "meta ai": 90,
    "microsoft": 90,
    "nvidia": 95,
    "hugging face": 90,
    "mistral ai": 90,
    "aws": 85,
    "apple": 90,

    # Research
    "arxiv": 95,
    "stanford hai": 95,
    "mit csail": 95,
    "berkeley bair": 95,
    "papers with code": 90,

    # Global Tech & AI Media
    "mit technology review": 90,
    "techcrunch": 85,
    "the verge": 85,
    "venturebeat": 85,
    "wired": 85,
    "ars technica": 85,
    "the information": 90,
    "marktechpost": 80,

    # Korean Media
    "ai타임스": 85,
    "aitimes": 85,
    "zdnet korea": 85,
    "zdnet": 85,
    "테크m": 85,
    "전자신문": 85,
    "geeknews": 80,
    "블로터": 80,
    "매일경제": 80,
    "한국경제": 80,

    # Regulators
    "eu ai office": 100,
    "nist": 100,
    "과기정통부": 100,
    "개인정보보호위원회": 100,
}


@dataclass(frozen=True, slots=True)
class SourceFeed:
    name: str
    bucket: str
    url: str
    id: str | None = None
    source_type: str = "media"
    authority_score: int | None = None
    region: str = "global"
    language: str = "en"
    catalog_group: str = "media"
    discovery_method: str = "rss"
    enabled: bool = True
    aliases: tuple[str, ...] = ()
    evidence_level: str | None = None
    is_primary_source: bool | None = None
    is_independent_source: bool | None = None
    is_discovery_source: bool | None = None

    def __post_init__(self) -> None:
        if self.id is None:
            clean = re.sub(r"[^a-z0-9_]+", "_", self.name.lower()).strip("_")
            object.__setattr__(self, "id", clean)
        if self.authority_score is None:
            auth = AUTHORITY_HIERARCHY.get(self.source_type, 70)
            object.__setattr__(self, "authority_score", auth)

        # 1. Resolve evidence_level
        level = self.evidence_level
        if level is None:
            if self.is_primary_source is True:
                level = "primary"
            elif self.is_independent_source is True:
                level = "independent"
            elif self.is_discovery_source is True:
                level = "discovery"
            elif self.source_type in ("official", "regulator") or self.catalog_group == "primary":
                level = "primary"
            elif self.source_type == "research" or self.catalog_group == "research":
                level = "research"
            elif self.source_type == "aggregator" or self.catalog_group == "aggregator" or "google news" in self.name.lower():
                level = "discovery"
            elif self.source_type == "community":
                level = "community"
            elif self.id in ("mit_tech_review",) or any(ind in self.name.lower() for ind in ("reuters", "bloomberg", "financial times", "the information", "technology review")):
                level = "independent"
            else:
                level = "industry_media"
        else:
            level = level.strip().lower()

        object.__setattr__(self, "evidence_level", level)

        # 2. Resolve boolean flags
        if self.is_primary_source is None:
            object.__setattr__(self, "is_primary_source", level == "primary")
        if self.is_independent_source is None:
            object.__setattr__(self, "is_independent_source", level == "independent")
        if self.is_discovery_source is None:
            object.__setattr__(self, "is_discovery_source", level == "discovery")


def google_news_rss(query: str) -> str:
    encoded = quote_plus(query)
    return f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"


def google_news_kr_rss(query: str) -> str:
    encoded = quote_plus(query)
    return f"https://news.google.com/rss/search?q={encoded}&hl=ko&gl=KR&ceid=KR:ko"


PRIMARY_SOURCES: list[SourceFeed] = [
    SourceFeed(
        id="openai_news",
        name="OpenAI Newsroom",
        bucket="frontier_models",
        url=google_news_rss('site:openai.com/news OR site:openai.com/index "OpenAI"'),
        source_type="official",
        authority_score=95,
        region="us",
        language="en",
        catalog_group="primary",
        discovery_method="search",
        aliases=("OpenAI", "ChatGPT"),
        evidence_level="primary",
        is_primary_source=True,
    ),
    SourceFeed(
        id="anthropic_news",
        name="Anthropic Announcements",
        bucket="frontier_models",
        url=google_news_rss('site:anthropic.com/news "Anthropic" OR "Claude"'),
        source_type="official",
        authority_score=95,
        region="us",
        language="en",
        catalog_group="primary",
        discovery_method="search",
        aliases=("Anthropic", "Claude"),
        evidence_level="primary",
        is_primary_source=True,
    ),
    SourceFeed(
        id="deepmind_blog",
        name="Google DeepMind Discover",
        bucket="frontier_models",
        url=google_news_rss('site:deepmind.google/discover/blog OR "Google DeepMind" Gemini'),
        source_type="official",
        authority_score=95,
        region="global",
        language="en",
        catalog_group="primary",
        discovery_method="search",
        aliases=("DeepMind", "Google DeepMind", "Gemini"),
        evidence_level="primary",
        is_primary_source=True,
    ),
    SourceFeed(
        id="meta_ai_blog",
        name="Meta AI Research Blog",
        bucket="open_source",
        url=google_news_rss('"Meta AI" "Llama" model announcement OR release'),
        source_type="official",
        authority_score=90,
        region="us",
        language="en",
        catalog_group="primary",
        discovery_method="search",
        aliases=("Meta AI", "Llama"),
        evidence_level="primary",
        is_primary_source=True,
    ),
    SourceFeed(
        id="microsoft_ai",
        name="Microsoft AI Blog",
        bucket="frontier_models",
        url=google_news_rss('site:blogs.microsoft.com/ai "Microsoft AI" OR Copilot'),
        source_type="official",
        authority_score=90,
        region="us",
        language="en",
        catalog_group="primary",
        discovery_method="search",
        aliases=("Microsoft", "Copilot"),
        evidence_level="primary",
        is_primary_source=True,
    ),
    SourceFeed(
        id="nvidia_news",
        name="NVIDIA News & AI Blog",
        bucket="hardware_infra",
        url=google_news_rss('site:nvidianews.nvidia.com OR site:blogs.nvidia.com "AI"'),
        source_type="official",
        authority_score=95,
        region="us",
        language="en",
        catalog_group="primary",
        discovery_method="search",
        aliases=("NVIDIA", "Blackwell", "CUDA"),
        evidence_level="primary",
        is_primary_source=True,
    ),
    SourceFeed(
        id="huggingface_blog",
        name="Hugging Face Blog",
        bucket="open_source",
        url="https://huggingface.co/blog/feed.xml",
        source_type="open_source",
        authority_score=90,
        region="global",
        language="en",
        catalog_group="primary",
        discovery_method="rss",
        aliases=("Hugging Face", "HF"),
        evidence_level="primary",
        is_primary_source=True,
    ),
]

RESEARCH_SOURCES: list[SourceFeed] = [
    SourceFeed(
        id="arxiv_ai",
        name="arXiv Artificial Intelligence",
        bucket="research_breakthroughs",
        url="https://export.arxiv.org/rss/cs.AI",
        source_type="research",
        authority_score=95,
        region="global",
        language="en",
        catalog_group="research",
        discovery_method="rss",
        aliases=("arXiv", "arXiv AI"),
        evidence_level="research",
    ),
    SourceFeed(
        id="arxiv_cl",
        name="arXiv Computation and Language (NLP)",
        bucket="research_breakthroughs",
        url="https://export.arxiv.org/rss/cs.CL",
        source_type="research",
        authority_score=95,
        region="global",
        language="en",
        catalog_group="research",
        discovery_method="rss",
        aliases=("arXiv CL", "arXiv NLP"),
        evidence_level="research",
    ),
    SourceFeed(
        id="stanford_hai",
        name="Stanford HAI Policy & Research",
        bucket="research_breakthroughs",
        url=google_news_rss('site:hai.stanford.edu/news OR "Stanford HAI" AI'),
        source_type="research",
        authority_score=90,
        region="us",
        language="en",
        catalog_group="research",
        discovery_method="search",
        aliases=("Stanford HAI",),
        evidence_level="research",
    ),
    SourceFeed(
        id="mit_csail",
        name="MIT CSAIL AI News",
        bucket="research_breakthroughs",
        url=google_news_rss('site:csail.mit.edu/news "AI" OR "robotics"'),
        source_type="research",
        authority_score=90,
        region="us",
        language="en",
        catalog_group="research",
        discovery_method="search",
        aliases=("MIT CSAIL",),
        evidence_level="research",
    ),
]

MEDIA_SOURCES: list[SourceFeed] = [
    SourceFeed(
        id="techcrunch_ai",
        name="TechCrunch AI",
        bucket="frontier_models",
        url="https://techcrunch.com/category/artificial-intelligence/feed/",
        source_type="media",
        authority_score=85,
        region="us",
        language="en",
        catalog_group="media",
        discovery_method="rss",
        aliases=("TechCrunch",),
        evidence_level="industry_media",
    ),
    SourceFeed(
        id="venturebeat_ai",
        name="VentureBeat AI",
        bucket="enterprise_app",
        url=google_news_rss('site:venturebeat.com/category/ai OR "VentureBeat" AI'),
        source_type="media",
        authority_score=85,
        region="us",
        language="en",
        catalog_group="media",
        discovery_method="search",
        aliases=("VentureBeat",),
        evidence_level="industry_media",
    ),
    SourceFeed(
        id="theverge_ai",
        name="The Verge AI",
        bucket="frontier_models",
        url=google_news_rss('site:theverge.com "AI" OR "Artificial Intelligence"'),
        source_type="media",
        authority_score=85,
        region="us",
        language="en",
        catalog_group="media",
        discovery_method="search",
        aliases=("The Verge",),
        evidence_level="industry_media",
    ),
    SourceFeed(
        id="marktechpost",
        name="MarkTechPost AI Research",
        bucket="research_breakthroughs",
        url=google_news_rss('site:marktechpost.com "AI"'),
        source_type="media",
        authority_score=80,
        region="global",
        language="en",
        catalog_group="media",
        discovery_method="search",
        aliases=("MarkTechPost",),
        evidence_level="industry_media",
    ),
    SourceFeed(
        id="mit_tech_review",
        name="MIT Technology Review AI",
        bucket="research_breakthroughs",
        url=google_news_rss('site:technologyreview.com "artificial intelligence"'),
        source_type="media",
        authority_score=90,
        region="us",
        language="en",
        catalog_group="media",
        discovery_method="search",
        aliases=("MIT Tech Review",),
        evidence_level="independent",
        is_independent_source=True,
    ),
]

KOREAN_SOURCES: list[SourceFeed] = [
    SourceFeed(
        id="aitimes_kr",
        name="AI타임스 (AITimes)",
        bucket="frontier_models",
        url=google_news_kr_rss('site:aitimes.com AI OR 인공지능 OR 생성형'),
        source_type="korean_media",
        authority_score=85,
        region="korea",
        language="ko",
        catalog_group="korean",
        discovery_method="search",
        aliases=("AI타임스", "AITimes"),
        evidence_level="industry_media",
    ),
    SourceFeed(
        id="zdnet_ai_kr",
        name="ZDNet Korea AI",
        bucket="enterprise_app",
        url=google_news_kr_rss('site:zdnet.co.kr "AI" OR "인공지능"'),
        source_type="korean_media",
        authority_score=85,
        region="korea",
        language="ko",
        catalog_group="korean",
        discovery_method="search",
        aliases=("ZDNet Korea", "지디넷"),
        evidence_level="industry_media",
    ),
    SourceFeed(
        id="techm_kr",
        name="테크M (TechM AI)",
        bucket="frontier_models",
        url=google_news_kr_rss('site:techm.kr "AI" OR "생성형 AI"'),
        source_type="korean_media",
        authority_score=85,
        region="korea",
        language="ko",
        catalog_group="korean",
        discovery_method="search",
        aliases=("테크M",),
        evidence_level="industry_media",
    ),
    SourceFeed(
        id="etnews_ai",
        name="전자신문 AI",
        bucket="hardware_infra",
        url=google_news_kr_rss('site:etnews.com "인공지능" OR "반도체" "HBM"'),
        source_type="korean_media",
        authority_score=85,
        region="korea",
        language="ko",
        catalog_group="korean",
        discovery_method="search",
        aliases=("전자신문",),
        evidence_level="industry_media",
    ),
    SourceFeed(
        id="geeknews_ai",
        name="GeekNews AI Topics",
        bucket="open_source",
        url=google_news_kr_rss('site:news.hada.io "AI" OR "LLM" OR "오픈소스"'),
        source_type="community",
        authority_score=80,
        region="korea",
        language="ko",
        catalog_group="korean",
        discovery_method="search",
        aliases=("GeekNews", "긱뉴스"),
        evidence_level="community",
    ),
]

REGULATION_SOURCES: list[SourceFeed] = [
    SourceFeed(
        id="eu_ai_office",
        name="EU AI Office & Regulations",
        bucket="regulation_policy",
        url=google_news_rss('"EU AI Act" OR "European AI Office" compliance implementation'),
        source_type="regulator",
        authority_score=100,
        region="europe",
        language="en",
        catalog_group="primary",
        discovery_method="search",
        aliases=("EU AI Office", "EU AI Act"),
        evidence_level="primary",
        is_primary_source=True,
    ),
    SourceFeed(
        id="us_nist_aisi",
        name="NIST US AI Safety Institute",
        bucket="regulation_policy",
        url=google_news_rss('"NIST" "AI Safety Institute" OR "AISI" standards'),
        source_type="regulator",
        authority_score=100,
        region="us",
        language="en",
        catalog_group="primary",
        discovery_method="search",
        aliases=("NIST", "AISI"),
        evidence_level="primary",
        is_primary_source=True,
    ),
    SourceFeed(
        id="msit_kr_ai",
        name="대한민국 과기정통부 AI 정책",
        bucket="regulation_policy",
        url=google_news_kr_rss('site:msit.go.kr "인공지능" OR "AI기본법"'),
        source_type="regulator",
        authority_score=100,
        region="korea",
        language="ko",
        catalog_group="primary",
        discovery_method="search",
        aliases=("과기정통부", "MSIT"),
        evidence_level="primary",
        is_primary_source=True,
    ),
]

AGGREGATOR_SOURCES: list[SourceFeed] = [
    SourceFeed(
        id="gnews_global_ai",
        name="Google News Global AI",
        bucket="frontier_models",
        url="https://news.google.com/rss/search?q=Artificial+Intelligence+LLM+OR+GenAI&hl=en-US&gl=US&ceid=US:en",
        source_type="aggregator",
        authority_score=50,
        region="global",
        language="en",
        catalog_group="aggregator",
        discovery_method="rss",
        evidence_level="discovery",
        is_discovery_source=True,
    ),
    SourceFeed(
        id="gnews_kr_ai",
        name="구글 뉴스 한국 AI",
        bucket="frontier_models",
        url="https://news.google.com/rss/search?q=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5+AI+%EC%83%9D%EC%84%B1%ED%98%95&hl=ko&gl=KR&ceid=KR:ko",
        source_type="aggregator",
        authority_score=50,
        region="korea",
        language="ko",
        catalog_group="aggregator",
        discovery_method="rss",
        evidence_level="discovery",
        is_discovery_source=True,
    ),
]

SOURCE_CATALOG: tuple[SourceFeed, ...] = tuple(
    PRIMARY_SOURCES
    + RESEARCH_SOURCES
    + MEDIA_SOURCES
    + KOREAN_SOURCES
    + REGULATION_SOURCES
    + AGGREGATOR_SOURCES
)

DEFAULT_FEEDS = SOURCE_CATALOG
_SOURCE_BY_ID: dict[str, SourceFeed] = {feed.id: feed for feed in SOURCE_CATALOG if feed.id}


def get_enabled_sources() -> list[SourceFeed]:
    return [feed for feed in SOURCE_CATALOG if feed.enabled]


def get_primary_sources() -> list[SourceFeed]:
    return [feed for feed in SOURCE_CATALOG if feed.catalog_group == "primary"]


def get_research_sources() -> list[SourceFeed]:
    return [feed for feed in SOURCE_CATALOG if feed.catalog_group == "research"]


def get_media_sources() -> list[SourceFeed]:
    return [feed for feed in SOURCE_CATALOG if feed.catalog_group == "media"]


def get_korean_sources() -> list[SourceFeed]:
    return [feed for feed in SOURCE_CATALOG if feed.catalog_group == "korean"]


def get_source(source_id: str) -> SourceFeed | None:
    return _SOURCE_BY_ID.get(source_id)


def source_metadata(source_name: str) -> SourceFeed | None:
    name_clean = source_name.strip().lower()
    for feed in SOURCE_CATALOG:
        if feed.name.lower() == name_clean or feed.id == name_clean:
            return feed
        if any(alias.lower() == name_clean for alias in feed.aliases):
            return feed
    return None


def source_authority(source_name: str, source_type: str = "media") -> int:
    name_clean = source_name.strip().lower()
    for pub, score in KNOWN_PUBLISHER_AUTHORITY.items():
        if pub in name_clean:
            return score
    meta = source_metadata(source_name)
    if meta and meta.authority_score is not None:
        return meta.authority_score
    return AUTHORITY_HIERARCHY.get(source_type, 70)


def classify_source_type(source_name: str) -> str:
    meta = source_metadata(source_name)
    if meta:
        return meta.source_type
    name_clean = source_name.lower()
    if any(k in name_clean for k in ["nist", "office", "commission", "과기정통부", "위원회"]):
        return "regulator"
    if any(k in name_clean for k in ["arxiv", "hai", "csail", "bair"]):
        return "research"
    if any(k in name_clean for k in ["openai", "anthropic", "deepmind", "nvidia", "meta"]):
        return "official"
    return "media"


def source_evidence_level(source: SourceFeed | Any | str | None) -> str:
    """Return the evidence level for a source object, name, or ID.

    Values:
    - 'primary': Organization directly responsible for announcement, model, product, research or policy.
    - 'research': Academic/research institutions or repositories (e.g. arXiv, Stanford HAI, MIT CSAIL).
    - 'independent': Independent journalism/reporting (e.g. Reuters, Bloomberg, FT, The Information, MIT Tech Review).
    - 'industry_media': Technology-focused publications (e.g. TechCrunch, The Verge, VentureBeat).
    - 'community': Discussion/community forums (e.g. Hacker News, GeekNews).
    - 'discovery': Aggregators/search systems (e.g. Google News).
    """
    if source is None:
        return "industry_media"
    if isinstance(source, SourceFeed):
        return source.evidence_level
    if hasattr(source, "evidence_level") and getattr(source, "evidence_level"):
        return str(getattr(source, "evidence_level"))

    source_str = str(source).strip()
    meta = get_source(source_str) or source_metadata(source_str)
    if meta is not None:
        return meta.evidence_level

    clean = source_str.lower()
    # Check discovery systems (Google News, aggregators)
    if any(disc in clean for disc in ("google news", "gnews")):
        return "discovery"
    # Check known independent journalism
    if any(ind in clean for ind in ("reuters", "bloomberg", "financial times", "the information", "mit technology review", "technology review")):
        return "independent"
    # Check known primary sources (frontier labs, chipmakers, regulators)
    if any(prim in clean for prim in ("openai", "anthropic", "deepmind", "google deepmind", "nvidia", "meta ai", "european commission", "eu ai office", "nist", "과기정통부", "microsoft ai", "hugging face")):
        return "primary"
    # Check known research repositories & academic labs
    if any(res in clean for res in ("arxiv", "stanford hai", "mit csail", "bair", "berkeley bair", "papers with code")):
        return "research"
    # Check known community
    if any(comm in clean for comm in ("hacker news", "geeknews", "reddit")):
        return "community"

    return "industry_media"


def get_source_role(source: SourceFeed | Any | str | None) -> str:
    """Return the role / evidence level of the given source."""
    return source_evidence_level(source)


def is_primary_source(source: SourceFeed | Any | str | None) -> bool:
    """Return True if the source is the primary entity behind the announcement, model, research or policy."""
    if isinstance(source, SourceFeed):
        return source.is_primary_source
    if hasattr(source, "is_primary_source") and getattr(source, "is_primary_source") is not None:
        return bool(getattr(source, "is_primary_source"))
    return source_evidence_level(source) == "primary"


def is_independent_source(source: SourceFeed | Any | str | None) -> bool:
    """Return True if the source is an independent journalism or investigative reporting outlet."""
    if isinstance(source, SourceFeed):
        return source.is_independent_source
    if hasattr(source, "is_independent_source") and getattr(source, "is_independent_source") is not None:
        return bool(getattr(source, "is_independent_source"))
    return source_evidence_level(source) == "independent"


def is_discovery_source(source: SourceFeed | Any | str | None) -> bool:
    """Return True if the source is an aggregator / discovery feed (e.g. Google News)."""
    if isinstance(source, SourceFeed):
        return source.is_discovery_source
    if hasattr(source, "is_discovery_source") and getattr(source, "is_discovery_source") is not None:
        return bool(getattr(source, "is_discovery_source"))
    return source_evidence_level(source) == "discovery"


def get_independent_sources() -> list[SourceFeed]:
    """Return all configured sources classified as independent journalism."""
    return [feed for feed in SOURCE_CATALOG if feed.is_independent_source]


def get_discovery_sources() -> list[SourceFeed]:
    """Return all configured aggregator/discovery sources."""
    return [feed for feed in SOURCE_CATALOG if feed.is_discovery_source]


