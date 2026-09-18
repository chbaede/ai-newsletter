from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from urllib.parse import urlsplit

from .models import Article, NewsletterIssue
from .priority import LEVELS, assess_priority
from .sources import source_evidence_level
from .taxonomy import CATEGORY_LABELS_EN, CATEGORY_LABELS_KO

HANGUL_RE = re.compile(r"[가-힣]")


@dataclass(frozen=True, slots=True)
class RegionSignal:
    key: str
    label_ko: str
    label_en: str


UI_REGION_FILTERS = [
    RegionSignal("all", "전체", "All"),
    RegionSignal("global", "글로벌", "Global"),
    RegionSignal("us", "미국", "United States"),
    RegionSignal("korea", "한국", "South Korea"),
    RegionSignal("europe", "유럽", "Europe"),
    RegionSignal("asia", "아시아", "Asia"),
]

REGION_FILTERS = UI_REGION_FILTERS

REGION_TERMS = {
    "us": [
        "us", "u.s.", "united states", "america", "american", "california",
        "silicon valley", "san francisco", "new york", "washington", "austin",
        "seattle", "sec", "ftc", "nist", "white house", "congress", "senate",
    ],
    "korea": [
        "korea", "korean", "seoul", "south korea", "과기정통부", "한국", "삼성",
        "sk하이닉스", "네이버", "카카오", "lg", "ai타임스", "지디넷", "전자신문",
        "매일경제", "한국경제", "긱뉴스",
    ],
    "europe": [
        "europe", "european", "eu", "eu ai act", "brussels", "germany", "france",
        "uk", "britain", "london", "paris", "berlin", "mistral", "asml",
        "european commission", "cma", "gdpr",
    ],
    "asia": [
        "asia", "china", "japan", "taiwan", "singapore", "tsmc", "kuaishou", "tokyo",
        "beijing", "alibabai", "bytedance", "korea",
    ],
}


@dataclass(frozen=True, slots=True)
class TopicFilter:
    key: str
    label: str
    terms: tuple[str, ...]


TOPIC_FILTERS = [
    TopicFilter("all", "All Topics", ()),
    TopicFilter("frontier_models", "Frontier Models", ("llm", "foundation model", "gpt", "claude", "gemini", "llama", "frontier")),
    TopicFilter("reasoning", "Reasoning & CoT", ("reasoning", "o1", "o3", "chain of thought", "cot", "추론")),
    TopicFilter("agents_coding", "Agents & Coding", ("agent", "agents", "coding", "cursor", "devin", "copilot", "swe-bench", "에이전트")),
    TopicFilter("hardware_infra", "Chips & Infra", ("gpu", "blackwell", "b200", "h100", "hbm", "cuda", "datacenter", "npu", "반도체")),
    TopicFilter("open_source", "Open Source", ("open source", "open weights", "hugging face", "vllm", "ollama", "오픈소스")),
    TopicFilter("multimodal", "Multimodal", ("multimodal", "vision", "video gen", "sora", "runway", "voice ai", "멀티모달")),
    TopicFilter("regulation_policy", "Policy & Safety", ("eu ai act", "safety", "alignment", "regulation", "copyright", "규제", "안전")),
]


@dataclass(frozen=True, slots=True)
class SourceTypeFilter:
    key: str
    label_en: str
    label_ko: str


SOURCE_TYPE_FILTERS = [
    SourceTypeFilter("all", "All", "전체"),
    SourceTypeFilter("primary", "Primary", "1차 출처"),
    SourceTypeFilter("research", "Research", "연구"),
    SourceTypeFilter("independent", "Independent", "독립 취재"),
    SourceTypeFilter("industry_media", "Industry Media", "산업 미디어"),
    SourceTypeFilter("community", "Community", "커뮤니티"),
    SourceTypeFilter("discovery", "Discovery", "검색 집계"),
]

INTELLIGENCE_SECTION_DEFINITIONS = [
    {
        "key": "top_stories",
        "label_en": "Top Stories",
        "label_ko": "주요 AI 뉴스",
        "subtitle_en": "High-impact artificial intelligence breakthroughs and strategic moves",
        "subtitle_ko": "가장 큰 파급력을 가진 핵심 AI 발표 및 전략 동향",
        "categories": {"big"},
    },
    {
        "key": "frontier_models",
        "label_en": "Frontier Models & LLMs",
        "label_ko": "프론티어 모델 & LLM",
        "subtitle_en": "Next-gen foundation models, reasoning, and major lab developments",
        "subtitle_ko": "차세대 파운데이션 모델, 추론 아키텍처 및 선도 랩 동향",
        "categories": {"frontier_models"},
    },
    {
        "key": "agents_coding",
        "label_en": "AI Agents & Coding Tools",
        "label_ko": "AI 에이전트 & 코딩 도구",
        "subtitle_en": "Autonomous software engineering, SWE benchmarks, and agentic workflows",
        "subtitle_ko": "자율 소프트웨어 개발, SWE-bench 및 워크플로 자동화",
        "categories": {"agents_coding"},
    },
    {
        "key": "hardware_infra",
        "label_en": "AI Chips & Infrastructure",
        "label_ko": "AI 반도체 & 인프라",
        "subtitle_en": "Accelerators, HBM, compute clusters, datacenters and power grids",
        "subtitle_ko": "차세대 가속기, HBM 메모리, 데이터센터 및 전력망 인프라",
        "categories": {"hardware_infra"},
    },
    {
        "key": "open_source",
        "label_en": "Open Source & Weights",
        "label_ko": "오픈소스 AI & 가중치",
        "subtitle_en": "Open weight releases, fine-tuning, inference engines, and tooling",
        "subtitle_ko": "오픈 모델 공개, 파인튜닝 기법 및 고속 추론 런타임",
        "categories": {"open_source"},
    },
    {
        "key": "research_breakthroughs",
        "label_en": "Research & Breakthroughs",
        "label_ko": "AI 연구 & 브레이크스루",
        "subtitle_en": "arXiv preprints, novel architectures, and academic research",
        "subtitle_ko": "arXiv 주요 논문, 신규 아키텍처 및 학술 연구 성과",
        "categories": {"research_breakthroughs"},
    },
    {
        "key": "multimodal_media",
        "label_en": "Multimodal & Generative Media",
        "label_ko": "멀티모달 & 생성 미디어",
        "subtitle_en": "Video generation, 3D, voice synthesis, and creative GenAI",
        "subtitle_ko": "영상 생성, 음성 합성 및 차세대 생성형 크리에이티브 도구",
        "categories": {"multimodal_media"},
    },
    {
        "key": "enterprise_app",
        "label_en": "Enterprise & Applications",
        "label_ko": "엔터프라이즈 & 산업 응용",
        "subtitle_en": "Industry deployments, business automation, and healthcare/robotics",
        "subtitle_ko": "기업용 업무 자동화, 버티컬 AI 및 피지컬 로보틱스 응용",
        "categories": {"enterprise_app"},
    },
    {
        "key": "regulation_policy",
        "label_en": "Policy, Safety & Governance",
        "label_ko": "정책, 규제 & AI 안전",
        "subtitle_en": "EU AI Act compliance, safety standards, copyright, and antitrust",
        "subtitle_ko": "글로벌 규제 컴플라이언스, 안전 프레임워크, 저작권 및 정책",
        "categories": {"regulation_policy"},
    },
    {
        "key": "industry_vc",
        "label_en": "Industry, VC & Business",
        "label_ko": "투자, 시장 & 스타트업",
        "subtitle_en": "Venture funding, valuations, M&A, and commercial dynamics",
        "subtitle_ko": "대규모 투자 유치, 기업가치 평가, M&A 및 빅테크 제휴",
        "categories": {"industry_vc"},
    },
]


@dataclass(frozen=True, slots=True)
class SourceBadge:
    key: str
    label_en: str
    label_ko: str
    css_class: str


EVIDENCE_BADGES = {
    "primary": SourceBadge("primary", "Primary", "1차 출처", "badge-primary"),
    "research": SourceBadge("research", "Research", "연구", "badge-research"),
    "independent": SourceBadge("independent", "Independent", "독립 취재", "badge-independent"),
    "industry_media": SourceBadge("industry_media", "Industry Media", "산업 미디어", "badge-industry-media"),
    "community": SourceBadge("community", "Community", "커뮤니티", "badge-community"),
    "discovery": SourceBadge("discovery", "Discovery", "검색 집계", "badge-discovery"),
}


@dataclass(frozen=True, slots=True)
class SourceTransparency:
    evidence_badge: SourceBadge
    sources_display: str
    evidence_label_en: str
    evidence_label_ko: str
    confidence_label: str
    confidence_label_ko: str
    confidence_explanation_en: str
    confidence_explanation_ko: str
    is_multi_source: bool
    independent_source_count: int
    primary_sources: list[str]
    independent_sources: list[str]
    other_sources: list[str]


@dataclass(frozen=True, slots=True)
class ArticleView:
    article: Article
    priority: Any
    source_url: str
    published_time: str
    regions: list[RegionSignal]
    region_keys: list[str]
    all_region_keys: list[str]
    topic_keys: list[str]
    source_type: str
    title_ko: str
    title_en: str
    factual_summary_ko: str
    factual_summary_en: str
    why_it_matters_ko: str
    why_it_matters_en: str
    primary_category_ko: str
    primary_category_en: str
    coverage: Any
    user_tags: list[str]
    publisher_display: str
    transparency: SourceTransparency
    key_points: list[str]
    topics_display: str = ""


def regions_for_article(article: Article) -> list[RegionSignal]:
    text = f"{article.title} {article.excerpt} {' '.join(article.tags)}".lower()
    matched = []
    for key in ["us", "korea", "europe", "asia"]:
        if any(term in text for term in REGION_TERMS[key]):
            matched.append(key)
    if not matched:
        matched = ["global"]
    return [next(r for r in REGION_FILTERS if r.key == key) for key in matched]


def all_region_keys_for_article(article: Article) -> list[str]:
    regs = regions_for_article(article)
    keys = ["all"] + [r.key for r in regs]
    return list(dict.fromkeys(keys))


def region_counts(articles: list[Article]) -> dict[str, int]:
    counts = {r.key: 0 for r in REGION_FILTERS}
    counts["all"] = len(articles)
    for a in articles:
        for r in regions_for_article(a):
            counts[r.key] = counts.get(r.key, 0) + 1
    return counts


def topic_keys_for_article(article: Article) -> list[str]:
    text = f"{article.title} {article.excerpt} {' '.join(article.topics)} {' '.join(article.tags)}".lower()
    keys = ["all"]
    for tf in TOPIC_FILTERS:
        if tf.key == "all":
            continue
        if any(term in text for term in tf.terms):
            keys.append(tf.key)
    return keys


def topic_counts(articles: list[Article]) -> dict[str, int]:
    counts = {t.key: 0 for t in TOPIC_FILTERS}
    counts["all"] = len(articles)
    for a in articles:
        for t_key in topic_keys_for_article(a):
            if t_key != "all":
                counts[t_key] = counts.get(t_key, 0) + 1
    return counts


def source_type_counts(articles: list[Article]) -> dict[str, int]:
    counts = {s.key: 0 for s in SOURCE_TYPE_FILTERS}
    counts["all"] = len(articles)
    for a in articles:
        st = canonical_source_type(a)
        if st in counts:
            counts[st] += 1
    return counts


def source_type_keys_for_article(article: Article) -> list[str]:
    return ["all", canonical_source_type(article)]


def canonical_source_type(article: Article) -> str:
    """Return the canonical evidence level for UI source filtering.

    Maps directly to the backend evidence taxonomy:
    primary, research, independent, industry_media, community, discovery.
    Supports backward-compatibility with legacy source_type values.
    """
    if article.evidence_level:
        lvl = article.evidence_level.strip().lower()
        if lvl in {"primary", "research", "independent", "industry_media", "community", "discovery"}:
            return lvl
    if article.is_primary_source:
        return "primary"
    if article.is_independent_source:
        return "independent"
    if article.is_discovery_source:
        return "discovery"

    # Fallback to source_evidence_level inference
    return source_evidence_level(article.publisher or article.source or article.source_type)


def display_title_ko(article: Article) -> str:
    if article.title_ko:
        return article.title_ko
    return article.title


def display_title_en(article: Article) -> str:
    if article.title_en:
        return article.title_en
    return article.title


def display_factual_summary_ko(article: Article) -> str:
    return article.summary_ko or article.excerpt or article.title


def display_factual_summary_en(article: Article) -> str:
    return article.summary_en or article.excerpt or article.title


def display_summary_ko(article: Article) -> str:
    return display_factual_summary_ko(article)


def display_summary_en(article: Article) -> str:
    return display_factual_summary_en(article)


def display_why_it_matters_ko(article: Article) -> str:
    return article.why_it_matters_ko or "인공지능 산업 생태계에 유의미한 전략적 파급효과가 예상됩니다."


def display_why_it_matters_en(article: Article) -> str:
    return article.why_it_matters_en or "Has notable strategic and technical implications for the AI ecosystem."


def display_primary_category(article: Article, lang: str = "ko") -> str:
    cat = article.primary_category or article.category or "frontier_models"
    if lang == "en":
        return CATEGORY_LABELS_EN.get(cat, "AI Technology")
    return CATEGORY_LABELS_KO.get(cat, "인공지능 기술")


def display_published_time(article: Article) -> str:
    if not article.published_at:
        return ""
    return article.published_at.strftime("%Y-%m-%d %H:%M")


def display_url(article: Article) -> str:
    return article.canonical_url or article.url


def display_topics_str(article: Article) -> str:
    if article.topics:
        return ", ".join(article.topics[:3])
    for pt in article.key_points or []:
        p = str(pt).strip()
        if "핵심 분야" in p or "Topics" in p or "주요 분야" in p or "핵심 토픽" in p:
            if ":" in p:
                val = p.split(":", 1)[1].strip()
                if val:
                    return val
    if article.tags:
        return ", ".join(article.tags[:3])
    return display_primary_category(article, "ko")


def display_key_points(article: Article) -> list[str]:
    points = article.key_points or []
    cleaned: list[str] = []
    seen = set()
    for pt in points:
        p_str = str(pt).strip()
        if not p_str:
            continue
        p_lower = p_str.lower()
        # Filter out redundant metadata like source, publisher, or category/topics
        if (
            p_str.startswith("발행처")
            or p_str.startswith("출처")
            or ("source" in p_lower and ("출처" in p_str or "발행처" in p_str))
            or p_str.startswith("핵심 분야")
            or p_str.startswith("핵심 토픽")
            or p_str.startswith("주요 분야")
            or p_str.startswith("카테고리")
            or ("topics" in p_lower and ("분야" in p_str or "토픽" in p_str))
        ):
            continue
        if p_str not in seen:
            seen.add(p_str)
            cleaned.append(p_str)
    return cleaned


def visible_tags(article: Article) -> list[str]:
    return article.tags


def display_event_coverage(article: Article) -> dict[str, Any]:
    count = article.event_source_count or 1
    return {
        "source_count": count,
        "independent_source_count": article.event_independent_source_count or 1,
        "has_official_source": article.event_has_official_source,
        "has_regulatory_source": article.event_has_regulatory_source,
        "official_source_url": article.event_official_source_url,
        "official_source_name": article.event_official_source_name,
        "related_sources": article.event_related_sources or [article.source],
        "label_ko": f"{count}개 매체 보도 중",
        "label_en": f"{count} sources covering this event",
    }


def display_source_transparency(article: Article) -> SourceTransparency:
    raw_pub = (article.publisher or article.source or "").strip()
    ev_level = article.evidence_level or source_evidence_level(raw_pub) or "industry_media"
    badge = EVIDENCE_BADGES.get(ev_level, EVIDENCE_BADGES["industry_media"])

    # Collect source list
    related = list(article.event_related_sources or [])
    if not related:
        primary_pub = raw_pub or "Unknown Source"
        related = [primary_pub]

    is_multi = len(related) > 1
    sources_display = " · ".join(related[:4]) + (f" (+{len(related)-4})" if len(related) > 4 else "")

    # Resolve / calibrate verification status
    status = article.event_verification_status
    if not status or status == "insufficient_evidence":
        if is_multi:
            status = "multi_source"
        elif ev_level == "primary" or article.is_primary_source or article.is_official or article.source_type == "official":
            status = "primary_only"
        elif ev_level == "independent" or article.is_independent_source:
            status = "independently_reported"
        elif ev_level == "research" or article.source_type == "research":
            status = "primary_only"
        elif ev_level == "industry_media" or article.source_type in {"media", "korean_media"}:
            status = "independently_reported"
        elif ev_level == "discovery" or article.is_discovery_source:
            status = "discovery_only"
        else:
            status = "insufficient_evidence"

    # Resolve bilingual evidence labels
    if status == "multi_source":
        ev_label_en = "Multi-source independent reporting"
        ev_label_ko = "다수 독립 언론 교차 취재"
    elif status == "independently_reported":
        ev_label_en = "Primary + independent reporting"
        ev_label_ko = "1차 출처 + 독립 언론 취재"
    elif status == "primary_only":
        if article.is_primary_source or article.source_type == "official":
            ev_label_en = "Primary announcement"
            ev_label_ko = "1차 공식 발표"
        elif ev_level == "research" or article.source_type == "research":
            ev_label_en = "Research publication"
            ev_label_ko = "연구기관 발표 기반"
        else:
            ev_label_en = "Primary announcement"
            ev_label_ko = "1차 공식 발표"
    elif status == "discovery_only":
        ev_label_en = "Aggregator discovery"
        ev_label_ko = "검색 집계 발견"
    else:
        if ev_level == "primary":
            ev_label_en = "Primary source"
            ev_label_ko = "1차 출처"
        elif ev_level == "research":
            ev_label_en = "Research publication"
            ev_label_ko = "연구기관 발표"
        elif ev_level == "independent":
            ev_label_en = "Independent reporting"
            ev_label_ko = "독립 취재 보도"
        elif ev_level == "industry_media":
            ev_label_en = "Industry media"
            ev_label_ko = "산업 전문 미디어"
        elif ev_level == "community":
            ev_label_en = "Community discussion"
            ev_label_ko = "커뮤니티 논의"
        else:
            ev_label_en = "Aggregator / Discovery"
            ev_label_ko = "검색 집계 / 발견"

    # Calibrate confidence score & labels
    conf_label = article.event_confidence_label
    conf_exp_ko = article.event_confidence_explanation_ko or ""
    conf_exp_en = article.event_confidence_explanation_en or ""

    if not conf_label or (conf_label == "Low" and status in {"multi_source", "independently_reported", "primary_only"}):
        auth = article.authority_score or article.source_authority or 70
        is_official = article.is_official or article.is_primary_source or article.source_type == "official" or ev_level == "primary"
        is_ind = article.is_independent_source or ev_level == "independent"

        if status == "multi_source" or (is_official and is_ind):
            conf_label = "High"
            conf_exp_ko = f"{raw_pub} 등 다수 출처 교차 검증"
            conf_exp_en = f"Multi-source verified reporting including {raw_pub}"
        elif is_official:
            conf_label = "High" if auth >= 90 else "Medium"
            conf_exp_ko = f"{raw_pub} 공식 발표 (1차 출처)"
            conf_exp_en = f"{raw_pub} official primary announcement"
        elif is_ind:
            conf_label = "High" if auth >= 90 else "Medium"
            conf_exp_ko = f"{raw_pub} 독립 언론 심층 보도"
            conf_exp_en = f"{raw_pub} independent journalistic coverage"
        elif ev_level == "research" or article.source_type == "research":
            conf_label = "High" if auth >= 90 else "Medium"
            conf_exp_ko = f"{raw_pub} 연구기관 논문 및 발표"
            conf_exp_en = f"{raw_pub} academic / research publication"
        elif ev_level == "industry_media" or article.source_type in {"media", "korean_media"}:
            conf_label = "Medium"
            conf_exp_ko = f"{raw_pub} IT/산업 전문 매체 보도"
            conf_exp_en = f"{raw_pub} industry media coverage"
        else:
            conf_label = "Low"
            conf_exp_ko = "커뮤니티 논의 또는 미확인 출처"
            conf_exp_en = "Community discussion or unverified source"

    if conf_label == "High":
        conf_label_ko = "높음"
    elif conf_label == "Medium":
        conf_label_ko = "보통"
    else:
        conf_label_ko = "낮음"

    primary_sources = list(article.event_primary_sources or [])
    ind_sources = list(article.event_independent_sources or [])
    other_sources = [s for s in related if s.lower() not in [p.lower() for p in primary_sources + ind_sources]]

    return SourceTransparency(
        evidence_badge=badge,
        sources_display=sources_display,
        evidence_label_en=ev_label_en,
        evidence_label_ko=ev_label_ko,
        confidence_label=conf_label,
        confidence_label_ko=conf_label_ko,
        confidence_explanation_en=conf_exp_en,
        confidence_explanation_ko=conf_exp_ko,
        is_multi_source=is_multi,
        independent_source_count=article.event_independent_source_count or (1 if article.is_independent_source else 0),
        primary_sources=primary_sources,
        independent_sources=ind_sources,
        other_sources=other_sources,
    )


def prepare_article_view(article: Article) -> ArticleView:
    return ArticleView(
        article=article,
        priority=assess_priority(article),
        source_url=display_url(article),
        published_time=display_published_time(article),
        regions=regions_for_article(article),
        region_keys=[r.key for r in regions_for_article(article)],
        all_region_keys=all_region_keys_for_article(article),
        topic_keys=topic_keys_for_article(article),
        source_type=canonical_source_type(article),
        title_ko=display_title_ko(article),
        title_en=display_title_en(article),
        factual_summary_ko=display_factual_summary_ko(article),
        factual_summary_en=display_factual_summary_en(article),
        why_it_matters_ko=display_why_it_matters_ko(article),
        why_it_matters_en=display_why_it_matters_en(article),
        primary_category_ko=display_primary_category(article, "ko"),
        primary_category_en=display_primary_category(article, "en"),
        coverage=display_event_coverage(article),
        user_tags=visible_tags(article),
        publisher_display=article.publisher or article.source or "Unknown Source",
        transparency=display_source_transparency(article),
        key_points=display_key_points(article),
        topics_display=display_topics_str(article),
    )


def build_intelligence_sections(
    issue: NewsletterIssue,
    lang: str = "ko",
    min_score: float = 60.0,
    max_per_section: int = 6,
) -> list[dict[str, Any]]:
    """Build intelligence sections displaying only high-priority featured articles on the main visual cards."""
    is_en = lang == "en"
    raw_articles = issue.articles

    # Filter out low-scoring articles for featured cards (keep Critical & High priority, score >= 60.0)
    has_scores = any((a.priority_score or a.score or 0) > 0 for a in raw_articles)
    if has_scores:
        qualified = [a for a in raw_articles if (a.priority_score or a.score or 0) >= min_score]
        if len(qualified) < 3:
            qualified = [a for a in raw_articles if (a.priority_score or a.score or 0) >= 45.0]
        if len(qualified) < 3:
            qualified = [a for a in raw_articles if (a.priority_score or a.score or 0) >= 30.0]
        articles = qualified if qualified else raw_articles
    else:
        articles = raw_articles

    sections = []
    assigned_ids = set()

    for defn in INTELLIGENCE_SECTION_DEFINITIONS:
        sec_articles = [
            a for a in articles
            if (a.primary_category in defn["categories"] or a.category in defn["categories"])
            and a.article_id not in assigned_ids
        ]
        sec_articles.sort(key=lambda a: a.priority_score or a.score or 0.0, reverse=True)
        sec_articles = sec_articles[:max_per_section]
        for a in sec_articles:
            assigned_ids.add(a.article_id)

        sections.append({
            "key": defn["key"],
            "label_en": defn["label_en"],
            "label_ko": defn["label_ko"],
            "label": defn["label_en"] if is_en else defn["label_ko"],
            "subtitle_en": defn["subtitle_en"],
            "subtitle_ko": defn["subtitle_ko"],
            "subtitle": defn["subtitle_en"] if is_en else defn["subtitle_ko"],
            "articles": sec_articles,
            "article_views": [prepare_article_view(a) for a in sec_articles],
        })

    leftovers = [a for a in articles if a.article_id not in assigned_ids]
    if leftovers and sections:
        sections[0]["articles"].extend(leftovers[:max_per_section])
        sections[0]["articles"].sort(key=lambda a: a.priority_score or a.score or 0.0, reverse=True)
        sections[0]["articles"] = sections[0]["articles"][:max_per_section]
        sections[0]["article_views"] = [prepare_article_view(a) for a in sections[0]["articles"]]

    return sections


def compute_issue_metrics(articles_or_views: list[Any]) -> dict[str, int]:
    articles: list[Article] = []
    for item in articles_or_views:
        if isinstance(item, Article):
            articles.append(item)
        elif isinstance(item, ArticleView):
            articles.append(item.article)
        elif isinstance(item, dict) and "articles" in item:
            articles.extend(item["articles"])

    critical = 0
    high = 0
    for a in articles:
        p = assess_priority(a)
        if p.level == "critical":
            critical += 1
        elif p.level == "high":
            high += 1

    return {
        "total": len(articles),
        "critical": critical,
        "high": high,
    }

