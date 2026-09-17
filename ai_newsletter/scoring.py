from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .models import Article

CORE_AI_TOPICS = {
    "llm",
    "large language model",
    "foundation model",
    "reasoning",
    "chain of thought",
    "cot",
    "agent",
    "agents",
    "autonomous agent",
    "swe-bench",
    "coding assistant",
    "gpu",
    "blackwell",
    "b200",
    "h100",
    "hbm",
    "hbm3e",
    "cuda",
    "npu",
    "asic",
    "tpu",
    "lpu",
    "groq",
    "datacenter",
    "power grid",
    "open weights",
    "open source",
    "llama",
    "gemma",
    "mistral",
    "vllm",
    "ollama",
    "fine-tuning",
    "rlhf",
    "multimodal",
    "vision",
    "video gen",
    "sora",
    "runway",
    "voice ai",
    "ai safety",
    "alignment",
    "eu ai act",
    "copyright",
    "regulation",
    "policy",
    "enterprise ai",
    "robotics",
    "humanoid",
    "biotech",
    "funding",
    "m&a",
}

IMPACT_SIGNALS = {
    "model_release": re.compile(r"\b(release|launch|unveil|announce|weights released|open weights|new model|sota|state-of-the-art)\b", re.I),
    "investment": re.compile(r"\b(billion|million|valuation|funding|raises|capex|series [a-z]|ipo|investment)\b", re.I),
    "partnership": re.compile(r"\b(partnership|alliance|collaboration|exclusive deal|supply agreement|joint venture)\b", re.I),
    "hardware_production": re.compile(r"\b(tape-out|mass production|shipment|foundry|gigawatt|datacenter expansion|wafer)\b", re.I),
    "benchmark": re.compile(r"\b(swe-bench|mmlu|humaneval|math benchmark|outperform|beats|arena)\b", re.I),
    "regulation_legal": re.compile(r"\b(eu ai act|antitrust|investigation|lawsuit|copyright|infringement|compliance|ban|subpoena)\b", re.I),
    "security_vulnerability": re.compile(r"\b(jailbreak|prompt injection|vulnerability|exploit|cve|red team|data breach)\b", re.I),
    "acquisition": re.compile(r"\b(acquisition|acquire[sd]?|merger|buyout|talent raid)\b", re.I),
}

EVERGREEN_PATTERNS = re.compile(
    r"\b(what is|how to|top 10|buyers? guide|overview of|guide to|explained|history of|beginner)\b",
    re.I,
)
PROMO_PATTERNS = re.compile(
    r"\b(proud to announce|honored to receive|award-winning|pioneering the future|presents at)\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class ScoreExplanation:
    reasons_ko: list[str] = field(default_factory=list)
    reasons_en: list[str] = field(default_factory=list)
    summary_ko: str = ""
    summary_en: str = ""


@dataclass(slots=True)
class ScoringResult:
    source_score: float
    relevance_score: float
    impact_score: float
    novelty_score: float
    recency_score: float
    priority_score: float
    explanation: ScoreExplanation
    evidence_quality_score: float = 0.0

    @property
    def source_authority_score(self) -> float:
        return self.source_score


def compute_source_score(article: Article) -> float:
    source_type = (article.source_type or "media").lower()
    source_name = (article.publisher or article.source or "").lower()

    if source_type in {"regulator"} or any(s in source_name for s in ["eu ai office", "nist", "과기정통부"]):
        base = 98.0
    elif source_type in {"research"} or any(s in source_name for s in ["arxiv", "stanford", "mit", "bair"]):
        base = 95.0
    elif any(s in source_name for s in ["openai", "anthropic", "deepmind", "meta", "nvidia"]):
        base = 94.0
    elif any(s in source_name for s in ["reuters", "bloomberg", "mit technology review", "the information"]):
        base = 92.0
    elif any(s in source_name for s in ["techcrunch", "the verge", "venturebeat", "wired", "ars technica", "aitimes", "zdnet", "전자신문"]):
        base = 86.0
    elif source_type == "open_source" or "hugging face" in source_name:
        base = 88.0
    elif source_type == "aggregator" or "google" in source_name:
        base = 55.0
    else:
        base = 75.0

    if article.source_authority is not None and article.source_authority != 70:
        score = 0.5 * base + 0.5 * float(article.source_authority)
    else:
        score = base

    return round(max(10.0, min(100.0, score)), 1)


def compute_evidence_quality_score(article: Article) -> float:
    """
    Computes evidence quality score based on the source's epistemic role.
    Distinct from source authority (institution prominence) and article importance (impact).

    Evidence levels and typical scores:
    - Primary source (official corporate/lab launch, regulator rule): 96.0 - 98.0
    - Independent reporting (reputable journalism with direct reporting): 94.0
    - Research (academic lab / peer-reviewed: 94.0; preprint e.g. arXiv: 88.0)
    - Industry media (curated news / trade reporting): 78.0
    - Community discussion (forums, social tech hubs, blogs): 55.0
    - Discovery / Aggregator (Google News, search feeds): 40.0

    CRITICAL RULE:
    Discovery/aggregator mechanisms (e.g. Google News) MUST NOT receive a high
    evidence quality score even if the destination article has a high-authority publisher.
    """
    source_type = (article.source_type or "").lower()
    source_name = (article.source or "").lower()
    publisher = (article.publisher or "").lower()
    evidence_level = (article.evidence_level or "").lower()
    url = (article.url or "").lower()

    # 1. Discovery / Aggregator check:
    # If the source feed is a discovery mechanism (like Google News) or marked as discovery aggregator,
    # it must remain low (<= 42.0) regardless of the publisher or destination authority.
    if (
        article.is_discovery_source
        or evidence_level == "discovery"
        or source_type == "aggregator"
        or "google news" in source_name
        or "google news" in publisher
        or "news.google.com" in url
    ):
        return 40.0

    # 2. Primary source check:
    # Official announcements by creators, primary press releases, or regulatory bodies.
    if (
        article.is_primary_source
        or evidence_level == "primary"
        or source_type in {"official", "regulator", "press_release"}
    ):
        if source_type == "regulator" or any(s in publisher or s in source_name for s in ["eu ai office", "nist", "과기정통부"]):
            return 98.0
        return 96.0

    # 3. Independent journalism / investigative reporting:
    if (
        article.is_independent_source
        or evidence_level == "independent"
    ):
        return 94.0

    # 4. Research: preprints vs academic / institutional labs:
    if (
        evidence_level == "research"
        or source_type in {"research", "paper", "preprint"}
    ):
        full_text = f"{source_name} {publisher} {url}"
        if any(term in full_text for term in ["arxiv", "biorxiv", "medrxiv", "preprint"]):
            return 88.0
        return 94.0

    # 5. Community sources (HN, Reddit, forums, individual developer blogs):
    if (
        evidence_level == "community"
        or source_type in {"community", "forum", "blog"}
    ):
        return 55.0

    # 6. Industry media / curated news outlets:
    if (
        evidence_level == "industry_media"
        or source_type in {"media", "news"}
    ):
        return 78.0

    return 70.0


def compute_relevance_score(article: Article) -> float:
    category = (article.primary_category or article.category or "").lower()
    text = f"{article.title} {article.excerpt} {' '.join(article.topics)} {' '.join(article.tags)}".lower()

    if category in {"frontier_models", "agents_coding", "hardware_infra"}:
        base = 65.0
    elif category in {"open_source", "research_breakthroughs", "multimodal_media"}:
        base = 58.0
    elif category in {"regulation_policy", "enterprise_app", "industry_vc"}:
        base = 45.0
    else:
        base = 30.0

    matched = set()
    for topic in [*article.topics, *article.tags]:
        t_low = topic.lower()
        if t_low in CORE_AI_TOPICS:
            matched.add(t_low)

    for term in CORE_AI_TOPICS:
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text):
            matched.add(term)

    topic_bonus = min(len(matched) * 10.0, 40.0)
    score = base + topic_bonus

    return round(max(10.0, min(100.0, score)), 1)


def compute_impact_score(article: Article) -> float:
    category = (article.primary_category or article.category or "").lower()
    text = f"{article.title} {article.excerpt}".lower()

    if category in {"big", "frontier_models", "hardware_infra"}:
        base = 45.0
    elif category in {"agents_coding", "regulation_policy", "industry_vc"}:
        base = 38.0
    else:
        base = 28.0

    signals_found = 0
    for pattern in IMPACT_SIGNALS.values():
        if pattern.search(text):
            signals_found += 1

    signal_bonus = min(signals_found * 16.0, 52.0)
    score = base + signal_bonus

    return round(max(10.0, min(100.0, score)), 1)


def compute_novelty_score(
    article: Article,
    past_articles: list[Article] | None = None,
) -> float:
    score = 90.0
    text = f"{article.title} {article.excerpt}".lower()

    if EVERGREEN_PATTERNS.search(article.title.lower()):
        score -= 35.0
    elif EVERGREEN_PATTERNS.search(text):
        score -= 20.0

    if PROMO_PATTERNS.search(text) and not any(p.search(text) for p in IMPACT_SIGNALS.values()):
        score -= 20.0

    if past_articles:
        title_tokens = _simple_tokens(article.title)
        if title_tokens:
            for past in past_articles:
                if past.url == article.url or past.canonical_url == article.canonical_url:
                    continue
                past_tokens = _simple_tokens(past.title)
                if not past_tokens:
                    continue
                common = title_tokens & past_tokens
                if len(common) >= 4 and len(common) / max(len(title_tokens), len(past_tokens)) >= 0.75:
                    score -= 30.0
                    break

    return round(max(10.0, min(100.0, score)), 1)


def compute_recency_score(
    article: Article,
    ref_time: datetime | None = None,
    half_life_hours: float = 36.0,
) -> float:
    if not article.published_at:
        return 50.0

    ref = ref_time or datetime.now(timezone.utc)
    if article.published_at.tzinfo is None:
        pub = article.published_at.replace(tzinfo=timezone.utc)
    else:
        pub = article.published_at

    diff_seconds = max(0.0, (ref - pub).total_seconds())
    diff_hours = diff_seconds / 3600.0

    decay = math.pow(0.5, diff_hours / max(1.0, half_life_hours))
    score = 100.0 * decay

    return round(max(5.0, min(100.0, score)), 1)


def compute_multi_dimensional_scores(
    article: Article,
    past_articles: list[Article] | None = None,
    ref_time: datetime | None = None,
    half_life_hours: float = 36.0,
) -> ScoringResult:
    src = compute_source_score(article)
    evq = compute_evidence_quality_score(article)
    rel = compute_relevance_score(article)
    imp = compute_impact_score(article)
    nov = compute_novelty_score(article, past_articles=past_articles)
    rec = compute_recency_score(article, ref_time=ref_time, half_life_hours=half_life_hours)

    base_priority = (
        0.25 * rel
        + 0.25 * imp
        + 0.15 * src
        + 0.10 * evq
        + 0.15 * rec
        + 0.10 * nov
    )

    impact_authority_boost = 0.0
    frontier_impact_boost = 0.0
    low_signal_penalty = 0.0

    if imp >= 65.0 and src >= 85.0:
        impact_authority_boost = 20.0

    if rel >= 80.0 and imp >= 45.0:
        frontier_impact_boost = 15.0

    if rel < 40.0 and imp < 40.0:
        low_signal_penalty = 10.0

    synergy_boost = max(impact_authority_boost, frontier_impact_boost)
    priority = base_priority + synergy_boost - low_signal_penalty
    priority = round(max(10.0, min(100.0, priority)), 1)

    is_preprint = False
    evidence_lvl = article.evidence_level
    if (
        article.is_discovery_source
        or (article.source_type or "").lower() == "aggregator"
        or "google news" in (article.source or "").lower()
        or "google news" in (article.publisher or "").lower()
    ):
        evidence_lvl = "discovery"
    elif article.is_primary_source and evidence_lvl != "discovery":
        evidence_lvl = "primary"
    elif article.is_independent_source and evidence_lvl != "discovery":
        evidence_lvl = "independent"

    if evidence_lvl == "research" or (article.source_type or "").lower() in {"research", "paper", "preprint"}:
        full_text = f"{article.source or ''} {article.publisher or ''} {article.url or ''}".lower()
        if any(term in full_text for term in ["arxiv", "biorxiv", "medrxiv", "preprint"]):
            is_preprint = True

    explanation = build_score_explanation(
        source_score=src,
        relevance_score=rel,
        impact_score=imp,
        novelty_score=nov,
        recency_score=rec,
        priority_score=priority,
        evidence_quality_score=evq,
        evidence_level=evidence_lvl,
        is_preprint=is_preprint,
        impact_authority_boost=impact_authority_boost > 0,
        frontier_impact_boost=frontier_impact_boost > 0,
        low_signal_penalty=low_signal_penalty > 0,
    )

    return ScoringResult(
        source_score=src,
        relevance_score=rel,
        impact_score=imp,
        novelty_score=nov,
        recency_score=rec,
        priority_score=priority,
        explanation=explanation,
        evidence_quality_score=evq,
    )


def build_score_explanation(
    source_score: float,
    relevance_score: float,
    impact_score: float,
    novelty_score: float,
    recency_score: float,
    priority_score: float,
    evidence_quality_score: float = 0.0,
    evidence_level: str = "",
    is_preprint: bool = False,
    impact_authority_boost: bool = False,
    frontier_impact_boost: bool = False,
    low_signal_penalty: bool = False,
) -> ScoreExplanation:
    reasons_ko: list[str] = []
    reasons_en: list[str] = []

    # Evidence type explanation (factual, neutral)
    if evidence_level == "discovery":
        reasons_ko.append("검색 집계 출처")
        reasons_en.append("Discovery source")
    elif evidence_level == "primary":
        reasons_ko.append("공식 발표 출처")
        reasons_en.append("Primary source")
    elif evidence_level == "independent":
        reasons_ko.append("독립 취재 출처")
        reasons_en.append("Independent reporting")
    elif evidence_level == "research":
        if is_preprint:
            reasons_ko.append("연구기관 사전 논문 (Preprint) 출처")
            reasons_en.append("Research preprint source")
        else:
            reasons_ko.append("공인 학술·연구기관 출처")
            reasons_en.append("Research source")
    elif evidence_level == "industry_media":
        reasons_ko.append("산업 전문 보도 출처")
        reasons_en.append("Industry media coverage")
    elif evidence_level == "community":
        reasons_ko.append("커뮤니티 논의 출처")
        reasons_en.append("Community discussion source")

    if impact_authority_boost:
        reasons_ko.append("공식 기관·선도 기업의 주요 발표")
        reasons_en.append("Authoritative major announcement")

    if frontier_impact_boost:
        reasons_ko.append("프론티어 모델·AI 아키텍처 핵심 혁신")
        reasons_en.append("Frontier model / AI architecture breakthrough")

    if relevance_score >= 80.0:
        reasons_ko.append("AI 핵심 기술·인프라 직결")
        reasons_en.append("Core AI technology & infrastructure relevance")

    if impact_score >= 70.0:
        reasons_ko.append("생태계 파급력 높은 대형 이벤트")
        reasons_en.append("High ecosystem market impact")

    if source_score >= 88.0:
        reasons_ko.append("공인 연구소 및 1차 취재 매체")
        reasons_en.append("High authority lab / primary news outlet")

    if recency_score >= 70.0:
        reasons_ko.append("최근 24시간 이내 신속 보도")
        reasons_en.append("Recent publication within 24h")

    if low_signal_penalty:
        reasons_ko.append("관찰 항목 (낮은 관련도·영향도)")
        reasons_en.append("Watch item (low relevance & impact)")

    if not reasons_ko:
        reasons_ko.append("일반 AI 산업 참고 항목")
        reasons_en.append("General AI ecosystem reference")

    summary_ko = " · ".join(reasons_ko[:3])
    summary_en = " · ".join(reasons_en[:3])

    return ScoreExplanation(
        reasons_ko=reasons_ko,
        reasons_en=reasons_en,
        summary_ko=summary_ko,
        summary_en=summary_en,
    )


def _simple_tokens(text: str) -> set[str]:
    cleaned = re.sub(r"[^a-z0-9가-힣\s]", " ", text.lower())
    stop_words = {
        "the", "and", "for", "with", "from", "that", "this", "about",
        "will", "are", "were", "was", "has", "have", "its", "new",
        "in", "on", "at", "by", "an", "a", "to", "of", "is", "it",
    }
    return {w for w in cleaned.split() if len(w) > 2 and w not in stop_words}

