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
    rel = compute_relevance_score(article)
    imp = compute_impact_score(article)
    nov = compute_novelty_score(article, past_articles=past_articles)
    rec = compute_recency_score(article, ref_time=ref_time, half_life_hours=half_life_hours)

    base_priority = (
        0.30 * rel
        + 0.25 * imp
        + 0.20 * src
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

    explanation = build_score_explanation(
        source_score=src,
        relevance_score=rel,
        impact_score=imp,
        novelty_score=nov,
        recency_score=rec,
        priority_score=priority,
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
    )


def build_score_explanation(
    source_score: float,
    relevance_score: float,
    impact_score: float,
    novelty_score: float,
    recency_score: float,
    priority_score: float,
    impact_authority_boost: bool = False,
    frontier_impact_boost: bool = False,
    low_signal_penalty: bool = False,
) -> ScoreExplanation:
    reasons_ko: list[str] = []
    reasons_en: list[str] = []

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

