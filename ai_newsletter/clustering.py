from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from .entity_registry import (
    find_entities_in_text,
    is_same_entity,
    is_same_parent_group,
    resolve_canonical_entity,
)
from .models import Article, Event, EventArticle

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "속보", "단독", "종합", "포토", "영상", "ai", "인공지능",
}

AI_MODEL_PATTERNS = [
    re.compile(r"\b(gpt-?4o?|gpt-?5|gpt-?3\.5|o1-preview|o1-mini|o1|o3)\b", re.I),
    re.compile(r"\b(claude\s*3(?:\.5)?(?:\s*(?:sonnet|opus|haiku))?|클로드\s*3(?:\.5)?)\b", re.I),
    re.compile(r"\b(gemini\s*(?:1\.5|2\.0)?(?:\s*(?:flash|pro|ultra))?|제미나이(?:\s*(?:1\.5|2\.0))?)\b", re.I),
    re.compile(r"\b(llama\s*(?:2|3|3\.1|3\.2|3\.3)?|라마\s*(?:2|3|3\.1|3\.2|3\.3)?)\b", re.I),
    re.compile(r"\b(blackwell|b200|b100|h100|h200|gb200|블랙웰)\b", re.I),
    re.compile(r"\b(mi300[xa]?|mi325x|mi350)\b", re.I),
    re.compile(r"\b(sora|gen-3|kling|flux|midjourney\s*v[0-9]+)\b", re.I),
    re.compile(r"\b(deepseek\s*(?:v2|v3|r1)?|qwen\s*(?:2|2\.5)?|gemma\s*(?:2)?|alphafold\s*(?:2|3)?|알파폴드\s*(?:2|3)?)\b", re.I),
]

AI_EVENT_THEMES = {
    "model_release": ["release", "releases", "released", "launch", "launches", "launched", "unveil", "unveils", "unveiled", "debut", "debuts", "debuted", "introduce", "introduces", "introduced", "open weights", "rollout", "announces", "announced", "announce", "출시", "공개", "발표", "선보여", "오픈소스"],
    "pricing": ["pricing", "price", "prices", "subscription", "cost", "tier", "tiers", "rates", "가격", "요금", "구독", "비용", "단가"],
    "benchmarking": ["swe-bench", "mmlu", "benchmark", "benchmarks", "outperforms", "beats", "sota", "evaluation", "evals", "벤치마크", "성능", "평가"],
    "investment": ["funding", "valuation", "billion", "raises funding", "series", "ipo", "invest", "investment", "투자", "유치", "펀딩"],
    "partnership": ["partnership", "partners", "collaboration", "collaborates", "alliance", "joint", "agreement", "pact", "initiative", "team up", "teams up", "pacts", "제휴", "협력", "파트너십", "동맹", "협약", "손잡고", "협업", "이니셔티브"],
    "legal_policy": ["lawsuit", "sues", "sued", "antitrust", "copyright", "ban", "compliance", "eu ai act", "regulation", "regulatory", "court", "probe", "investigation", "investigates", "fine", "fined", "소송", "고소", "제소", "규제", "위반", "법안", "조사", "과징금"],
    "security_safety": ["jailbreak", "vulnerability", "prompt injection", "exploit", "red team", "safety", "aisi", "alignment", "guardrail", "risk", "risks", "breach", "보안", "취약점", "안전", "안전성", "위험"],
    "datacenter_hardware": ["datacenter", "tape-out", "foundry", "hbm", "hbm3e", "gigawatt", "반도체", "양산", "데이터센터"],
    "office_expansion": ["office", "offices", "headquarters", "hq", "사옥", "지사", "신사옥"],
    "executive_movement": ["hire", "hires", "hired", "executive", "executives", "resign", "resigns", "resignation", "appointment", "appoints", "appointed", "영입", "사임", "임명", "선임"],
}

AI_ACTION_ANCHORS = {
    "release": ["release", "releases", "released", "launch", "launches", "launched", "unveil", "unveils", "unveiled", "debut", "debuts", "introduce", "introduces", "introduced", "rollout", "announces", "announced", "announce", "출시", "공개", "발표", "선보여"],
    "pricing": ["pricing", "price", "prices", "subscription", "cost", "tier", "tiers", "rates", "가격", "요금", "구독", "비용", "단가"],
    "partnership": ["partner", "partners", "partnership", "collaborate", "collaborates", "collaboration", "alliance", "joint", "agreement", "pact", "initiative", "team up", "teams up", "제휴", "협력", "파트너십", "동맹", "협약", "손잡고", "협업", "이니셔티브"],
    "legal": ["lawsuit", "sue", "sues", "sued", "antitrust", "copyright", "ban", "compliance", "regulation", "regulatory", "court", "probe", "investigation", "investigates", "fine", "fined", "소송", "고소", "제소", "규제", "위반", "법안", "조사", "과징금"],
    "security": ["safety", "aisi", "alignment", "jailbreak", "vulnerability", "exploit", "red team", "guardrail", "safeguard", "breach", "안전", "안전성", "보안", "취약점", "위험"],
    "investment": ["funding", "valuation", "billion", "raises funding", "series", "ipo", "invest", "investment", "투자", "유치", "펀딩"],
    "office_expansion": ["office", "offices", "headquarters", "hq", "사옥", "지사", "신사옥"],
    "executive_movement": ["hire", "hires", "hired", "executive", "executives", "resign", "resigns", "resignation", "appointment", "appoints", "appointed", "영입", "사임", "임명", "선임"],
}


def _match_keyword_in_text(keyword: str, text: str) -> bool:
    if re.search(r"^[a-z0-9\s_-]+$", keyword):
        return bool(re.search(r"\b" + re.escape(keyword) + r"\b", text, re.I))
    return keyword in text


@dataclass(frozen=True, slots=True)
class EventAnchor:
    """Structured semantic anchor of an event or article."""
    entities: frozenset[str]
    models: frozenset[str]
    themes: frozenset[str]
    actions: frozenset[str]


ACTION_CONFLICTS: dict[frozenset[str], float] = {
    frozenset({"release", "office_expansion"}): 0.40,
    frozenset({"release", "executive_movement"}): 0.40,
    frozenset({"release", "legal"}): 0.45,
    frozenset({"release", "security"}): 0.40,
    frozenset({"partnership", "legal"}): 0.45,
    frozenset({"pricing", "office_expansion"}): 0.50,
    frozenset({"pricing", "executive_movement"}): 0.50,
    frozenset({"office_expansion", "executive_movement"}): 0.40,
    frozenset({"investment", "legal"}): 0.40,
}


@dataclass(slots=True)
class ArticleClusteringFeatures:
    text: str
    title_text: str
    models: set[str]
    entities: set[str]
    tokens: set[str]
    themes: set[str]
    actions: set[str]
    anchors: set[tuple[str, str | None, str | None]]


def extract_event_anchors(
    entities: set[str],
    models: set[str],
    actions: set[str],
    themes: set[str],
) -> set[tuple[str, str | None, str | None]]:
    """Extract structured (action, entity, model) anchors from extracted features."""
    anchors = set()
    all_actions = actions | {
        "office_expansion" if "office_expansion" in themes else None,
        "executive_movement" if "executive_movement" in themes else None,
        "legal" if "legal_policy" in themes else None,
        "security" if "security_safety" in themes else None,
        "investment" if "investment" in themes else None,
        "pricing" if "pricing" in themes else None,
        "release" if "model_release" in themes else None,
    }
    cleaned_actions = {a for a in all_actions if a is not None}

    if not cleaned_actions:
        cleaned_actions = {"general"}

    target_entities = entities if entities else {None}
    target_models = models if models else {None}

    for act in cleaned_actions:
        for ent in target_entities:
            for mod in target_models:
                anchors.add((act, ent, mod))

    return anchors


def extract_clustering_features(article: Article) -> ArticleClusteringFeatures:
    title_parts = [article.title or "", article.title_ko or "", article.title_en or ""]
    full_title_text = " ".join(p for p in title_parts if p).strip()

    body_parts = [
        article.excerpt or "",
        article.summary_ko or "",
        article.summary_en or "",
        " ".join(article.tags or []),
        " ".join(article.key_points or []),
    ]
    full_text = f"{full_title_text} {' '.join(p for p in body_parts if p)}".lower()

    # Model identifiers
    models = set()
    for pat in AI_MODEL_PATTERNS:
        for match in pat.finditer(full_text):
            models.add(match.group(0).lower().replace(" ", "").replace("-", ""))

    # Entities
    found_entities = set(find_entities_in_text(full_text))

    # Clean tokens from title text + key metadata (supports English and Korean)
    token_seed_text = f"{full_title_text} {' '.join(article.tags or [])} {article.excerpt or ''}".lower()
    cleaned = re.sub(r"[^a-z0-9가-힣\s]", " ", token_seed_text)
    tokens = {w for w in cleaned.split() if len(w) >= 2 and w not in STOP_WORDS}

    # Themes
    themes = set()
    for theme_name, keywords in AI_EVENT_THEMES.items():
        if any(_match_keyword_in_text(kw, full_text) for kw in keywords):
            themes.add(theme_name)

    # Actions
    actions = set()
    for action_name, keywords in AI_ACTION_ANCHORS.items():
        if any(_match_keyword_in_text(kw, full_text) for kw in keywords):
            actions.add(action_name)

    anchors = extract_event_anchors(found_entities, models, actions, themes)

    return ArticleClusteringFeatures(
        text=full_text,
        title_text=full_title_text.lower(),
        models=models,
        entities=found_entities,
        tokens=tokens,
        themes=themes,
        actions=actions,
        anchors=anchors,
    )


def calculate_event_similarity(
    a1: Article,
    a2: Article,
    window_hours: float = 72.0,
    f1: ArticleClusteringFeatures | None = None,
    f2: ArticleClusteringFeatures | None = None,
) -> float:
    if a1.article_id and a1.article_id == a2.article_id:
        return 1.0
    if a1.canonical_url and a1.canonical_url == a2.canonical_url:
        return 1.0
    if a1.url and a1.url == a2.url:
        return 1.0

    # Temporal window
    if a1.published_at and a2.published_at:
        diff_hours = abs((a1.published_at - a2.published_at).total_seconds()) / 3600.0
        if diff_hours > window_hours:
            return 0.0

    if f1 is None:
        f1 = extract_clustering_features(a1)
    if f2 is None:
        f2 = extract_clustering_features(a2)

    # Token overlap (Jaccard similarity)
    if not f1.tokens or not f2.tokens:
        return 0.0

    intersect = len(f1.tokens & f2.tokens)
    union = len(f1.tokens | f2.tokens)
    jaccard = intersect / union if union > 0 else 0.0

    # Layered Hard-Negative Guards
    # 1. Competitor entity collision (pure competitor stories without alliance or shared framework)
    if f1.entities and f2.entities and f1.entities.isdisjoint(f2.entities):
        shared_joint_context = (
            bool((f1.themes & f2.themes) & {"partnership", "legal_policy", "security_safety"})
            or bool((f1.actions & f2.actions) & {"partnership", "legal", "security"})
        )
        if not (shared_joint_context and jaccard >= 0.15):
            return 0.0

    # 2. Model version collision
    # Strictly block disjoint models UNLESS there is strong multi-entity alliance / joint safety / benchmark context
    if f1.models and f2.models and f1.models.isdisjoint(f2.models):
        has_multi_entity_joint_anchor = len(f1.entities & f2.entities) >= 2
        has_joint_event_anchor = bool(
            (f1.themes & f2.themes) & {"partnership", "security_safety", "legal_policy", "benchmarking"}
            or (f1.actions & f2.actions) & {"partnership", "security", "legal"}
        )
        if not (has_multi_entity_joint_anchor or has_joint_event_anchor):
            return 0.0

    # 3. Action & Theme hard clash guards
    # Strict separation between pricing updates and non-pricing stories
    if ("pricing" in f1.actions) ^ ("pricing" in f2.actions):
        return 0.0

    # Strict separation between pure office expansion and non-office stories (unless shared release/partnership theme)
    if (("office_expansion" in f1.themes) or ("office_expansion" in f1.actions)) ^ (
        ("office_expansion" in f2.themes) or ("office_expansion" in f2.actions)
    ):
        if not (f1.themes & f2.themes & {"partnership", "investment", "model_release"}):
            return 0.0

    # Strict separation between pure executive movement and non-executive stories (unless shared release/partnership theme)
    if (("executive_movement" in f1.themes) or ("executive_movement" in f1.actions)) ^ (
        ("executive_movement" in f2.themes) or ("executive_movement" in f2.actions)
    ):
        if not (f1.themes & f2.themes & {"partnership", "investment", "model_release"}):
            return 0.0

    # 4. Action & Theme clash penalties
    clash_penalty = 0.0
    # Legal / investigation vs pure release / product (unless shared legal/policy theme)
    if (("legal" in f1.actions) ^ ("legal" in f2.actions)) and (("release" in f1.actions) ^ ("release" in f2.actions)):
        if not ((f1.themes & f2.themes) & {"legal_policy", "partnership"}):
            clash_penalty += 0.45
    # Office expansion vs pure release / legal / security (when not sharing joint theme)
    if (("office_expansion" in f1.actions) ^ ("office_expansion" in f2.actions)) and (("release" in f1.actions) ^ ("release" in f2.actions)):
        if not (f1.themes & f2.themes & {"model_release", "partnership"}):
            clash_penalty += 0.40
    # Executive movement vs pure release / legal / security
    if (("executive_movement" in f1.actions) ^ ("executive_movement" in f2.actions)) and (("release" in f1.actions) ^ ("release" in f2.actions)):
        if not (f1.themes & f2.themes & {"model_release", "partnership"}):
            clash_penalty += 0.40
    # Security vulnerability vs pure release
    if (("security" in f1.actions) ^ ("security" in f2.actions)) and (("release" in f1.actions) ^ ("release" in f2.actions)):
        if not ((f1.themes & f2.themes) & {"security_safety", "partnership"}):
            clash_penalty += 0.40
    # Partnership vs pure release (without shared partnership or multi-entity context)
    if (("partnership" in f1.actions) ^ ("partnership" in f2.actions)) and (("release" in f1.actions) ^ ("release" in f2.actions)):
        if not ((f1.themes & f2.themes) & {"partnership", "security_safety"} or len(f1.entities & f2.entities) >= 2):
            clash_penalty += 0.40

    # Layered Score Aggregation
    score = jaccard

    # Entities bonus
    shared_entities = f1.entities & f2.entities
    if len(shared_entities) >= 2:
        score += 0.30
    elif len(shared_entities) == 1:
        has_shared_action_or_theme = bool(
            (f1.actions & f2.actions)
            or (f1.themes & f2.themes)
            or (f1.models & f2.models)
            or ((f1.themes | f2.themes) <= {"model_release", "benchmarking"} and (f1.themes or f2.themes))
            or jaccard >= 0.20
        )
        if has_shared_action_or_theme:
            score += 0.25
        else:
            # Merely mentioning the same company with completely unrelated actions/themes
            score += 0.05

    # Models bonus
    if f1.models and f2.models and not f1.models.isdisjoint(f2.models):
        score += 0.25

    # Themes bonus
    if f1.themes and f2.themes and not f1.themes.isdisjoint(f2.themes):
        score += 0.15
    elif (f1.themes | f2.themes) <= {"model_release", "benchmarking"} and (f1.themes and f2.themes):
        # Benchmarking coverage of a new model release
        score += 0.10

    # Actions bonus
    if f1.actions and f2.actions and not f1.actions.isdisjoint(f2.actions):
        score += 0.15

    score -= clash_penalty

    return min(1.0, max(0.0, score))


# ── Verification status constants ─────────────────────────────────────────────

VERIFICATION_STATUS_PRIMARY_ONLY = "primary_only"
VERIFICATION_STATUS_INDEPENDENTLY_REPORTED = "independently_reported"
VERIFICATION_STATUS_MULTI_SOURCE = "multi_source"
VERIFICATION_STATUS_DISCOVERY_ONLY = "discovery_only"
VERIFICATION_STATUS_INSUFFICIENT = "insufficient_evidence"

VALID_VERIFICATION_STATUSES = frozenset({
    VERIFICATION_STATUS_PRIMARY_ONLY,
    VERIFICATION_STATUS_INDEPENDENTLY_REPORTED,
    VERIFICATION_STATUS_MULTI_SOURCE,
    VERIFICATION_STATUS_DISCOVERY_ONLY,
    VERIFICATION_STATUS_INSUFFICIENT,
})


def _normalize_publisher(publisher: str) -> str:
    """Canonical form of a publisher name for deduplication.

    Strips trailing punctuation, lowercases, and removes common suffixes
    so that 'Reuters', 'Reuters Technology', and 'REUTERS' all map to
    the same canonical key.
    """
    name = publisher.strip().lower()
    # Remove trailing domain components that appear when publisher is inferred
    # from a Google News result, e.g. "reuters.com"
    name = re.sub(r"\.com$|\.net$|\.org$|\.io$", "", name)
    # Remove common trailing words that don't change the publisher identity
    for suffix in (" technology", " tech", " news", " media", " newsroom",
                   " official", " blog", " online", " digital", " ai",
                   " business", " finance", " magazine", " report"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.strip()


@dataclass(slots=True)
class EventEvidence:
    """Computed evidence metadata for a single event cluster."""
    # All distinct canonical publisher names in the cluster
    evidence_sources: list[str]
    # Canonical publisher names with evidence_level == "primary"
    primary_sources: list[str]
    # Canonical publisher names with evidence_level == "independent"
    independent_sources: list[str]
    # len(evidence_sources) — 0 means discovery/aggregator only
    evidence_diversity: int
    # Derived verification status
    verification_status: str


def compute_event_evidence(articles: list[Article]) -> EventEvidence:
    """Compute deduplicated evidence metadata for a cluster of articles.

    Key invariant: multiple articles from the same publisher (e.g. two Reuters
    follow-up pieces, or an OpenAI post + repost) count as ONE evidence source.

    Verification logic
    ------------------
    discovery_only:          all articles are is_discovery_source or evidence_level=="discovery"
    insufficient_evidence:   all articles are community or unknown, no real publishers
    primary_only:            ≥1 distinct primary publisher, 0 independent publishers
    independently_reported:  ≥1 primary + ≥1 independent publisher
    multi_source:            ≥2 independent publishers (with or without primary)
    """
    # Bucket articles by canonical publisher name and evidence level.
    # We use a dict: canonical_publisher -> set of evidence_levels seen from that publisher.
    publisher_evidence: dict[str, set[str]] = {}

    for a in articles:
        raw_pub = (a.publisher or a.source or "").strip()
        if not raw_pub:
            continue
        canonical = _normalize_publisher(raw_pub)
        if not canonical:
            continue
        publisher_evidence.setdefault(canonical, set()).add(
            a.evidence_level or "industry_media"
        )

    # Separate into evidence-type buckets (per publisher, not per article)
    primary_pubs: list[str] = []
    independent_pubs: list[str] = []
    research_pubs: list[str] = []
    discovery_pubs: list[str] = []
    community_pubs: list[str] = []
    other_pubs: list[str] = []

    for pub, levels in publisher_evidence.items():
        # A publisher is classified by its *strongest* evidence level.
        if "primary" in levels:
            primary_pubs.append(pub)
        elif "independent" in levels:
            independent_pubs.append(pub)
        elif "research" in levels:
            research_pubs.append(pub)
        elif "discovery" in levels:
            discovery_pubs.append(pub)
        elif "community" in levels:
            community_pubs.append(pub)
        else:
            other_pubs.append(pub)

    primary_pubs.sort()
    independent_pubs.sort()
    research_pubs.sort()

    # All publishers except pure-discovery and community
    substantive_pubs = sorted(set(primary_pubs) | set(independent_pubs) | set(research_pubs) | set(other_pubs))
    all_pubs = sorted(publisher_evidence.keys())

    # Determine verification status
    n_primary = len(primary_pubs)
    n_independent = len(independent_pubs)
    n_research = len(research_pubs)
    n_substantive = len(substantive_pubs)
    has_only_discovery = bool(discovery_pubs) and n_substantive == 0 and not community_pubs
    has_only_community = bool(community_pubs) and n_substantive == 0 and not discovery_pubs
    has_nothing_substantial = n_substantive == 0

    if has_only_discovery:
        status = VERIFICATION_STATUS_DISCOVERY_ONLY
    elif has_nothing_substantial:
        status = VERIFICATION_STATUS_INSUFFICIENT
    elif n_independent >= 2:
        # Two or more *different* independent publishers = multi_source
        status = VERIFICATION_STATUS_MULTI_SOURCE
    elif n_primary >= 1 and n_independent >= 1:
        status = VERIFICATION_STATUS_INDEPENDENTLY_REPORTED
    elif n_primary >= 1:
        status = VERIFICATION_STATUS_PRIMARY_ONLY
    elif n_independent == 1:
        # Single independent, no primary — treat as independently_reported
        status = VERIFICATION_STATUS_INDEPENDENTLY_REPORTED
    elif n_research >= 1:
        # Research publication without independent reporting
        status = VERIFICATION_STATUS_PRIMARY_ONLY
    else:
        status = VERIFICATION_STATUS_INSUFFICIENT

    evidence_diversity = len(substantive_pubs)

    return EventEvidence(
        evidence_sources=all_pubs,
        primary_sources=primary_pubs,
        independent_sources=independent_pubs,
        evidence_diversity=evidence_diversity,
        verification_status=status,
    )


# ── Evidence confidence ───────────────────────────────────────────────────────

CONFIDENCE_LABEL_HIGH = "High"
CONFIDENCE_LABEL_MEDIUM = "Medium"
CONFIDENCE_LABEL_LOW = "Low"

CONFIDENCE_LABEL_HIGH_KO = "높음"
CONFIDENCE_LABEL_MEDIUM_KO = "보통"
CONFIDENCE_LABEL_LOW_KO = "낮음"

# Score thresholds
_THRESHOLD_HIGH = 70.0
_THRESHOLD_MEDIUM = 40.0

# Display caps on publisher names in explanations (keep it short)
_MAX_PUB_NAMES = 3


@dataclass(slots=True)
class ConfidenceResult:
    """Evidence confidence for an event cluster.

    confidence_score:   0–100 composite score.
    confidence_label:   "High" | "Medium" | "Low"
    label_ko:           "높음" | "보통" | "낮음"
    explanation_en:     Short English explanation, e.g.:
                        "OpenAI primary announcement + independent Reuters reporting"
    explanation_ko:     Short Korean explanation, e.g.:
                        "OpenAI 공식 발표 + Reuters 독립 취재"
    """
    confidence_score: float
    confidence_label: str
    label_ko: str
    explanation_en: str
    explanation_ko: str


def _pretty_pub(name: str) -> str:
    """Title-case a canonical (lowercase) publisher name for display."""
    return " ".join(w.capitalize() for w in name.split())


def _build_explanation(
    primary_sources: list[str],
    independent_sources: list[str],
    verification_status: str,
    has_research: bool,
    has_regulatory: bool,
) -> tuple[str, str]:
    """Build short bilingual explanations (English, Korean).

    Rules:
    - Never say "verified" or "확인됨" as a factual claim
    - Neutral, factual phrasing only
    - Cap at _MAX_PUB_NAMES names each side to keep it short
    """
    en_parts: list[str] = []
    ko_parts: list[str] = []

    if verification_status == VERIFICATION_STATUS_DISCOVERY_ONLY:
        return (
            "Only discovered through an aggregator",
            "검색 집계 결과만 확인됨",
        )
    if verification_status == VERIFICATION_STATUS_INSUFFICIENT:
        return (
            "Insufficient evidence — community discussion or no known publisher",
            "충분한 출처 없음 — 커뮤니티 논의 또는 알 수 없는 출처",
        )

    if has_regulatory:
        en_parts.append("regulatory source")
        ko_parts.append("규제기관 발표")

    if primary_sources:
        display = [_pretty_pub(p) for p in primary_sources[:_MAX_PUB_NAMES]]
        more = len(primary_sources) - _MAX_PUB_NAMES
        names_en = ", ".join(display) + (f" (+{more} more)" if more > 0 else "")
        names_ko = ", ".join(display) + (f" 외 {more}곳" if more > 0 else "")
        en_parts.append(f"{names_en} primary announcement")
        ko_parts.append(f"{names_ko} 공식 발표")

    if has_research and not primary_sources:
        en_parts.append("research publication")
        ko_parts.append("연구기관 발표 기반")

    if independent_sources:
        display = [_pretty_pub(p) for p in independent_sources[:_MAX_PUB_NAMES]]
        more = len(independent_sources) - _MAX_PUB_NAMES
        names_en = ", ".join(display) + (f" (+{more} more)" if more > 0 else "")
        names_ko = ", ".join(display) + (f" 외 {more}곳" if more > 0 else "")
        en_parts.append(f"independent {names_en} reporting")
        ko_parts.append(f"{names_ko} 독립 취재")

    if not en_parts:
        # Industry media or other fallback
        return (
            "Industry media coverage only",
            "산업 전문 매체 보도만 확인됨",
        )

    return " + ".join(en_parts), " + ".join(ko_parts)


def compute_event_confidence(
    evidence: EventEvidence,
    articles: list[Article],
) -> ConfidenceResult:
    """Compute evidence confidence for an event cluster.

    Score is 0–100.  It describes the strength and diversity of *available*
    evidence, NOT a claim that the underlying fact is objectively true.

    Scoring approach
    ----------------
    Base score from verification_status:
      multi_source            85
      independently_reported  70
      primary_only            58
      (fallback/other)        40
      discovery_only          22
      insufficient_evidence   10

    Modifiers (all additive/subtractive, capped at 0–100):
      +12  regulatory source present
      +8   research evidence present (evidence_level==research)
      +6   each additional independent publisher beyond 1 (max +18)
      +5   primary source authority score ≥ 95 (frontier lab)
      +4   primary source authority score ≥ 90
      +3   at least one article has non-empty content
      -8   all articles are discovery-source only
      -5   evidence_diversity == 1 and no independent source
           (single industry_media outlet, no primary or independent)
    """
    status = evidence.verification_status

    # Base score
    if status == VERIFICATION_STATUS_MULTI_SOURCE:
        score = 85.0
    elif status == VERIFICATION_STATUS_INDEPENDENTLY_REPORTED:
        score = 70.0
    elif status == VERIFICATION_STATUS_PRIMARY_ONLY:
        score = 58.0
    elif status == VERIFICATION_STATUS_DISCOVERY_ONLY:
        score = 22.0
    else:  # insufficient_evidence
        score = 10.0

    has_research = any(
        (a.evidence_level or "") == "research"
        or (a.source_type or "") in {"research", "paper", "preprint"}
        for a in articles
    )
    has_regulatory = any(
        (a.source_type or "") == "regulator"
        for a in articles
    )
    has_content = any(bool((a.content or "").strip()) for a in articles)

    # Modifier: regulatory source
    if has_regulatory:
        score += 12.0

    # Modifier: research evidence
    if has_research:
        score += 8.0

    # Modifier: extra independent publishers (each beyond the first)
    extra_ind = max(0, len(evidence.independent_sources) - 1)
    score += min(18.0, extra_ind * 6.0)

    # Modifier: primary source authority
    max_authority = 0
    for a in articles:
        if a.is_primary_source or (a.source_type or "") in {"official", "regulator"}:
            auth = a.authority_score or a.source_authority or 0
            if auth and auth > max_authority:
                max_authority = auth
    if max_authority >= 95:
        score += 5.0
    elif max_authority >= 90:
        score += 4.0

    # Modifier: article content present
    if has_content:
        score += 3.0

    # Penalty: discovery only
    if status == VERIFICATION_STATUS_DISCOVERY_ONLY:
        score -= 8.0

    # Penalty: single non-primary/non-independent publisher
    if (
        evidence.evidence_diversity == 1
        and len(evidence.primary_sources) == 0
        and len(evidence.independent_sources) == 0
    ):
        score -= 5.0

    score = round(max(0.0, min(100.0, score)), 1)

    # Label
    if score >= _THRESHOLD_HIGH:
        label = CONFIDENCE_LABEL_HIGH
        label_ko = CONFIDENCE_LABEL_HIGH_KO
    elif score >= _THRESHOLD_MEDIUM:
        label = CONFIDENCE_LABEL_MEDIUM
        label_ko = CONFIDENCE_LABEL_MEDIUM_KO
    else:
        label = CONFIDENCE_LABEL_LOW
        label_ko = CONFIDENCE_LABEL_LOW_KO

    explanation_en, explanation_ko = _build_explanation(
        primary_sources=evidence.primary_sources,
        independent_sources=evidence.independent_sources,
        verification_status=status,
        has_research=has_research,
        has_regulatory=has_regulatory,
    )

    return ConfidenceResult(
        confidence_score=score,
        confidence_label=label,
        label_ko=label_ko,
        explanation_en=explanation_en,
        explanation_ko=explanation_ko,
    )


def select_cluster_representative(
    cluster: list[int],
    articles: list[Article],
    features: list[ArticleClusteringFeatures],
    window_hours: float = 72.0,
) -> int:
    """Select the most authoritative and central representative article for a cluster.

    Deterministic ranking:
    1. Primary/Official status (1 if official/primary else 0)
    2. Evidence level strength (primary: 3, independent/research: 2, industry_media: 1, other: 0)
    3. Authority score (authority_score or source_authority or 0.0)
    4. Priority score (priority_score or score or 0.0)
    5. Anchor richness (number of models + entities + actions)
    6. Average pairwise similarity to other cluster members (centrality)
    7. Tie-breaker: oldest published_at, then article_id / url
    """
    if not cluster:
        raise ValueError("Cannot select representative from empty cluster")
    if len(cluster) == 1:
        return cluster[0]

    def _evidence_rank(a: Article) -> int:
        lvl = a.evidence_level or ""
        if lvl == "primary" or a.is_primary_source or a.is_official or a.source_type == "official":
            return 3
        if lvl in {"independent", "research"} or a.is_independent_source:
            return 2
        if lvl == "industry_media" or a.source_type in {"media", "korean_media"}:
            return 1
        return 0

    scores = []
    for idx in cluster:
        art = articles[idx]
        feat = features[idx]

        is_off = 1 if (art.is_official or art.source_type == "official" or art.is_primary_source) else 0
        ev_rank = _evidence_rank(art)
        auth_score = getattr(art, "authority_score", 0.0) or getattr(art, "source_authority", 0.0) or 0.0
        p_score = art.priority_score or art.score or 0.0
        anchor_richness = len(feat.models) * 2 + len(feat.entities) + len(feat.actions)

        # Average similarity to other cluster members
        other_sims = [
            calculate_event_similarity(
                art,
                articles[o_idx],
                window_hours=window_hours,
                f1=feat,
                f2=features[o_idx],
            )
            for o_idx in cluster
            if o_idx != idx
        ]
        avg_centrality = sum(other_sims) / len(other_sims) if other_sims else 1.0

        pub_iso = art.published_at.isoformat() if art.published_at else ""
        uid = art.article_id or art.url or ""

        # For sorting: higher ranks/scores first, then earliest published_at (reverse string/negate)
        scores.append((
            is_off,
            ev_rank,
            auth_score,
            p_score,
            anchor_richness,
            avg_centrality,
            pub_iso,
            uid,
            idx,
        ))

    # Sort key: descending for quality/centrality, ascending for date/uid (using standard deterministic comparator)
    # We sort by (is_off DESC, ev_rank DESC, auth_score DESC, p_score DESC, anchor_richness DESC, avg_centrality DESC, pub_iso ASC, uid ASC)
    def rep_sort_key(item):
        return (
            -item[0],
            -item[1],
            -item[2],
            -item[3],
            -item[4],
            -item[5],
            item[6],  # oldest first
            item[7],  # stable uid
        )

    scores.sort(key=rep_sort_key)
    return scores[0][8]


def compute_cluster_anchor(
    cluster: list[int],
    features: list[ArticleClusteringFeatures],
) -> EventAnchor:
    """Compute the combined EventAnchor representing the cluster's aggregate semantic identity."""
    entities: set[str] = set()
    models: set[str] = set()
    themes: set[str] = set()
    actions: set[str] = set()

    for idx in cluster:
        f = features[idx]
        entities.update(f.entities)
        models.update(f.models)
        themes.update(f.themes)
        actions.update(f.actions)
        # Also map theme-derived actions
        if "model_release" in f.themes:
            actions.add("release")
        if "pricing" in f.themes:
            actions.add("pricing")
        if "legal_policy" in f.themes:
            actions.add("legal")
        if "security_safety" in f.themes:
            actions.add("security")
        if "office_expansion" in f.themes:
            actions.add("office_expansion")
        if "executive_movement" in f.themes:
            actions.add("executive_movement")
        if "investment" in f.themes:
            actions.add("investment")
        if "partnership" in f.themes:
            actions.add("partnership")

    return EventAnchor(
        entities=frozenset(entities),
        models=frozenset(models),
        themes=frozenset(themes),
        actions=frozenset(actions),
    )


def is_candidate_compatible_with_cluster(
    candidate_idx: int,
    cluster: list[int],
    articles: list[Article],
    features: list[ArticleClusteringFeatures],
    window_hours: float,
    similarity_threshold: float,
    min_cohesion_threshold: float,
) -> tuple[bool, float]:
    """Check if candidate article is cohesive with the cluster representative, all members, and cluster anchors.

    Returns (is_compatible, average_similarity_to_cluster).
    """
    candidate_art = articles[candidate_idx]
    candidate_feat = features[candidate_idx]

    # Select dynamic representative for the cluster
    rep_idx = select_cluster_representative(cluster, articles, features, window_hours=window_hours)
    rep_art = articles[rep_idx]
    rep_feat = features[rep_idx]

    sim_to_rep = calculate_event_similarity(
        candidate_art,
        rep_art,
        window_hours=window_hours,
        f1=candidate_feat,
        f2=rep_feat,
    )

    # Check maximum similarity to any cluster member as well
    member_sims = [
        calculate_event_similarity(
            candidate_art,
            articles[m_idx],
            window_hours=window_hours,
            f1=candidate_feat,
            f2=features[m_idx],
        )
        for m_idx in cluster
    ]
    max_member_sim = max(member_sims) if member_sims else 0.0

    if sim_to_rep < similarity_threshold and max_member_sim < similarity_threshold:
        return False, 0.0


    # Aggregate cluster anchor
    cluster_anchor = compute_cluster_anchor(cluster, features)

    # Candidate effective actions
    candidate_actions = set(candidate_feat.actions)
    if "model_release" in candidate_feat.themes:
        candidate_actions.add("release")
    if "pricing" in candidate_feat.themes:
        candidate_actions.add("pricing")
    if "legal_policy" in candidate_feat.themes:
        candidate_actions.add("legal")
    if "security_safety" in candidate_feat.themes:
        candidate_actions.add("security")
    if "office_expansion" in candidate_feat.themes:
        candidate_actions.add("office_expansion")
    if "executive_movement" in candidate_feat.themes:
        candidate_actions.add("executive_movement")
    if "investment" in candidate_feat.themes:
        candidate_actions.add("investment")
    if "partnership" in candidate_feat.themes:
        candidate_actions.add("partnership")

    # Cluster-level action conflict checking against ACTION_CONFLICTS
    # A true conflict occurs when candidate and cluster have disjoint clashing actions,
    # without a bridging action or joint context.
    has_shared_action = bool(candidate_actions & cluster_anchor.actions)
    has_shared_model = bool(candidate_feat.models & cluster_anchor.models)
    has_joint_context = (
        bool(cluster_anchor.themes & candidate_feat.themes & {"partnership", "security_safety", "legal_policy", "model_release"})
        or (len(cluster_anchor.entities & candidate_feat.entities) >= 2)
        or has_shared_model
    )

    for (act1, act2), penalty in ACTION_CONFLICTS.items():
        if (act1 in candidate_actions and act2 in cluster_anchor.actions) or (
            act2 in candidate_actions and act1 in cluster_anchor.actions
        ):
            if not (has_shared_action or has_joint_context):
                return False, 0.0


    # Verify pairwise cohesion across ALL members of the cluster
    sim_sum = 0.0
    for member_idx in cluster:
        member_art = articles[member_idx]
        member_feat = features[member_idx]
        member_sim = calculate_event_similarity(
            candidate_art,
            member_art,
            window_hours=window_hours,
            f1=candidate_feat,
            f2=member_feat,
        )
        if member_sim < min_cohesion_threshold or member_sim <= 0.0:
            return False, 0.0

        sim_sum += member_sim

    avg_sim = sim_sum / len(cluster)
    return True, avg_sim


def cluster_articles(
    articles: list[Article],
    similarity_threshold: float = 0.60,
    window_hours: float = 72.0,
) -> tuple[list[Event], list[EventArticle], list[Article]]:
    """Cluster articles into Events with parent-child relationships, chaining safeguards, and coverage metrics."""
    if not articles:
        return [], [], []

    features = [extract_clustering_features(a) for a in articles]
    n = len(articles)

    # Deterministic sorting order: Official first, then highest priority_score/score, oldest published_at, then article_id/url
    def sort_key(idx: int) -> tuple[int, float, str, str]:
        a = articles[idx]
        off = 1 if (a.is_official or a.source_type == "official" or a.is_primary_source) else 0
        p_score = a.priority_score or a.score or 0.0
        pub = a.published_at.isoformat() if a.published_at else ""
        uid = a.article_id or a.url or ""
        return (off, p_score, pub, uid)

    sorted_indices = sorted(range(n), key=sort_key, reverse=True)

    # Form clusters with cluster-representative and event anchor safeguards
    clusters: list[list[int]] = []
    min_cohesion_threshold = similarity_threshold * 0.65  # e.g. 0.39 for 0.60 threshold

    for idx in sorted_indices:
        best_cluster_idx = -1
        best_avg_sim = -1.0

        for c_idx, cluster in enumerate(clusters):
            is_compatible, avg_sim = is_candidate_compatible_with_cluster(
                candidate_idx=idx,
                cluster=cluster,
                articles=articles,
                features=features,
                window_hours=window_hours,
                similarity_threshold=similarity_threshold,
                min_cohesion_threshold=min_cohesion_threshold,
            )

            if is_compatible and avg_sim > best_avg_sim:
                best_avg_sim = avg_sim
                best_cluster_idx = c_idx

        if best_cluster_idx >= 0:
            clusters[best_cluster_idx].append(idx)
        else:
            clusters.append([idx])

    events: list[Event] = []
    event_articles: list[EventArticle] = []
    updated_articles: list[Article] = []

    for cluster_indices in clusters:
        cluster_items = [articles[idx] for idx in cluster_indices]
        primary = select_primary_article(cluster_items)

        # Generate event_id
        ev_id = f"ev_{hashlib.sha256(primary.url.encode()).hexdigest()[:16]}"
        event_title = primary.title

        # Legacy publisher set (raw, for backward-compat source_count fields)
        publishers = {a.publisher or a.source for a in cluster_items if (a.publisher or a.source)}
        has_official = any(a.is_official or a.source_type == "official" for a in cluster_items)
        has_reg = any(a.source_type == "regulator" for a in cluster_items)
        has_media = any(a.source_type in {"media", "korean_media"} for a in cluster_items)

        official_url = None
        official_name = None
        for a in cluster_items:
            if a.is_official or a.source_type == "official":
                official_url = a.url
                official_name = a.publisher or a.source
                break

        # Compute deduplicated evidence metadata
        ev_evidence = compute_event_evidence(cluster_items)

        # Compute evidence confidence score and explanation
        ev_confidence = compute_event_confidence(ev_evidence, cluster_items)

        # source_count: Number of distinct normalized publishers contributing coverage
        # independent_source_count: Number of distinct normalized publishers with evidence_level == "independent"
        src_count = len(ev_evidence.evidence_sources) if ev_evidence.evidence_sources else len(publishers)
        ind_src_count = len(ev_evidence.independent_sources)

        event = Event(
            event_id=ev_id,
            title=event_title,
            category=primary.category,
            importance=max(a.priority_score for a in cluster_items),
            primary_article_id=primary.article_id,
            source_count=src_count,
            independent_source_count=ind_src_count,
            has_official_source=has_official,
            has_regulatory_source=has_reg,
            has_major_media_source=has_media,
            official_source_url=official_url,
            official_source_name=official_name,
            related_sources=sorted(publishers),
            # Evidence metadata
            evidence_sources=ev_evidence.evidence_sources,
            primary_sources=ev_evidence.primary_sources,
            independent_sources=ev_evidence.independent_sources,
            evidence_diversity=ev_evidence.evidence_diversity,
            verification_status=ev_evidence.verification_status,
            # Confidence
            confidence_score=ev_confidence.confidence_score,
            confidence_label=ev_confidence.confidence_label,
            confidence_explanation_ko=ev_confidence.explanation_ko,
            confidence_explanation_en=ev_confidence.explanation_en,
        )
        events.append(event)


        rel_ids = [a.article_id for a in cluster_items if a.article_id]
        for a in cluster_items:
            rel = "primary" if a.article_id == primary.article_id else "coverage"
            event_articles.append(
                EventArticle(
                    event_id=ev_id,
                    article_id=a.article_id or "",
                    relationship=rel,
                    similarity=1.0 if rel == "primary" else 0.85,
                )
            )

            # Update article event fields
            a.event_id = ev_id
            a.event_title = event_title
            a.related_article_ids = rel_ids
            a.event_source_count = src_count
            a.event_independent_source_count = ind_src_count
            a.event_has_official_source = has_official
            a.event_has_regulatory_source = has_reg
            a.event_has_major_media_source = has_media
            a.event_official_source_url = official_url
            a.event_official_source_name = official_name
            a.event_related_sources = sorted(publishers)
            a.event_evidence_sources = ev_evidence.evidence_sources
            a.event_primary_sources = ev_evidence.primary_sources
            a.event_independent_sources = ev_evidence.independent_sources
            a.event_evidence_diversity = ev_evidence.evidence_diversity
            a.event_verification_status = ev_evidence.verification_status
            a.event_confidence_score = ev_confidence.confidence_score
            a.event_confidence_label = ev_confidence.confidence_label
            a.event_confidence_explanation_ko = ev_confidence.explanation_ko
            a.event_confidence_explanation_en = ev_confidence.explanation_en
            updated_articles.append(a)

    return events, event_articles, updated_articles


def select_primary_article(articles: list[Article]) -> Article:
    """Select the best primary article for an event:
    1. Official source
    2. Highest priority / source score
    3. Oldest / first-break timestamp
    """
    if not articles:
        raise ValueError("Cannot select primary article from empty list")

    def sort_key(a: Article) -> tuple[int, float, float]:
        official_rank = 1 if (a.is_official or a.source_type == "official") else 0
        p_score = a.priority_score or a.score
        s_score = a.source_score or 0.0
        return (official_rank, p_score, s_score)

    sorted_arts = sorted(articles, key=sort_key, reverse=True)
    return sorted_arts[0]

