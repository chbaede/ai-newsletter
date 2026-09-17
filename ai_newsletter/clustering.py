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
    re.compile(r"\b(claude\s*3(?:\.5)?(?:\s*(?:sonnet|opus|haiku))?)\b", re.I),
    re.compile(r"\b(gemini\s*(?:1\.5|2\.0)?(?:\s*(?:flash|pro|ultra))?)\b", re.I),
    re.compile(r"\b(llama\s*(?:2|3|3\.1|3\.2|3\.3)?)\b", re.I),
    re.compile(r"\b(blackwell|b200|b100|h100|h200|gb200)\b", re.I),
    re.compile(r"\b(mi300[xa]?|mi325x|mi350)\b", re.I),
    re.compile(r"\b(sora|gen-3|kling|flux|midjourney\s*v[0-9]+)\b", re.I),
]

AI_EVENT_THEMES = {
    "model_release": ["release", "launches", "unveils", "debuts", "introduces", "open weights", "출시", "공개", "발표"],
    "benchmarking": ["swe-bench", "mmlu", "benchmark", "outperforms", "beats", "sota", "벤치마크", "성능"],
    "investment": ["funding", "valuation", "billion", "raises", "series", "ipo", "투자", "유치"],
    "legal_policy": ["lawsuit", "sues", "antitrust", "copyright", "ban", "compliance", "eu ai act", "소송", "규제"],
    "security": ["jailbreak", "vulnerability", "prompt injection", "exploit", "red team", "보안", "취약점"],
    "datacenter_hardware": ["datacenter", "tape-out", "foundry", "hbm", "hbm3e", "gigawatt", "반도체", "양산"],
}


@dataclass(slots=True)
class ArticleClusteringFeatures:
    text: str
    models: set[str]
    entities: set[str]
    tokens: set[str]
    themes: set[str]


def extract_clustering_features(article: Article) -> ArticleClusteringFeatures:
    text = f"{article.title} {article.excerpt} {' '.join(article.tags)}".lower()

    # Model identifiers
    models = set()
    for pat in AI_MODEL_PATTERNS:
        for match in pat.finditer(text):
            models.add(match.group(0).lower().replace(" ", "").replace("-", ""))

    # Entities
    found_entities = set(find_entities_in_text(text))

    # Clean tokens
    cleaned = re.sub(r"[^a-z0-9가-힣\s]", " ", article.title.lower())
    tokens = {w for w in cleaned.split() if len(w) >= 2 and w not in STOP_WORDS}

    # Themes
    themes = set()
    for theme_name, keywords in AI_EVENT_THEMES.items():
        if any(kw in text for kw in keywords):
            themes.add(theme_name)

    return ArticleClusteringFeatures(
        text=text,
        models=models,
        entities=found_entities,
        tokens=tokens,
        themes=themes,
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

    # Hard Negative Guards
    # 1. Model version collision (e.g. GPT-4o vs Claude 3.5, or Llama 2 vs Llama 3)
    if f1.models and f2.models and f1.models.isdisjoint(f2.models):
        return 0.0

    # 2. Competitor entity collision (e.g. purely Anthropic vs purely OpenAI without common theme or partnership)
    if f1.entities and f2.entities and f1.entities.isdisjoint(f2.entities):
        # Unless both talk about an alliance or common regulation
        if not (("legal_policy" in f1.themes and "legal_policy" in f2.themes) or "partnership" in f1.text):
            return 0.0

    # Token overlap (Jaccard similarity)
    if not f1.tokens or not f2.tokens:
        return 0.0

    intersect = len(f1.tokens & f2.tokens)
    union = len(f1.tokens | f2.tokens)
    jaccard = intersect / union if union > 0 else 0.0

    # Bonus for common entities and themes
    score = jaccard
    if f1.entities and f2.entities and not f1.entities.isdisjoint(f2.entities):
        score += 0.35
    if f1.themes and f2.themes and not f1.themes.isdisjoint(f2.themes):
        score += 0.20
    if f1.models and f2.models and not f1.models.isdisjoint(f2.models):
        score += 0.30

    return min(1.0, score)


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


def cluster_articles(
    articles: list[Article],
    similarity_threshold: float = 0.60,
    window_hours: float = 72.0,
) -> tuple[list[Event], list[EventArticle], list[Article]]:
    """Cluster articles into Events with parent-child relationships and coverage metrics."""
    if not articles:
        return [], [], []

    features = [extract_clustering_features(a) for a in articles]
    n = len(articles)

    # Disjoint-set forest for connected components
    parent = list(range(n))

    def find(i: int) -> int:
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    for i in range(n):
        for j in range(i + 1, n):
            sim = calculate_event_similarity(
                articles[i],
                articles[j],
                window_hours=window_hours,
                f1=features[i],
                f2=features[j],
            )
            if sim >= similarity_threshold:
                union(i, j)

    # Group into clusters
    clusters: dict[int, list[int]] = {}
    for idx in range(n):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    events: list[Event] = []
    event_articles: list[EventArticle] = []
    updated_articles: list[Article] = []

    for cluster_indices in clusters.values():
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

        event = Event(
            event_id=ev_id,
            title=event_title,
            category=primary.category,
            importance=max(a.priority_score for a in cluster_items),
            primary_article_id=primary.article_id,
            source_count=len(cluster_items),
            independent_source_count=len(publishers),
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
            a.event_source_count = len(cluster_items)
            a.event_independent_source_count = len(publishers)
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

