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

