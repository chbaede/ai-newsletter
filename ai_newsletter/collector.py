from __future__ import annotations

import base64
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from typing import Iterable
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import feedparser
import httpx

from .clustering import STOP_WORDS, _normalize_publisher, cluster_articles, select_primary_article
from .config import Settings, load_settings
from .content_extractor import extract_usable_article_text
from .logging import logger
from .models import Article, CollectionMetrics, FeedEntry, NewsletterIssue
from .sources import (
    DEFAULT_FEEDS,
    SECTION_ORDER,
    SourceFeed,
    classify_source_type,
    get_enabled_sources,
    get_source,
    source_authority,
    source_metadata,
)
from .store import NewsletterStore
from .summarizer import BaseSummarizer, classify_article, get_summarizer

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "utm_id",
    "fbclid", "gclid", "gclsrc", "dclid", "msclkid", "mc_cid", "mc_eid",
    "igshid", "yclid", "_hsenc", "_hsmi", "ref", "ref_src", "ref_url",
}
REQUEST_HEADERS = {"User-Agent": "AINewsletter/0.1 (+local research intelligence app)"}

TITLE_STOP_WORDS = STOP_WORDS | {"속보", "단독", "종합", "포토", "영상"}


def clean_title_for_comparison(title: str) -> str:
    title = re.sub(r"^\[.*?\]|\【.*?\】|\(.*?\)", " ", title)
    title = re.sub(r"(\s+[-–—]\s+|\s+\|\s+)[^–—|-]+$", "", title)
    title = re.sub(r"[^a-z0-9가-힣\s]", " ", title.lower())
    return " ".join(title.split())


def extract_title_tokens(title: str) -> set[str]:
    cleaned = clean_title_for_comparison(title)
    words = cleaned.split()
    return {w for w in words if w not in TITLE_STOP_WORDS and len(w) >= 2}


def _title_numbers(title: str) -> set[str]:
    return set(re.findall(r"\b\d+[a-z]?\b", title.lower()))


def are_titles_similar(t1: str, t2: str) -> bool:
    norm1 = normalize_title(t1)
    norm2 = normalize_title(t2)
    if norm1 == norm2 and norm1:
        return True

    if _title_numbers(t1) != _title_numbers(t2):
        return False

    tokens1 = extract_title_tokens(t1)
    tokens2 = extract_title_tokens(t2)
    if not tokens1 or not tokens2:
        return False

    common = tokens1 & tokens2
    max_len = max(len(tokens1), len(tokens2))
    min_len = min(len(tokens1), len(tokens2))

    if len(common) >= 4 and (len(common) / max_len >= 0.65 or len(common) / min_len >= 0.8):
        return True

    if len(common) >= 3 and min_len >= 4:
        jaccard = len(common) / len(tokens1 | tokens2)
        if jaccard >= 0.75:
            return True

    return False


def normalize_title(title: str) -> str:
    return " ".join(clean_title_for_comparison(title).split())


def canonicalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlsplit(url)
        filtered_query = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if k.lower() not in TRACKING_PARAMS
        ]
        from urllib.parse import urlencode
        clean_query = urlencode(filtered_query)
        clean_path = re.sub(r"/{2,}", "/", parsed.path)
        if clean_path.endswith("/") and clean_path != "/":
            clean_path = clean_path[:-1]
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), clean_path, clean_query, ""))
    except Exception:
        return url.strip()


def resolve_redirect_url(url: str, timeout: float = 6.0) -> str:
    if not url:
        return url
    if "news.google.com" not in url:
        return canonicalize_url(url)

    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=REQUEST_HEADERS) as client:
            resp = client.head(url)
            if resp.status_code < 400:
                return canonicalize_url(str(resp.url))
    except Exception:
        pass
    return canonicalize_url(url)


def parse_feed_entry(entry: dict, source_feed: SourceFeed) -> FeedEntry | None:
    title = unescape(entry.get("title", "")).strip()
    link = entry.get("link", "").strip()
    if not title or not link:
        return None

    published_at = None
    if "published" in entry:
        try:
            published_at = parsedate_to_datetime(entry["published"])
        except Exception:
            pass

    summary = unescape(entry.get("summary", "") or entry.get("description", "")).strip()
    clean_summary = re.sub(r"<[^>]+>", " ", summary)
    clean_summary = " ".join(clean_summary.split())

    content = ""
    for c in entry.get("content", []):
        if isinstance(c, dict) and "value" in c:
            content = c["value"]
            break

    # Determine publisher
    publisher = source_feed.name
    source_obj = entry.get("source")
    if isinstance(source_obj, dict) and source_obj.get("title"):
        publisher = source_obj.get("title").strip()
    elif " - " in title:
        parts = title.rsplit(" - ", 1)
        if len(parts) == 2 and len(parts[1].strip()) <= 40:
            publisher = parts[1].strip()

    return FeedEntry(
        title=title,
        url=link,
        source=source_feed.name,
        bucket=source_feed.bucket,
        published_at=published_at,
        excerpt=clean_summary[:500],
        discovered_via=source_feed.id,
        publisher=publisher,
        source_id=source_feed.id,
        authority_score=source_feed.authority_score,
        source_type=source_feed.source_type,
        source_authority=source_feed.authority_score,
        canonical_url=canonicalize_url(link),
        content=content,
        evidence_level=source_feed.evidence_level,
        is_primary_source=source_feed.is_primary_source,
        is_independent_source=source_feed.is_independent_source,
        is_discovery_source=source_feed.is_discovery_source,
    )


@dataclass(slots=True)
class FeedFetchResult:
    entries: list[FeedEntry]
    error: str | None = None
    http_status: int | None = None
    response_time: float = 0.0
    parsing_error: str | None = None


def fetch_source_feed(
    source_feed: SourceFeed,
    settings: Settings,
    client: httpx.Client,
) -> FeedFetchResult:
    start = time.perf_counter()
    http_status = None
    parsing_error = None
    try:
        resp = client.get(
            source_feed.url,
            headers=REQUEST_HEADERS,
            timeout=settings.request_timeout_seconds,
        )
        http_status = resp.status_code
        resp.raise_for_status()
        duration = round(time.perf_counter() - start, 4)

        parsed = feedparser.parse(resp.content)
        if parsed.bozo and not parsed.entries:
            parsing_error = str(getattr(parsed, 'bozo_exception', 'XML parsing failed'))
            err = f"피드 파싱 실패: {parsing_error}"
            logger.warning("collector", "feed_parse_error", source=source_feed.id, duration=duration, error=err)
            return FeedFetchResult(
                entries=[],
                error=err,
                http_status=http_status,
                response_time=duration,
                parsing_error=parsing_error,
            )

        entries: list[FeedEntry] = []
        for raw in parsed.entries[: settings.max_entries_per_feed]:
            parsed_entry = parse_feed_entry(raw, source_feed)
            if parsed_entry:
                entries.append(parsed_entry)

        logger.info("collector", "feed_fetched", source=source_feed.id, duration=duration)
        return FeedFetchResult(
            entries=entries,
            error=None,
            http_status=http_status,
            response_time=duration,
            parsing_error=None,
        )

    except httpx.HTTPStatusError as exc:
        duration = round(time.perf_counter() - start, 4)
        err = f"HTTP {exc.response.status_code}: {exc}"
        logger.warning("collector", "feed_fetch_failed", source=source_feed.id, duration=duration, error=err)
        return FeedFetchResult(
            entries=[],
            error=err,
            http_status=exc.response.status_code,
            response_time=duration,
        )
    except Exception as exc:
        duration = round(time.perf_counter() - start, 4)
        err = str(exc)
        logger.warning("collector", "feed_fetch_failed", source=source_feed.id, duration=duration, error=err)
        return FeedFetchResult(
            entries=[],
            error=err,
            http_status=http_status,
            response_time=duration,
        )


def collect_from_entries(
    entries: Iterable[FeedEntry],
    recent_articles: Iterable[Article] | None = None,
    summarizer: BaseSummarizer | None = None,
    settings: Settings | None = None,
) -> tuple[list[Article], list[str], int]:
    articles: list[Article] = []
    seen_urls: set[str] = set()
    # Scoped to canonical publisher: (canonical_publisher, normalized_title)
    seen_pub_titles: set[tuple[str, str]] = set()
    # Scoped to canonical publisher: canonical_publisher -> token -> list[title]
    pub_title_index: dict[str, dict[str, list[str]]] = {}
    suppressed_count = 0

    if recent_articles:
        for past in recent_articles:
            p_url = past.canonical_url or canonicalize_url(past.url)
            p_norm = normalize_title(past.title)
            p_pub = _normalize_publisher((past.publisher or past.source or "unknown").strip())
            if p_url:
                seen_urls.add(p_url)
            if p_norm and p_pub:
                seen_pub_titles.add((p_pub, p_norm))
            if p_pub:
                p_idx = pub_title_index.setdefault(p_pub, {})
                for tok in extract_title_tokens(past.title):
                    p_idx.setdefault(tok, []).append(past.title)

    for entry in entries:
        canonical_url = canonicalize_url(entry.url)
        title_key = normalize_title(entry.title)
        if not canonical_url or not title_key:
            continue

        canonical_pub = _normalize_publisher((entry.publisher or entry.source or "unknown").strip())

        # Exact canonical URL duplicate across any source is suppressed (Identity deduplication)
        if canonical_url in seen_urls:
            suppressed_count += 1
            continue

        # Exact same normalized title from the SAME publisher is suppressed
        if (canonical_pub, title_key) in seen_pub_titles:
            suppressed_count += 1
            continue

        # Similar title from the SAME publisher is suppressed
        tokens = extract_title_tokens(entry.title)
        p_idx = pub_title_index.setdefault(canonical_pub, {})
        if tokens:
            candidates = {cand for tok in tokens for cand in p_idx.get(tok, [])}
            if any(are_titles_similar(cand, entry.title) for cand in candidates):
                suppressed_count += 1
                continue

        seen_urls.add(canonical_url)
        seen_pub_titles.add((canonical_pub, title_key))
        for tok in tokens:
            p_idx.setdefault(tok, []).append(entry.title)

        # Build initial article
        raw_article = Article(
            title=entry.title,
            url=entry.url,
            source=entry.source,
            category=entry.bucket,
            published_at=entry.published_at,
            excerpt=entry.excerpt,
            discovered_via=entry.discovered_via,
            source_id=entry.source_id,
            authority_score=entry.authority_score,
            source_type=entry.source_type,
            source_authority=entry.source_authority,
            publisher=entry.publisher,
            canonical_url=canonical_url,
            content=entry.content,
            evidence_level=entry.evidence_level,
            is_primary_source=entry.is_primary_source,
            is_independent_source=entry.is_independent_source,
            is_discovery_source=entry.is_discovery_source,
        )

        classified = classify_article(
            raw_article,
            summarizer=summarizer,
            content=entry.content,
            is_short_excerpt=len(entry.content or entry.excerpt) < 200,
        )
        articles.append(classified)

    return articles, [], suppressed_count


def collect_and_store(
    store: NewsletterStore,
    settings: Settings | None = None,
    target_date: str | None = None,
    sources: list[SourceFeed] | None = None,
) -> NewsletterIssue:
    start_time = time.perf_counter()
    settings = settings or load_settings()
    today_str = target_date or date.today().isoformat()
    active_sources = sources or get_enabled_sources()

    all_entries: list[FeedEntry] = []
    warnings: list[str] = []
    feeds_ok = 0
    feeds_failed = 0

    # Source-role feed counts
    primary_feeds = 0
    research_feeds = 0
    independent_media_feeds = 0
    industry_media_feeds = 0
    community_feeds = 0
    discovery_feeds = 0

    for feed in active_sources:
        lvl = feed.evidence_level or "industry_media"
        if lvl == "primary":
            primary_feeds += 1
        elif lvl == "research":
            research_feeds += 1
        elif lvl == "independent":
            independent_media_feeds += 1
        elif lvl == "community":
            community_feeds += 1
        elif lvl == "discovery":
            discovery_feeds += 1
        else:
            industry_media_feeds += 1

    with httpx.Client(verify=settings.verify_tls) as client:
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_feed = {
                executor.submit(fetch_source_feed, feed, settings, client): feed
                for feed in active_sources
            }
            for future in as_completed(future_to_feed):
                feed = future_to_feed[future]
                try:
                    res = future.result()
                    entries, err = res.entries, res.error
                except Exception as exc:
                    entries, err = [], str(exc)
                if err:
                    feeds_failed += 1
                    warnings.append(f"{feed.name}: {err}")
                else:
                    feeds_ok += 1
                    all_entries.extend(entries)

    recent = store.recent_articles(days=3)
    summarizer = get_summarizer(settings)
    articles, dedupe_warnings, suppressed_count = collect_from_entries(
        all_entries,
        recent_articles=recent,
        summarizer=summarizer,
        settings=settings,
    )
    warnings.extend(dedupe_warnings)

    # Event clustering
    events, event_articles, clustered_articles = cluster_articles(articles)

    # Select top articles per category / priority
    clustered_articles.sort(key=lambda a: a.priority_score, reverse=True)

    # Compute quality metrics
    total_events = len(events)
    events_with_ind = sum(
        1 for ev in events
        if ev.verification_status in {"independently_reported", "multi_source"}
        or len(ev.independent_sources) > 0
    )
    avg_diversity = (
        sum(ev.evidence_diversity for ev in events) / total_events
        if total_events > 0
        else 0.0
    )

    top_stories = clustered_articles[:5]
    top_total = len(top_stories)
    pct_top_primary = (
        (sum(1 for a in top_stories if a.is_primary_source or a.evidence_level == "primary") / top_total * 100)
        if top_total > 0
        else 0.0
    )
    pct_with_ind = (
        (sum(1 for a in clustered_articles if a.is_independent_source or a.event_independent_source_count > 0 or a.evidence_level == "independent") / len(clustered_articles) * 100)
        if clustered_articles
        else 0.0
    )
    pct_discovery = (
        (sum(1 for a in clustered_articles if a.is_discovery_source or a.evidence_level == "discovery") / len(clustered_articles) * 100)
        if clustered_articles
        else 0.0
    )

    metrics = CollectionMetrics(
        feeds_total=len(active_sources),
        feeds_ok=feeds_ok,
        feeds_failed=feeds_failed,
        articles_collected=len(all_entries),
        articles_after_dedupe=len(clustered_articles),
        articles_selected=len(clustered_articles),
        collection_duration=time.perf_counter() - start_time,
        primary_feeds=primary_feeds,
        research_feeds=research_feeds,
        independent_media_feeds=independent_media_feeds,
        industry_media_feeds=industry_media_feeds,
        community_feeds=community_feeds,
        discovery_feeds=discovery_feeds,
        successful_feeds=feeds_ok,
        failed_feeds=feeds_failed,
        events_created=total_events,
        events_with_independent_confirmation=events_with_ind,
        duplicate_suppression_count=suppressed_count,
        percentage_top_stories_primary=pct_top_primary,
        percentage_with_independent_reporting=pct_with_ind,
        percentage_discovery_only=pct_discovery,
        average_evidence_diversity=avg_diversity,
    )

    issue = NewsletterIssue(
        issue_date=today_str,
        articles=clustered_articles,
        warnings=warnings,
        title="AI Newsletter & Intelligence Briefing",
        events=events,
        metrics=metrics.to_dict(),
    )

    store.save_issue(issue, event_articles=event_articles)
    return issue


def check_feed_health(
    feed_or_sources: SourceFeed | list[SourceFeed] | None = None,
    settings: Settings | None = None,
) -> list[dict[str, object]]:
    settings = settings or load_settings()
    if feed_or_sources is None:
        targets = get_enabled_sources()
    elif isinstance(feed_or_sources, SourceFeed):
        targets = [feed_or_sources]
    else:
        targets = feed_or_sources

    results = []
    with httpx.Client(verify=settings.verify_tls) as client:
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_feed = {
                executor.submit(fetch_source_feed, feed, settings, client): feed
                for feed in targets
            }
            for future in as_completed(future_to_feed):
                feed = future_to_feed[future]
                try:
                    res = future.result()
                    entries, err = res.entries, res.error
                    http_status = res.http_status
                    resp_time = res.response_time
                    parsing_err = res.parsing_error
                except Exception as exc:
                    entries, err = [], str(exc)
                    http_status = None
                    resp_time = 0.0
                    parsing_err = None

                # Source health classification: healthy, degraded, failing, disabled
                if not feed.enabled:
                    health_state = "disabled"
                elif err:
                    health_state = "failing"
                elif len(entries) == 0 or (resp_time > 5.0):
                    health_state = "degraded"
                else:
                    health_state = "healthy"

                results.append({
                    "id": feed.id,
                    "name": feed.name,
                    "url": feed.url,
                    "status": "ok" if not err else "error",
                    "health_state": health_state,
                    "entries_count": len(entries),
                    "http_status": http_status,
                    "response_time": resp_time,
                    "parsing_error": parsing_err,
                    "feed_type": feed.discovery_method,
                    "source_role": feed.evidence_level,
                    "error": err,
                    "last_successful_fetch": datetime.now(timezone.utc).isoformat() if not err else None,
                    "consecutive_failures": 1 if err else 0,
                })
    return results
