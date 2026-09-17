from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import Article, Event, EventArticle, NewsletterIssue


class NewsletterStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists issues (
                    id integer primary key autoincrement,
                    issue_date text unique not null,
                    title text not null,
                    warnings text not null default '[]',
                    metrics text not null default '{}',
                    created_at text not null,
                    updated_at text not null,
                    sent_at text
                )
                """
            )
            conn.execute(
                """
                create table if not exists articles (
                    id integer primary key autoincrement,
                    issue_id integer not null references issues(id) on delete cascade,
                    position integer not null,
                    title text not null,
                    url text not null,
                    source text not null,
                    category text not null,
                    published_at text,
                    summary_ko text not null,
                    summary_en text not null default '',
                    why_it_matters_ko text not null default '',
                    why_it_matters_en text not null default '',
                    title_ko text not null default '',
                    title_en text not null default '',
                    excerpt text not null,
                    tags text not null,
                    score real not null,
                    article_id text,
                    canonical_url text,
                    original_url text,
                    source_id text,
                    publisher text,
                    discovered_via text,
                    source_type text not null default 'media',
                    source_authority integer not null default 70,
                    primary_category text,
                    secondary_categories text not null default '[]',
                    topics text not null default '[]',
                    entities text not null default '[]',
                    source_score real not null default 0,
                    relevance_score real not null default 0,
                    impact_score real not null default 0,
                    novelty_score real not null default 0,
                    recency_score real not null default 0,
                    evidence_quality_score real not null default 0,
                    priority_score real not null default 0,
                    event_id text,
                    event_title text,
                    related_article_ids text not null default '[]',
                    is_official integer not null default 0,
                    is_reference integer not null default 0,
                    is_primary_source integer not null default 0,
                    is_independent_source integer not null default 0,
                    is_discovery_source integer not null default 0,
                    evidence_level text not null default 'industry_media',
                    collected_at text,
                    content text not null default '',
                    key_points text not null default '[]',
                    summary_model text,
                    summary_version text,
                    summary_created_at text,
                    content_source_type text not null default 'fallback'
                )
                """
            )
            article_cols = [r[1] for r in conn.execute("pragma table_info(articles)").fetchall()]
            if "is_independent_source" not in article_cols:
                conn.execute("alter table articles add column is_independent_source integer not null default 0")
            if "is_discovery_source" not in article_cols:
                conn.execute("alter table articles add column is_discovery_source integer not null default 0")
            if "evidence_level" not in article_cols:
                conn.execute("alter table articles add column evidence_level text not null default 'industry_media'")
            if "evidence_quality_score" not in article_cols:
                conn.execute("alter table articles add column evidence_quality_score real not null default 0")
            conn.execute(
                """
                create table if not exists events (
                    event_id text primary key,
                    issue_id integer not null references issues(id) on delete cascade,
                    title text not null,
                    category text not null default 'big',
                    importance real not null default 0.0,
                    primary_article_id text,
                    created_at text not null,
                    source_count integer not null default 1,
                    independent_source_count integer not null default 1,
                    has_official_source integer not null default 0,
                    has_regulatory_source integer not null default 0,
                    has_major_media_source integer not null default 0,
                    official_source_url text,
                    official_source_name text,
                    reference_source_name text,
                    related_sources text not null default '[]',
                    evidence_sources text not null default '[]',
                    primary_sources text not null default '[]',
                    independent_sources text not null default '[]',
                    evidence_diversity integer not null default 0,
                    verification_status text not null default 'insufficient_evidence',
                    confidence_score real not null default 0.0,
                    confidence_label text not null default 'Low',
                    confidence_explanation_ko text not null default '',
                    confidence_explanation_en text not null default ''
                )
                """
            )
            # Migration: add new evidence and confidence columns to existing tables
            event_cols = [r[1] for r in conn.execute("pragma table_info(events)").fetchall()]
            for col, col_type, default in [
                ("evidence_sources", "text", "'[]'"),
                ("primary_sources", "text", "'[]'"),
                ("independent_sources", "text", "'[]'"),
                ("evidence_diversity", "integer", "0"),
                ("verification_status", "text", "'insufficient_evidence'"),
                ("confidence_score", "real", "0.0"),
                ("confidence_label", "text", "'Low'"),
                ("confidence_explanation_ko", "text", "''"),
                ("confidence_explanation_en", "text", "''"),
            ]:
                if col not in event_cols:
                    conn.execute(
                        f"alter table events add column {col} {col_type} not null default {default}"
                    )


            conn.execute(
                """
                create table if not exists event_articles (
                    id integer primary key autoincrement,
                    event_id text not null references events(event_id) on delete cascade,
                    article_id text not null,
                    relationship text not null default 'coverage',
                    similarity real not null default 1.0,
                    unique(event_id, article_id)
                )
                """
            )
            conn.execute(
                """
                create table if not exists mail_settings (
                    key text primary key,
                    value text not null
                )
                """
            )
            conn.execute(
                """
                create table if not exists scheduler_runs (
                    id integer primary key autoincrement,
                    job_name text not null,
                    run_date text not null,
                    status text not null,
                    started_at text not null,
                    finished_at text,
                    metrics text not null default '{}',
                    error text,
                    unique(job_name, run_date)
                )
                """
            )


    def save_issue(
        self,
        issue_or_date: NewsletterIssue | str,
        articles: Iterable[Article] | None = None,
        warnings: list[str] | None = None,
        events: Iterable[Event] | None = None,
        event_articles: Iterable[EventArticle] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> NewsletterIssue:
        if isinstance(issue_or_date, NewsletterIssue):
            issue = issue_or_date
            issue_date = issue.issue_date
            article_list = issue.articles
            warning_list = issue.warnings
            event_list = issue.events
            metrics_dict = issue.metrics
            title_text = issue.title
        else:
            issue_date = issue_or_date
            article_list = list(articles or [])
            warning_list = warnings or []
            event_list = list(events or [])
            metrics_dict = metrics or {}
            title_text = f"{issue_date} AI Newsletter"

        event_article_list = list(event_articles or [])
        now = datetime.now(timezone.utc).isoformat()
        metrics_payload = json.dumps(metrics_dict, ensure_ascii=False)
        warnings_payload = json.dumps(warning_list, ensure_ascii=False)

        with self._connect() as conn:
            existing = conn.execute(
                "select id from issues where issue_date = ?", (issue_date,)
            ).fetchone()

            if existing:
                issue_id = existing["id"]
                conn.execute(
                    "update issues set title = ?, warnings = ?, metrics = ?, updated_at = ? where id = ?",
                    (title_text, warnings_payload, metrics_payload, now, issue_id),
                )
                conn.execute(
                    "delete from event_articles where event_id in (select event_id from events where issue_id = ?)",
                    (issue_id,),
                )
                conn.execute("delete from events where issue_id = ?", (issue_id,))
                conn.execute("delete from articles where issue_id = ?", (issue_id,))
            else:
                cursor = conn.execute(
                    """
                    insert into issues (issue_date, title, warnings, metrics, created_at, updated_at)
                    values (?, ?, ?, ?, ?, ?)
                    """,
                    (issue_date, title_text, warnings_payload, metrics_payload, now, now),
                )
                issue_id = cursor.lastrowid

            for index, article in enumerate(article_list):
                pub_iso = article.published_at.isoformat() if article.published_at else None
                col_iso = article.collected_at.isoformat() if article.collected_at else None
                sum_iso = article.summary_created_at.isoformat() if article.summary_created_at else None

                conn.execute(
                    """
                    insert into articles (
                        issue_id, position, title, url, source, category, published_at,
                        summary_ko, summary_en, why_it_matters_ko, why_it_matters_en,
                        title_ko, title_en, excerpt, tags, score,
                        article_id, canonical_url, original_url, source_id, publisher,
                        discovered_via, source_type, source_authority,
                        primary_category, secondary_categories, topics, entities,
                        source_score, relevance_score, impact_score, novelty_score, recency_score,
                        evidence_quality_score,
                        priority_score, event_id, event_title, related_article_ids,
                        is_official, is_reference, is_primary_source,
                        is_independent_source, is_discovery_source, evidence_level,
                        collected_at,
                        content, key_points, summary_model, summary_version, summary_created_at,
                        content_source_type
                    )
                    values (
                        ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?,
                        ?,
                        ?, ?, ?, ?, ?,
                        ?
                    )
                    """,
                    (
                        issue_id,
                        index,
                        article.title,
                        article.url,
                        article.source,
                        article.category,
                        pub_iso,
                        article.summary_ko,
                        article.summary_en,
                        article.why_it_matters_ko,
                        article.why_it_matters_en,
                        article.title_ko,
                        article.title_en,
                        article.excerpt,
                        json.dumps(article.tags, ensure_ascii=False),
                        article.score,
                        article.article_id,
                        article.canonical_url,
                        article.original_url,
                        article.source_id,
                        article.publisher,
                        article.discovered_via,
                        article.source_type,
                        article.source_authority,
                        article.primary_category,
                        json.dumps(article.secondary_categories, ensure_ascii=False),
                        json.dumps(article.topics, ensure_ascii=False),
                        json.dumps(article.entities, ensure_ascii=False),
                        article.source_score,
                        article.relevance_score,
                        article.impact_score,
                        article.novelty_score,
                        article.recency_score,
                        article.evidence_quality_score,
                        article.priority_score,
                        article.event_id,
                        article.event_title,
                        json.dumps(article.related_article_ids, ensure_ascii=False),
                        1 if article.is_official else 0,
                        1 if article.is_reference else 0,
                        1 if article.is_primary_source else 0,
                        1 if article.is_independent_source else 0,
                        1 if article.is_discovery_source else 0,
                        article.evidence_level,
                        col_iso,
                        article.content,
                        json.dumps(article.key_points, ensure_ascii=False),
                        article.summary_model,
                        article.summary_version,
                        sum_iso,
                        article.content_source_type,
                    ),
                )

            for event in event_list:
                ev_created = event.created_at.isoformat() if event.created_at else now
                conn.execute(
                    """
                    insert or replace into events (
                        event_id, issue_id, title, category, importance, primary_article_id,
                        created_at, source_count, independent_source_count,
                        has_official_source, has_regulatory_source, has_major_media_source,
                        official_source_url, official_source_name, reference_source_name,
                        related_sources,
                        evidence_sources, primary_sources, independent_sources,
                        evidence_diversity, verification_status,
                        confidence_score, confidence_label,
                        confidence_explanation_ko, confidence_explanation_en
                    )
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        issue_id,
                        event.title,
                        event.category,
                        event.importance,
                        event.primary_article_id,
                        ev_created,
                        event.source_count,
                        event.independent_source_count,
                        1 if event.has_official_source else 0,
                        1 if event.has_regulatory_source else 0,
                        1 if event.has_major_media_source else 0,
                        event.official_source_url,
                        event.official_source_name,
                        event.reference_source_name,
                        json.dumps(event.related_sources, ensure_ascii=False),
                        json.dumps(event.evidence_sources, ensure_ascii=False),
                        json.dumps(event.primary_sources, ensure_ascii=False),
                        json.dumps(event.independent_sources, ensure_ascii=False),
                        event.evidence_diversity,
                        event.verification_status,
                        event.confidence_score,
                        event.confidence_label,
                        event.confidence_explanation_ko,
                        event.confidence_explanation_en,
                    ),
                )


            for ea in event_article_list:
                conn.execute(
                    """
                    insert or ignore into event_articles (event_id, article_id, relationship, similarity)
                    values (?, ?, ?, ?)
                    """,
                    (ea.event_id, ea.article_id, ea.relationship, ea.similarity),
                )

        return self.get_issue(issue_date) or NewsletterIssue(
            issue_date=issue_date, articles=article_list, warnings=warning_list, events=event_list, metrics=metrics_dict
        )

    def get_issue(self, issue_date: str) -> NewsletterIssue | None:
        with self._connect() as conn:
            issue_row = conn.execute(
                "select * from issues where issue_date = ?", (issue_date,)
            ).fetchone()
            if not issue_row:
                return None
            return self._build_issue(conn, issue_row)

    def latest_issue(self) -> NewsletterIssue | None:
        with self._connect() as conn:
            issue_row = conn.execute(
                "select * from issues order by issue_date desc limit 1"
            ).fetchone()
            if not issue_row:
                return None
            return self._build_issue(conn, issue_row)

    def list_issues(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                select i.issue_date, i.title, i.created_at, i.sent_at, count(a.id) as article_count
                from issues i
                left join articles a on i.id = a.issue_id
                group by i.id
                order by i.issue_date desc
                """
            ).fetchall()
            return [
                {
                    "issue_date": row["issue_date"],
                    "title": row["title"],
                    "created_at": row["created_at"],
                    "sent_at": row["sent_at"],
                    "article_count": row["article_count"],
                }
                for row in rows
            ]

    def mark_issue_sent(self, issue_date: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "update issues set sent_at = ? where issue_date = ?", (now, issue_date)
            )

    def recent_articles(self, days: int = 3, now_dt: datetime | None = None) -> list[Article]:
        """Retrieve recent articles within the specified number of days for deduplication.

        Uses published_at when available, falling back to collected_at.
        Articles with invalid/non-positive days return an empty list.
        """
        if not isinstance(days, (int, float)) or days <= 0:
            return []

        now = now_dt or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)

        cutoff = now - timedelta(days=days)

        with self._connect() as conn:
            rows = conn.execute("select * from articles order by id desc").fetchall()
            results: list[Article] = []
            for r in rows:
                art = self._row_to_article(r)
                ref_dt = art.published_at or art.collected_at
                if ref_dt is not None:
                    if ref_dt.tzinfo is None:
                        ref_dt = ref_dt.replace(tzinfo=timezone.utc)
                    else:
                        ref_dt = ref_dt.astimezone(timezone.utc)

                    if ref_dt >= cutoff:
                        results.append(art)
            return results

    def mail_settings(self) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute("select key, value from mail_settings").fetchall()
            return {r["key"]: r["value"] for r in rows}

    def update_mail_settings(self, settings_dict: dict[str, Any]) -> None:
        with self._connect() as conn:
            for k, v in settings_dict.items():
                val_str = str(v) if v is not None else ""
                conn.execute(
                    "insert or replace into mail_settings (key, value) values (?, ?)",
                    (k, val_str),
                )

    def _build_issue(self, conn: sqlite3.Connection, issue_row: sqlite3.Row) -> NewsletterIssue:
        issue_id = issue_row["id"]
        article_rows = conn.execute(
            "select * from articles where issue_id = ? order by position asc", (issue_id,)
        ).fetchall()
        articles = [self._row_to_article(r) for r in article_rows]

        event_rows = conn.execute(
            "select * from events where issue_id = ?", (issue_id,)
        ).fetchall()
        events = [self._row_to_event(r) for r in event_rows]

        warnings = json.loads(issue_row["warnings"] or "[]")
        metrics = json.loads(issue_row["metrics"] or "{}")

        created_at = None
        if issue_row["created_at"]:
            try:
                created_at = datetime.fromisoformat(issue_row["created_at"])
            except Exception:
                pass

        sent_at = None
        if issue_row["sent_at"]:
            try:
                sent_at = datetime.fromisoformat(issue_row["sent_at"])
            except Exception:
                pass

        return NewsletterIssue(
            issue_date=issue_row["issue_date"],
            articles=articles,
            warnings=warnings,
            title=issue_row["title"],
            created_at=created_at,
            sent_at=sent_at,
            events=events,
            metrics=metrics,
        )

    def _row_to_article(self, row: sqlite3.Row) -> Article:
        def _parse_dt(val: str | None) -> datetime | None:
            if not val:
                return None
            try:
                return datetime.fromisoformat(val)
            except Exception:
                return None

        def _parse_json(val: str | None, default):
            if not val:
                return default
            try:
                return json.loads(val)
            except Exception:
                return default

        keys = row.keys()
        title_ko = row["title_ko"] if "title_ko" in keys else ""
        title_en = row["title_en"] if "title_en" in keys else ""
        why_it_matters_en = row["why_it_matters_en"] if "why_it_matters_en" in keys else ""

        return Article(
            title=row["title"],
            url=row["url"],
            source=row["source"],
            category=row["category"],
            published_at=_parse_dt(row["published_at"]),
            summary_ko=row["summary_ko"],
            summary_en=row["summary_en"] or "",
            why_it_matters_ko=row["why_it_matters_ko"] or "",
            why_it_matters_en=why_it_matters_en,
            title_ko=title_ko,
            title_en=title_en,
            excerpt=row["excerpt"] or "",
            tags=_parse_json(row["tags"], []),
            score=row["score"] or 0.0,
            discovered_via=row["discovered_via"],
            source_id=row["source_id"],
            authority_score=row["source_authority"],
            source_type=row["source_type"] or "media",
            source_authority=row["source_authority"],
            publisher=row["publisher"],
            article_id=row["article_id"],
            canonical_url=row["canonical_url"],
            original_url=row["original_url"],
            content=row["content"] or "",
            content_source_type=row["content_source_type"] or "fallback",
            key_points=_parse_json(row["key_points"], []),
            summary_model=row["summary_model"],
            summary_version=row["summary_version"],
            summary_created_at=_parse_dt(row["summary_created_at"]),
            primary_category=row["primary_category"],
            secondary_categories=_parse_json(row["secondary_categories"], []),
            topics=_parse_json(row["topics"], []),
            entities=_parse_json(row["entities"], []),
            source_score=row["source_score"] or 0.0,
            relevance_score=row["relevance_score"] or 0.0,
            impact_score=row["impact_score"] or 0.0,
            novelty_score=row["novelty_score"] or 0.0,
            recency_score=row["recency_score"] or 0.0,
            evidence_quality_score=row["evidence_quality_score"] if "evidence_quality_score" in keys and row["evidence_quality_score"] is not None else 0.0,
            priority_score=row["priority_score"] or 0.0,
            event_id=row["event_id"],
            event_title=row["event_title"],
            related_article_ids=_parse_json(row["related_article_ids"], []),
            is_official=bool(row["is_official"]),
            is_reference=bool(row["is_reference"]),
            is_primary_source=bool(row["is_primary_source"]),
            is_independent_source=bool(row["is_independent_source"]) if "is_independent_source" in keys else False,
            is_discovery_source=bool(row["is_discovery_source"]) if "is_discovery_source" in keys else False,
            evidence_level=row["evidence_level"] if "evidence_level" in keys and row["evidence_level"] else "industry_media",
            collected_at=_parse_dt(row["collected_at"]),
        )

    def _row_to_event(self, row: sqlite3.Row) -> Event:
        created_at = None
        if row["created_at"]:
            try:
                created_at = datetime.fromisoformat(row["created_at"])
            except Exception:
                pass
        keys = row.keys() if hasattr(row, "keys") else []
        rel_sources = []
        if row["related_sources"]:
            try:
                rel_sources = json.loads(row["related_sources"])
            except Exception:
                pass

        def _load_list(col: str) -> list:
            if col not in keys:
                return []
            raw = row[col]
            if not raw:
                return []
            try:
                return json.loads(raw)
            except Exception:
                return []

        return Event(
            event_id=row["event_id"],
            title=row["title"],
            category=row["category"],
            importance=row["importance"],
            primary_article_id=row["primary_article_id"],
            created_at=created_at,
            source_count=row["source_count"],
            independent_source_count=row["independent_source_count"],
            has_official_source=bool(row["has_official_source"]),
            has_regulatory_source=bool(row["has_regulatory_source"]),
            has_major_media_source=bool(row["has_major_media_source"]),
            official_source_url=row["official_source_url"],
            official_source_name=row["official_source_name"],
            reference_source_name=row["reference_source_name"],
            related_sources=rel_sources,
            evidence_sources=_load_list("evidence_sources"),
            primary_sources=_load_list("primary_sources"),
            independent_sources=_load_list("independent_sources"),
            evidence_diversity=row["evidence_diversity"] if "evidence_diversity" in keys else 0,
            verification_status=(
                row["verification_status"]
                if "verification_status" in keys and row["verification_status"]
                else "insufficient_evidence"
            ),
            confidence_score=row["confidence_score"] if "confidence_score" in keys and row["confidence_score"] is not None else 0.0,
            confidence_label=row["confidence_label"] if "confidence_label" in keys and row["confidence_label"] else "Low",
            confidence_explanation_ko=row["confidence_explanation_ko"] if "confidence_explanation_ko" in keys and row["confidence_explanation_ko"] else "",
            confidence_explanation_en=row["confidence_explanation_en"] if "confidence_explanation_en" in keys and row["confidence_explanation_en"] else "",
        )


    def acquire_daily_run(
        self,
        run_date: str,
        job_name: str = "daily_collection",
        timeout_minutes: int = 30,
    ) -> bool:
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "select status, started_at from scheduler_runs where job_name = ? and run_date = ?",
                (job_name, run_date),
            ).fetchone()
            if row is not None:
                status = row["status"]
                if status == "completed":
                    return False
                if status == "running":
                    started = None
                    try:
                        started = datetime.fromisoformat(row["started_at"])
                    except Exception:
                        pass
                    if started and (now - started).total_seconds() < timeout_minutes * 60:
                        return False
                    conn.execute(
                        "update scheduler_runs set status = 'running', started_at = ?, error = null where job_name = ? and run_date = ?",
                        (now_iso, job_name, run_date),
                    )
                    return True
                conn.execute(
                    "update scheduler_runs set status = 'running', started_at = ?, error = null where job_name = ? and run_date = ?",
                    (now_iso, job_name, run_date),
                )
                return True
            try:
                conn.execute(
                    "insert into scheduler_runs (job_name, run_date, status, started_at, metrics) values (?, ?, 'running', ?, '{}')",
                    (job_name, run_date, now_iso),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def finish_daily_run(
        self,
        run_date: str,
        metrics: dict[str, Any] | None = None,
        job_name: str = "daily_collection",
        status: str = "completed",
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        metrics_json = json.dumps(metrics or {}, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                insert into scheduler_runs (job_name, run_date, status, started_at, finished_at, metrics, error)
                values (?, ?, ?, ?, ?, ?, null)
                on conflict(job_name, run_date) do update set
                    status = excluded.status,
                    finished_at = excluded.finished_at,
                    metrics = excluded.metrics,
                    error = null
                """,
                (job_name, run_date, status, now_iso, now_iso, metrics_json),
            )

    def fail_daily_run(
        self,
        run_date: str,
        error: str,
        job_name: str = "daily_collection",
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                insert into scheduler_runs (job_name, run_date, status, started_at, finished_at, error)
                values (?, ?, 'failed', ?, ?, ?)
                on conflict(job_name, run_date) do update set
                    status = 'failed',
                    finished_at = excluded.finished_at,
                    error = excluded.error
                """,
                (job_name, run_date, now_iso, now_iso, str(error)),
            )

    def has_daily_run_completed(self, run_date: str, job_name: str = "daily_collection") -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "select status from scheduler_runs where job_name = ? and run_date = ?",
                (job_name, run_date),
            ).fetchone()
            if row and row["status"] == "completed":
                return True
            issue_row = conn.execute(
                "select id from issues where issue_date = ?", (run_date,)
            ).fetchone()
            return issue_row is not None

