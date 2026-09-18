from __future__ import annotations

import hmac
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from .collector import check_feed_health, collect_and_store
from .config import Settings, get_newsletter_timezone, load_settings, settings_with_mail_overrides
from .logging import logger
from .mailer import MailConfigError, send_issue
from .models import NewsletterIssue
from .presentation import (
    REGION_FILTERS,
    SOURCE_TYPE_FILTERS,
    TOPIC_FILTERS,
    UI_REGION_FILTERS,
    all_region_keys_for_article,
    build_intelligence_sections,
    canonical_source_type,
    compute_issue_metrics,
    display_event_coverage,
    display_factual_summary_en,
    display_factual_summary_ko,
    display_key_points,
    display_primary_category,
    display_published_time,
    display_source_transparency,
    display_summary_en,
    display_summary_ko,
    display_title_en,
    display_title_ko,
    display_url,
    display_why_it_matters_en,
    display_why_it_matters_ko,
    prepare_article_view,
    region_counts,
    regions_for_article,
    source_type_counts,
    source_type_keys_for_article,
    topic_counts,
    topic_keys_for_article,
    visible_tags,
)
from .priority import assess_priority, priority_summary
from .scheduler import DailyScheduler
from .sources import SECTION_LABELS, SECTION_LABELS_EN, SECTION_ORDER
from .store import NewsletterStore

PACKAGE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))


def create_app(store: NewsletterStore | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    store = store or NewsletterStore(settings.db_path)
    app = FastAPI(title="AI Newsletter & Intelligence Platform")

    proxies = (settings.trusted_proxies or "127.0.0.1,::1").strip()
    trusted = "*" if proxies == "*" else [h.strip() for h in proxies.split(",") if h.strip()]
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=trusted)
    app.state.store = store
    app.state.settings = settings
    app.mount("/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        issue = store.latest_issue()
        return _render_index(request, store, settings, issue)

    @app.get("/issues/{issue_date}", response_class=HTMLResponse)
    def issue_page(issue_date: str, request: Request) -> HTMLResponse:
        issue = store.get_issue(issue_date)
        if issue is None:
            raise HTTPException(status_code=404, detail="Issue not found")
        return _render_index(request, store, settings, issue)

    @app.get("/ads.txt", response_class=PlainTextResponse)
    def ads_txt() -> PlainTextResponse:
        content = "google.com, pub-6854824605420161, DIRECT, f08c47fec0942fa0\n"
        return PlainTextResponse(content, media_type="text/plain")

    @app.get("/robots.txt", response_class=PlainTextResponse)
    def robots_txt() -> PlainTextResponse:
        content = "User-agent: *\nAllow: /\nDisallow: /api/\n"
        return PlainTextResponse(content, media_type="text/plain")

    def _is_admin(request: Request) -> bool:
        if not settings.admin_key:
            # Fail closed in production. Only permit anonymous access when explicitly configured for dev/test
            if settings.allow_anonymous_admin:
                return True
            return False
        key = request.headers.get("X-Admin-Key")
        if not key:
            auth = request.headers.get("Authorization")
            if auth:
                parts = auth.split()
                if len(parts) == 2 and parts[0].lower() in {"bearer", "token"}:
                    key = parts[1]
                elif len(parts) == 1:
                    key = parts[0]
        if not key:
            return False
        return hmac.compare_digest(key.encode("utf-8"), settings.admin_key.encode("utf-8"))

    @app.get("/api/admin/verify")
    def verify_admin(request: Request) -> JSONResponse:
        return JSONResponse({"ok": True, "admin": _is_admin(request)})

    @app.post("/api/collect")
    def collect_today(request: Request) -> JSONResponse:
        if not _is_admin(request):
            return JSONResponse({"ok": False, "message": "관리자 권한이 필요합니다."}, status_code=401)
        tz = get_newsletter_timezone(settings.newsletter_timezone)
        today = datetime.now(tz).strftime("%Y-%m-%d")
        try:
            issue = collect_and_store(store=store, settings=settings, target_date=today)
            return JSONResponse({
                "ok": True,
                "message": f"{today} AI 뉴스 수집이 완료되었습니다. ({len(issue.articles)}개 기사)",
                "issue_date": today,
                "article_count": len(issue.articles),
            })
        except Exception as exc:
            logger.error("web", "manual_collect_failed", error=str(exc))
            return JSONResponse({"ok": False, "message": f"수집 실패: {exc}"}, status_code=500)

    @app.post("/api/issues/{issue_date}/send")
    def send_issue_api(issue_date: str, request: Request, lang: str = "ko") -> JSONResponse:
        if not _is_admin(request):
            return JSONResponse({"ok": False, "message": "관리자 권한이 필요합니다."}, status_code=401)
        issue = store.get_issue(issue_date)
        if issue is None:
            return JSONResponse({"ok": False, "message": "해당 이슈를 찾을 수 없습니다."}, status_code=404)
        try:
            send_issue(issue, settings=_effective_mail_settings(settings, store), lang=lang)
            store.mark_issue_sent(issue_date)
            return JSONResponse({"ok": True, "message": f"{issue_date} AI 뉴스레터({lang.upper()})가 발송되었습니다."})
        except MailConfigError as exc:
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
        except Exception as exc:
            logger.error("web", "send_issue_failed", error=str(exc))
            return JSONResponse({"ok": False, "message": f"발송 실패: {exc}"}, status_code=500)

    @app.get("/api/mail-settings")
    def get_mail_settings(request: Request) -> JSONResponse:
        if not _is_admin(request):
            return JSONResponse({"ok": False, "message": "관리자 권한이 필요합니다."}, status_code=401)
        return JSONResponse({"ok": True, "settings": _public_mail_settings(settings, store.mail_settings())})

    @app.post("/api/mail-settings")
    def update_mail_settings_api(request: Request, payload: dict[str, Any] = Body(...)) -> JSONResponse:
        if not _is_admin(request):
            return JSONResponse({"ok": False, "message": "관리자 권한이 필요합니다."}, status_code=401)
        try:
            current = store.mail_settings()
            normalized = _normalize_mail_settings_payload(payload, current)
            store.update_mail_settings(normalized)
            return JSONResponse({"ok": True, "message": "메일 설정이 저장되었습니다."})
        except ValueError as exc:
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)

    @app.get("/api/sources/health")
    def sources_health(request: Request) -> JSONResponse:
        if not _is_admin(request):
            return JSONResponse({"ok": False, "message": "관리자 권한이 필요합니다."}, status_code=401)
        health_data = check_feed_health(settings=settings)
        return JSONResponse({"ok": True, "sources": health_data})

    @app.get("/api/health")
    def health_check() -> JSONResponse:
        health_info = store.check_health()
        status_code = 200 if health_info.get("status") in {"healthy", "degraded"} else 503
        return JSONResponse(health_info, status_code=status_code)

    if settings.enable_daily_scheduler:
        start_daily_scheduler(app, collection_time=settings.daily_collection_time)

    return app


def _render_index(
    request: Request,
    store: NewsletterStore,
    settings: Settings,
    issue: NewsletterIssue | None,
) -> HTMLResponse:
    lang = request.query_params.get("lang") or settings.default_language or "ko"
    sections = []
    displayed_articles = []
    all_table_views = []
    if issue:
        sections = build_intelligence_sections(issue, lang=lang, min_score=60.0, max_per_section=6)
        displayed_articles = [article for sec in sections for article in sec["articles"]]
        sorted_all = sorted(issue.articles, key=lambda a: a.priority_score or a.score or 0.0, reverse=True)
        all_table_views = [prepare_article_view(a) for a in sorted_all]

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "issue": issue,
            "sections": sections,
            "all_table_views": all_table_views,
            "history": store.list_issues(),
            "today": date.today().isoformat(),
            "lang": lang,
            "assess_priority": assess_priority,
            "display_event_coverage": display_event_coverage,
            "display_factual_summary_ko": display_factual_summary_ko,
            "display_factual_summary_en": display_factual_summary_en,
            "display_why_it_matters_ko": display_why_it_matters_ko,
            "display_why_it_matters_en": display_why_it_matters_en,
            "display_primary_category": display_primary_category,
            "display_published_time": display_published_time,
            "display_source_transparency": display_source_transparency,
            "display_key_points": display_key_points,
            "display_summary_ko": display_summary_ko,
            "display_summary_en": display_summary_en,
            "display_title_ko": display_title_ko,
            "display_title_en": display_title_en,
            "display_url": display_url,
            "canonical_source_type": canonical_source_type,
            "region_counts": region_counts(displayed_articles) if issue else {},
            "region_filter_options": UI_REGION_FILTERS,
            "ui_region_filters": UI_REGION_FILTERS,
            "topic_filters": TOPIC_FILTERS,
            "topic_counts": topic_counts(displayed_articles) if issue else {},
            "source_type_filters": SOURCE_TYPE_FILTERS,
            "source_type_counts": source_type_counts(displayed_articles) if issue else {},
            "regions_for_article": regions_for_article,
            "all_region_keys_for_article": all_region_keys_for_article,
            "topic_keys_for_article": topic_keys_for_article,
            "source_type_keys_for_article": source_type_keys_for_article,
            "visible_tags": visible_tags,
            "mail_settings": _public_mail_settings(settings, store.mail_settings()),
            "issue_metrics": compute_issue_metrics(displayed_articles) if issue else {},
            "warning_items": _warning_items(issue.warnings) if issue else [],
            "priority_summary": priority_summary(issue) if issue else None,
        },
    )


def _warning_items(warnings: list[str]) -> list[dict[str, str]]:
    items = []
    for warning in warnings:
        source, separator, detail = warning.partition(":")
        if not separator:
            source = "수집기"
            detail = warning
        items.append({
            "source": source.strip() or "수집기",
            "detail": detail.strip(),
        })
    return items


def _effective_mail_settings(settings: Settings, store: NewsletterStore) -> Settings:
    return settings_with_mail_overrides(settings, store.mail_settings())


def _public_mail_settings(settings: Settings, saved_values: dict[str, object]) -> dict[str, object]:
    effective = settings_with_mail_overrides(settings, saved_values)
    return {
        "smtp_host": effective.smtp_host or "",
        "smtp_port": effective.smtp_port,
        "smtp_user": effective.smtp_user or "",
        "smtp_from": effective.smtp_from or "",
        "newsletter_to": effective.newsletter_to or "",
        "smtp_tls": effective.smtp_tls,
        "smtp_password_saved": bool(effective.smtp_password),
    }


def _normalize_mail_settings_payload(
    payload: dict[str, object], existing: dict[str, object]
) -> dict[str, object]:
    def text(name: str) -> str:
        value = payload.get(name)
        return str(value).strip() if value is not None else ""

    try:
        smtp_port = int(payload.get("smtp_port") or 587)
    except (TypeError, ValueError) as exc:
        raise ValueError("SMTP_PORT는 숫자로 입력해야 합니다.") from exc

    password = text("smtp_password") or str(existing.get("smtp_password") or "")
    smtp_tls = payload.get("smtp_tls")
    use_tls = bool(smtp_tls) if isinstance(smtp_tls, bool) else str(smtp_tls or "").lower() in {"1", "true", "yes", "on"}

    return {
        "smtp_host": text("smtp_host"),
        "smtp_port": smtp_port,
        "smtp_user": text("smtp_user"),
        "smtp_password": password,
        "smtp_from": text("smtp_from"),
        "newsletter_to": text("newsletter_to"),
        "smtp_tls": use_tls,
    }


def start_daily_scheduler(app: FastAPI, collection_time: str) -> None:
    scheduler = DailyScheduler(store=app.state.store, settings=app.state.settings)

    @app.on_event("startup")
    def _start_scheduler() -> None:
        thread = threading.Thread(target=scheduler.run_loop, daemon=True)
        thread.start()

