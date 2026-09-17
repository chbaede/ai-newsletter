from __future__ import annotations

import argparse
import sys
from datetime import date, datetime

import uvicorn

from .collector import check_feed_health, collect_and_store
from .config import get_newsletter_timezone, load_settings
from .mailer import MailConfigError, send_issue
from .scheduler import DailyScheduler, run_collection_job
from .sources import SOURCE_CATALOG, get_enabled_sources
from .store import NewsletterStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ai-newsletter", description="AI Newsletter & Intelligence CLI")
    subparsers = parser.add_subparsers(dest="command")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI web dashboard")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", default=None, type=int, help="Port to bind (default: 8001 or $PORT)")
    serve_parser.add_argument("--reload", action="store_true")

    # collect
    collect_parser = subparsers.add_parser("collect", help="Collect and store AI news issue")
    collect_parser.add_argument("--date", dest="issue_date", help="Issue date (YYYY-MM-DD)")
    collect_parser.add_argument("--force", action="store_true", help="Force collection even if already completed")

    # send
    send_parser = subparsers.add_parser("send", help="Send an issue by email")
    send_parser.add_argument("--date", dest="issue_date", help="Issue date (YYYY-MM-DD)")
    send_parser.add_argument("--lang", default="ko", choices=["ko", "en"], help="Email language edition (ko or en)")

    # sources
    subparsers.add_parser("sources", help="Check feed catalog status and network health")

    # schedule
    schedule_parser = subparsers.add_parser("schedule", help="Run standalone background scheduler daemon")
    schedule_parser.add_argument("--time", dest="collection_time", help="Collection time HH:MM")

    args = parser.parse_args(argv)
    command = args.command or "serve"

    if command == "serve":
        settings = load_settings()
        port = args.port if args.port is not None else settings.port
        uvicorn.run(
            "ai_newsletter.web:create_app",
            factory=True,
            host=args.host,
            port=port,
            reload=args.reload,
            proxy_headers=True,
            forwarded_allow_ips=settings.forwarded_allow_ips,
        )
        return 0

    if command == "collect":
        settings = load_settings()
        tz = get_newsletter_timezone(settings.newsletter_timezone)
        issue_date = args.issue_date or datetime.now(tz).strftime("%Y-%m-%d")
        store = NewsletterStore(settings.db_path)
        if not args.force and store.has_daily_run_completed(issue_date):
            print(f"Collection for {issue_date} has already completed (use --force to re-collect)")
            return 0

        success = run_collection_job(
            store=store,
            settings=settings,
            issue_date=issue_date,
            force=args.force,
        )
        if success:
            issue = store.get_issue(issue_date)
            count = len(issue.articles) if issue else 0
            print(f"Successfully collected {issue_date}: {count} articles")
            return 0
        else:
            print(f"Collection for {issue_date} skipped (locked by another worker)")
            return 1

    if command == "send":
        settings = load_settings()
        store = NewsletterStore(settings.db_path)
        tz = get_newsletter_timezone(settings.newsletter_timezone)
        issue_date = args.issue_date or datetime.now(tz).strftime("%Y-%m-%d")
        issue = store.get_issue(issue_date)
        if not issue:
            print(f"No issue found for date {issue_date}")
            return 1
        try:
            send_issue(issue, settings=settings, lang=args.lang)
            store.mark_issue_sent(issue_date)
            print(f"Email for {issue_date} ({args.lang.upper()}) sent successfully to {', '.join(settings.recipients)}")
            return 0
        except MailConfigError as exc:
            print(f"Mail configuration error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Failed to send email: {exc}", file=sys.stderr)
            return 1

    if command == "sources":
        settings = load_settings()
        print(f"AI Newsletter Feed Catalog ({len(SOURCE_CATALOG)} feeds total):")
        print("-" * 75)
        for feed in SOURCE_CATALOG:
            status_tag = "[ENABLED]" if feed.enabled else "[DISABLED]"
            print(f"{status_tag:11} {feed.id:20} | {feed.name[:28]:28} | Score: {feed.authority_score:3} | Group: {feed.catalog_group}")
        print("-" * 75)
        print("Checking active feeds connectivity...")
        health = check_feed_health(settings=settings)
        ok_count = sum(1 for h in health if h["status"] == "ok")
        print(f"Network health check: {ok_count}/{len(health)} feeds OK.")
        return 0

    if command == "schedule":
        settings = load_settings()
        if args.collection_time:
            settings.daily_collection_time = args.collection_time
        scheduler = DailyScheduler(settings=settings)
        print(f"Starting AI Newsletter Daily Scheduler at {settings.daily_collection_time} ({settings.newsletter_timezone})...")
        try:
            scheduler.run_loop()
        except KeyboardInterrupt:
            print("Scheduler stopped by user.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())


