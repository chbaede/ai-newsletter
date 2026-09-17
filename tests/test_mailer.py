import pytest

from ai_newsletter.config import Settings
from ai_newsletter.mailer import (
    MailConfigError,
    build_email_html,
    build_email_text,
    send_issue,
)
from ai_newsletter.models import Article, NewsletterIssue


def test_build_email_bilingual():
    article = Article(
        title="OpenAI announces GPT-4o",
        title_ko="오픈AI, GPT-4o 전격 공개",
        title_en="OpenAI Unveils GPT-4o",
        url="https://openai.com",
        source="OpenAI",
        category="frontier_models",
        summary_ko="새로운 플래그십 멀티모달 모델입니다.",
        summary_en="New flagship multimodal model released.",
        why_it_matters_ko="실시간 음성 및 영상 반응성이 비약적으로 향상되었습니다.",
        why_it_matters_en="Significantly enhances real-time voice and vision responsiveness.",
        priority_score=90.0,
    )
    issue = NewsletterIssue(issue_date="2026-09-17", articles=[article])

    html_ko = build_email_html(issue, lang="ko")
    assert "AI 산업 핵심 인텔리전스" in html_ko
    assert "오픈AI, GPT-4o 전격 공개" in html_ko
    assert "새로운 플래그십 멀티모달 모델입니다." in html_ko

    html_en = build_email_html(issue, lang="en")
    assert "AI Industry & Frontier Briefing" in html_en
    assert "OpenAI Unveils GPT-4o" in html_en
    assert "New flagship multimodal model released." in html_en

    text_ko = build_email_text(issue, lang="ko")
    assert "총 브리핑" in text_ko

    text_en = build_email_text(issue, lang="en")
    assert "Total News" in text_en


def test_send_issue_missing_config():
    issue = NewsletterIssue(issue_date="2026-09-17", articles=[])
    settings = Settings(smtp_host=None, smtp_from=None, newsletter_to=None)
    with pytest.raises(MailConfigError):
        send_issue(issue, settings=settings)

