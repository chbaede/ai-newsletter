from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import httpx

from .models import Article
from .prompts import SUMMARIZATION_SYSTEM_PROMPT, build_user_prompt
from .scoring import compute_multi_dimensional_scores
from .taxonomy import (
    CATEGORY_LABELS_EN,
    CATEGORY_LABELS_KO,
    classify_taxonomy,
)

logger = logging.getLogger(__name__)

HANGUL_RE = re.compile(r"[가-힣]")


@dataclass(slots=True)
class SummaryResult:
    title_ko: str
    title_en: str
    summary_ko: str
    summary_en: str
    why_it_matters_ko: str
    why_it_matters_en: str
    key_points: list[str] = field(default_factory=list)
    summary_model: str = "template"
    summary_version: str = "v1"
    summary_created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BaseSummarizer(ABC):
    @abstractmethod
    def summarize(
        self, article: Article, content: str = "", is_short_excerpt: bool = False
    ) -> SummaryResult:
        """Produce factual summary result for article."""
        pass


class TemplateSummarizer(BaseSummarizer):
    def summarize(
        self, article: Article, content: str = "", is_short_excerpt: bool = False
    ) -> SummaryResult:
        cat_ko = CATEGORY_LABELS_KO.get(article.category, "인공지능 산업")
        cat_en = CATEGORY_LABELS_EN.get(article.category, "Artificial Intelligence")

        # Titles
        is_hangul = bool(HANGUL_RE.search(article.title))
        title_ko = article.title if is_hangul else f"[AI 동향] {article.title}"
        title_en = article.title

        # Summaries
        excerpt_clean = article.excerpt.strip() if article.excerpt else ""
        if is_hangul:
            summary_ko = excerpt_clean or f"{article.title} 관련 {cat_ko} 주요 동향입니다."
            summary_en = f"{article.title}. Updates in the {cat_en} sector reported by {article.publisher or article.source}."
        else:
            summary_ko = f"{article.publisher or article.source}에 따르면, {article.title} 내용이 발표되었습니다. {cat_ko} 분야의 주요 전개입니다."
            summary_en = excerpt_clean or f"{article.title}. Strategic updates in the {cat_en} ecosystem."

        why_ko = _impact_sentence_ko(article.category)
        why_en = _impact_sentence_en(article.category)

        entities = article.entities or []
        topics = article.topics or []

        key_points: list[str] = []
        if entities:
            key_points.append(f"주요 기업/모델 (Entities): {', '.join(entities[:4])}")
        elif article.source:
            key_points.append(f"출처 (Source): {article.source}")

        if topics:
            key_points.append(f"핵심 분야 (Topics): {', '.join(topics[:4])}")
        else:
            key_points.append(f"카테고리: {cat_ko} ({cat_en})")

        key_points.append(f"발행처: {article.publisher or article.source or 'Global Media'}")

        return SummaryResult(
            title_ko=title_ko,
            title_en=title_en,
            summary_ko=summary_ko,
            summary_en=summary_en,
            why_it_matters_ko=why_ko,
            why_it_matters_en=why_en,
            key_points=key_points,
            summary_model="template",
            summary_version="v1",
        )


class OllamaSummarizer(BaseSummarizer):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def summarize(
        self, article: Article, content: str = "", is_short_excerpt: bool = False
    ) -> SummaryResult:
        text_body = content or article.content or article.excerpt
        user_prompt = build_user_prompt(
            title=article.title,
            source=article.publisher or article.source,
            content=text_body,
            excerpt=article.excerpt,
            is_short_excerpt=is_short_excerpt,
        )

        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": SUMMARIZATION_SYSTEM_PROMPT,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.1,
            },
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()

        raw_response = data.get("response", "")
        parsed = json.loads(raw_response)

        title_ko = str(parsed.get("title_ko", "")).strip() or article.title
        title_en = str(parsed.get("title_en", "")).strip() or article.title
        summary_ko = str(parsed.get("summary_ko", "")).strip()
        summary_en = str(parsed.get("summary_en", "")).strip()
        why_it_matters_ko = str(parsed.get("why_it_matters_ko", "")).strip()
        why_it_matters_en = str(parsed.get("why_it_matters_en", "")).strip()
        raw_points = parsed.get("key_points", [])
        key_points = [str(pt).strip() for pt in raw_points if str(pt).strip()]

        if not summary_ko or not summary_en:
            raise ValueError("LLM response missing required summary fields")

        return SummaryResult(
            title_ko=title_ko,
            title_en=title_en,
            summary_ko=summary_ko,
            summary_en=summary_en,
            why_it_matters_ko=why_it_matters_ko,
            why_it_matters_en=why_it_matters_en,
            key_points=key_points,
            summary_model=f"ollama:{self.model}",
            summary_version="v1",
        )


class FallbackSummarizer(BaseSummarizer):
    def __init__(self, primary: BaseSummarizer, fallback: BaseSummarizer):
        self.primary = primary
        self.fallback = fallback

    def summarize(
        self, article: Article, content: str = "", is_short_excerpt: bool = False
    ) -> SummaryResult:
        try:
            return self.primary.summarize(
                article, content=content, is_short_excerpt=is_short_excerpt
            )
        except Exception as exc:
            logger.warning("Primary summarizer failed, falling back: %s", exc)
            return self.fallback.summarize(
                article, content=content, is_short_excerpt=is_short_excerpt
            )


def get_summarizer(settings: object = None) -> BaseSummarizer:
    if settings is None:
        from .config import load_settings
        settings = load_settings()

    template = TemplateSummarizer()
    if getattr(settings, "enable_ai_summary", False):
        ollama = OllamaSummarizer(
            base_url=getattr(settings, "ollama_base_url", "http://localhost:11434"),
            model=getattr(settings, "ollama_model", "llama3.2"),
            timeout=getattr(settings, "ollama_timeout", 30.0),
        )
        return FallbackSummarizer(primary=ollama, fallback=template)

    return template


def classify_article(
    article: Article,
    summarizer: BaseSummarizer | None = None,
    content: str = "",
    is_short_excerpt: bool = False,
) -> Article:
    tax = classify_taxonomy(
        title=article.title,
        excerpt=article.excerpt,
        source=article.source,
        source_type=article.source_type,
        bucket=article.category,
    )
    category = tax.primary_category
    primary_category = tax.primary_category
    secondary_categories = tax.secondary_categories
    topics = tax.topics
    entities = tax.entities

    tags = entities + [t for t in topics if t not in entities]

    tax_article = replace(
        article,
        category=category,
        primary_category=primary_category,
        secondary_categories=secondary_categories,
        tags=tags,
        topics=topics,
        entities=entities,
        content=content or article.content,
    )

    if summarizer is None:
        summarizer = TemplateSummarizer()

    summary_result = summarizer.summarize(
        tax_article, content=content, is_short_excerpt=is_short_excerpt
    )

    scored_article = replace(
        tax_article,
        title_ko=summary_result.title_ko or article.title,
        title_en=summary_result.title_en or article.title,
        summary_ko=summary_result.summary_ko,
        summary_en=summary_result.summary_en,
        why_it_matters_ko=summary_result.why_it_matters_ko,
        why_it_matters_en=summary_result.why_it_matters_en,
        key_points=summary_result.key_points,
        summary_model=summary_result.summary_model,
        summary_version=summary_result.summary_version,
        summary_created_at=summary_result.summary_created_at,
    )

    # Compute multi-dimensional scores
    scores = compute_multi_dimensional_scores(scored_article)
    scored_article = replace(
        scored_article,
        score=scores.priority_score,
        priority_score=scores.priority_score,
        source_score=scores.source_score,
        relevance_score=scores.relevance_score,
        impact_score=scores.impact_score,
        novelty_score=scores.novelty_score,
        recency_score=scores.recency_score,
        evidence_quality_score=scores.evidence_quality_score,
        evidence_explanation=scores.explanation.summary_ko,
    )
    return scored_article



def summarize_article(article: Article) -> str:
    """Helper returning concise Korean summary."""
    if article.summary_ko:
        return article.summary_ko
    return TemplateSummarizer().summarize(article).summary_ko


def _impact_sentence_ko(category: str) -> str:
    impacts = {
        "frontier_models": "차세대 파운데이션 모델 경쟁과 추론 능력 향상이 산업 전반의 생성형 AI 도입 속도를 가속합니다.",
        "agents_coding": "자율 AI 에이전트와 소프트웨어 엔지니어링 도구의 발전으로 개발 생산성 및 워크플로 혁신이 본격화됩니다.",
        "hardware_infra": "초고속 AI 가속기 및 HBM 메모리 수급, 데이터센터 전력 확보가 AI 인프라 확장의 핵심 병목으로 부상하고 있습니다.",
        "open_source": "고성능 오픈 가중치 모델과 경량화 추론 엔진의 확산으로 온프레미스 및 커스텀 AI 구축 비용이 대폭 절감됩니다.",
        "research_breakthroughs": "기초 연구와 새로운 모델 아키텍처는 향후 1~2년 내 상용 AI 제품의 성능 상한을 결정짓는 핵심 지표입니다.",
        "multimodal_media": "영상, 음성, 3D 등 멀티모달 생성 기술의 고도화가 콘텐츠 제작 및 디지털 인터랙션 패러다임을 전환합니다.",
        "enterprise_app": "단순 챗봇을 넘어 실제 엔터프라이즈 업무 자동화와 고부가가치 비즈니스 프로세스 통합이 가속화됩니다.",
        "regulation_policy": "EU AI Act 및 각국 안전 프레임워크 시행에 따라 컴플라이언스 준수와 모델 거버넌스가 기업의 핵심 과제로 부각됩니다.",
        "industry_vc": "빅테크 간 파트너십 재편과 스타트업 펀딩 흐름은 차세대 AI 밸류체인의 헤게모니를 결정합니다.",
        "conference": "글로벌 AI 학회 및 빅테크 개발자 컨퍼런스는 최신 기술 로드맵과 차세대 전략이 공개되는 무대입니다.",
    }
    return impacts.get(category, "인공지능 생태계와 글로벌 산업 전반에 걸쳐 유의미한 기술적·사업적 파급효과가 예상됩니다.")


def _impact_sentence_en(category: str) -> str:
    impacts = {
        "frontier_models": "Next-gen foundation model competition and reasoning advances accelerate enterprise generative AI adoption.",
        "agents_coding": "Progress in autonomous agents and SWE tools transforms software development productivity and workflows.",
        "hardware_infra": "Next-gen accelerators, HBM memory supply, and datacenter power grids remain critical infrastructure bottlenecks.",
        "open_source": "High-capability open weights and optimized inference runtimes significantly lower the barrier to custom AI deployment.",
        "research_breakthroughs": "Fundamental architectural breakthroughs and benchmark achievements set the trajectory for commercial AI systems.",
        "multimodal_media": "Advanced multimodal video, audio, and visual generation reshapes content creation and human-computer interfaces.",
        "enterprise_app": "Adoption shifts from general chatbots to deep workflow automation and mission-critical enterprise integrations.",
        "regulation_policy": "Enforcement of the EU AI Act and global safety standards makes compliance and model governance a top priority.",
        "industry_vc": "Strategic tech partnerships and venture funding dynamics redefine the market structure and value chain in AI.",
        "conference": "Global academic venues and developer keynotes offer crucial visibility into forthcoming model architectures and roadmaps.",
    }
    return impacts.get(category, "Carries meaningful technological and strategic implications across the global AI ecosystem.")
