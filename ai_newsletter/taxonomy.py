from __future__ import annotations

import re
from dataclasses import dataclass, field

from .entity_registry import find_entities_in_text

PRIMARY_CATEGORIES = [
    "big",
    "frontier_models",
    "agents_coding",
    "hardware_infra",
    "open_source",
    "research_breakthroughs",
    "multimodal_media",
    "enterprise_app",
    "regulation_policy",
    "industry_vc",
    "conference",
    "reference",
]

CATEGORY_LABELS_KO = {
    "big": "주요 AI 뉴스",
    "frontier_models": "프론티어 모델 & LLM",
    "agents_coding": "AI 에이전트 & 코딩 도구",
    "hardware_infra": "AI 반도체 & 인프라",
    "open_source": "오픈소스 AI & 가중치",
    "research_breakthroughs": "AI 연구 & 브레이크스루",
    "multimodal_media": "멀티모달 & 생성 미디어",
    "enterprise_app": "엔터프라이즈 & 산업 응용",
    "regulation_policy": "규제, 정책 & 안전",
    "industry_vc": "투자, 시장 & 스타트업",
    "conference": "학회 & 컨퍼런스",
    "reference": "참고자료 / 기타",
}

CATEGORY_LABELS_EN = {
    "big": "Top AI Stories",
    "frontier_models": "Frontier Models & LLMs",
    "agents_coding": "AI Agents & Coding Tools",
    "hardware_infra": "AI Chips & Infrastructure",
    "open_source": "Open Source & Weights",
    "research_breakthroughs": "Research & Breakthroughs",
    "multimodal_media": "Multimodal & Generative Media",
    "enterprise_app": "Enterprise & Applications",
    "regulation_policy": "Policy, Safety & Governance",
    "industry_vc": "Industry, VC & Business",
    "conference": "Conferences & Events",
    "reference": "Reference & Resources",
}

TOPIC_PATTERNS = {
    "LLM": [r"\bllm\b", r"\blarge language model\b", r"언어모델", r"초거대 ai"],
    "Reasoning": [r"\breasoning\b", r"\bo1\b", r"\bo3\b", r"\bchain of thought\b", r"\bcot\b", r"추론 모델"],
    "Agents": [r"\bagent\b", r"\bagents\b", r"\bautonomous agent\b", r"\bmulti-agent\b", r"에이전트"],
    "Coding Assistant": [r"\bcoding\b", r"\bcode generation\b", r"\bswe\b", r"\bcursor\b", r"\bdevin\b", r"\bcopilot\b", r"코딩"],
    "GPU/Semiconductor": [r"\bgpu\b", r"\bblackwell\b", r"\bh100\b", r"\bb200\b", r"\bhbm\b", r"\bhbm3e\b", r"\bcuda\b", r"반도체", r"엔비디아"],
    "NPU/ASIC": [r"\bnpu\b", r"\basic\b", r"\blpu\b", r"\btpu\b", r"\bgroq\b", r"신경망처리장치"],
    "Datacenter & Energy": [r"\bdatacenters?\b", r"\bdata centers?\b", r"\bpower grid\b", r"\bnuclear\b", r"데이터센터", r"전력"],
    "Open Weights": [r"\bopen weights\b", r"\bopen source\b", r"\bllama\b", r"\bgemma\b", r"\bmistral\b", r"오픈소스", r"가중치"],
    "Fine-tuning & RL": [r"\bfine-tuning\b", r"\bfinetuning\b", r"\brlhf\b", r"\bdpo\b", r"\brlaif\b", r"파인튜닝", r"강화학습"],
    "Inference Engine": [r"\bvllm\b", r"\bollama\b", r"\btensorrt\b", r"\btgi\b", r"추론 엔진"],
    "Multimodal": [r"\bmultimodal\b", r"\bvision\b", r"\bvideo gen\b", r"\bsora\b", r"\brunway\b", r"멀티모달", r"영상 생성"],
    "Voice/Audio": [r"\bvoice ai\b", r"\baudio\b", r"\belevenlabs\b", r"\bspeech-to-text\b", r"\btts\b", r"음성 ai"],
    "AI Safety & Alignment": [r"\bsafety\b", r"\balignment\b", r"\bred teaming\b", r"\bguardrails\b", r"안전", r"얼라인먼트"],
    "AI Regulation": [r"\beu ai act\b", r"\bcopyright\b", r"\bregulation\b", r"\bpolicy\b", r"ai법", r"규제", r"저작권"],
    "Enterprise AI": [r"\benterprise\b", r"\bsaas\b", r"\bworkflow\b", r"\bproductivity\b", r"기업용 ai"],
    "Robotics & Embodied": [r"\brobotics\b", r"\bhumanoid\b", r"\bembodied ai\b", r"로봇", r"휴머노이드"],
    "Healthcare & Bio": [r"\bbiotech\b", r"\bdrug discovery\b", r"\balphafold\b", r"\bhealth\b", r"바이오", r"신약"],
    "Funding & M&A": [r"\bfunding\b", r"\bvaluation\b", r"\bseries [a-z]\b", r"\bacquisition\b", r"\binvestment\b", r"투자", r"유치", r"인수"],
}

CONFERENCE_NAMES = [
    "NeurIPS",
    "ICML",
    "ICLR",
    "CVPR",
    "ACL",
    "EMNLP",
    "Google I/O",
    "NVIDIA GTC",
    "Apple WWDC",
    "Microsoft Build",
    "AWS re:Invent",
]


@dataclass(slots=True)
class TaxonomyResult:
    primary_category: str
    secondary_categories: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)


def extract_topics(text: str) -> list[str]:
    text_lower = text.lower()
    matched = []
    for topic, patterns in TOPIC_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                matched.append(topic)
                break
    return matched


def classify_taxonomy(
    title: str,
    excerpt: str = "",
    source: str = "",
    source_type: str = "media",
    bucket: str = "frontier_models",
) -> TaxonomyResult:
    full_text = f"{title} {excerpt} {source}"
    entities = find_entities_in_text(full_text)
    topics = extract_topics(full_text)
    lower = full_text.lower()

    # Rule-based primary category inference
    primary: str = bucket if bucket in PRIMARY_CATEGORIES else "frontier_models"
    secondary: list[str] = []

    # Check conferences
    if any(conf.lower() in lower for conf in CONFERENCE_NAMES):
        secondary.append("conference")
        if bucket == "conference" or any(kw in lower for kw in ["call for papers", "deadline", "conference", "학회 개최"]):
            primary = "conference"

    # Check regulation & policy
    if any(re.search(pat, lower) for pat in [r"eu ai act", r"regulation", r"regulatory", r"ftc", r"copyright", r"저작권", r"규제", r"과기정통부", r"ai기본법"]):
        if primary != "regulation_policy":
            secondary.append("regulation_policy")
        if source_type == "regulator" or any(kw in lower for kw in ["법안", "규제안", "lawsuit", "antitrust", "소송"]):
            primary = "regulation_policy"

    # Check hardware & infra
    elif any(re.search(pat, lower) for pat in [r"\bgpu\b", r"\bh100\b", r"\bb200\b", r"\bblackwell\b", r"\bhbm\b", r"datacenter", r"tsmc", r"반도체", r"엔비디아 실적"]):
        if primary != "hardware_infra":
            secondary.append("hardware_infra")
        if any(h_ent in entities for h_ent in ["NVIDIA", "AMD", "Intel", "TSMC", "SK Hynix", "Samsung Electronics", "Groq"]):
            primary = "hardware_infra"

    # Check agents & coding
    elif any(re.search(pat, lower) for pat in [r"\bagent\b", r"\bcoding\b", r"\bcursor\b", r"\bdevin\b", r"\bcopilot\b", r"swe-bench", r"에이전트"]):
        if primary != "agents_coding":
            secondary.append("agents_coding")
        if any(t in topics for t in ["Agents", "Coding Assistant"]):
            primary = "agents_coding"

    # Check multimodal & media
    elif any(re.search(pat, lower) for pat in [r"\bvideo gen\b", r"\bsora\b", r"\brunway\b", r"\bmidjourney\b", r"\belevenlabs\b", r"동영상 생성", r"음성 ai"]):
        if primary != "multimodal_media":
            secondary.append("multimodal_media")
        if any(m_ent in entities for m_ent in ["Runway", "Midjourney", "ElevenLabs", "Kling AI", "Pika", "Suno"]):
            primary = "multimodal_media"

    # Check open source
    elif any(re.search(pat, lower) for pat in [r"\bhugging face\b", r"\bhuggingface\b", r"\bopen weights\b", r"\bopen-source\b", r"\bvllm\b", r"\bollama\b", r"오픈소스 모델"]):
        if primary != "open_source":
            secondary.append("open_source")
        if "Hugging Face" in entities or any(t in topics for t in ["Open Weights", "Inference Engine"]):
            primary = "open_source"

    # Check research / breakthroughs
    elif any(re.search(pat, lower) for pat in [r"\barxiv\b", r"\bpaper\b", r"\bbreakthrough\b", r"\bbenchmark\b", r"논문", r"벤치마크", r"학술"]):
        if primary != "research_breakthroughs":
            secondary.append("research_breakthroughs")
        if source_type == "research" or "arXiv" in entities:
            primary = "research_breakthroughs"

    # Check funding / VC
    elif any(re.search(pat, lower) for pat in [r"\bvaluation\b", r"\braises\b", r"\bseries [a-z]\b", r"\bipo\b", r"투자 유치", r"기업가치"]):
        if primary != "industry_vc":
            secondary.append("industry_vc")
        if "Funding & M&A" in topics:
            primary = "industry_vc"

    # Frontier models check
    elif any(ent in entities for ent in ["OpenAI", "Anthropic", "Google DeepMind", "Meta AI", "Mistral AI", "xAI"]):
        primary = "frontier_models"

    # Fallback to bucket if specified and valid
    if primary not in PRIMARY_CATEGORIES:
        primary = "frontier_models"

    return TaxonomyResult(
        primary_category=primary,
        secondary_categories=list(dict.fromkeys(secondary)),
        topics=topics,
        entities=entities,
    )
