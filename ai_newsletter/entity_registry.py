"""Centralized AI Entity Registry and Disambiguation.

This module is the single source of truth for canonical AI organizations,
frontier labs, chipmakers, open-source frameworks, parent group hierarchies,
and entity disambiguation across taxonomy classification and event clustering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Article


@dataclass(frozen=True, slots=True)
class EntityDefinition:
    """Definition of a canonical AI industry entity."""

    canonical_name: str  # e.g. "OpenAI", "Anthropic", "NVIDIA"
    canonical_id: str  # e.g. "openai", "anthropic", "nvidia"
    entity_type: str  # "frontier_lab", "hardware", "infra", "tools_agent", "multimodal", "korean_ai", "research", "regulator"
    parent_group: str | None = None  # e.g. "Microsoft", "Alphabet", "Meta"
    aliases: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Canonical Entity Definitions
# ---------------------------------------------------------------------------

ENTITY_DEFINITIONS: list[EntityDefinition] = [
    # --- Frontier Labs & Big Tech ---
    EntityDefinition("OpenAI", "openai", "frontier_lab", "Microsoft Ecosystem", ["openai", "open ai", "오픈ai", "오픈에이아이", "chatgpt", "gpt-4", "gpt-4o", "gpt-5", "o1-preview", "o1", "o3", "sora", "sam altman", "알트만"]),
    EntityDefinition("Anthropic", "anthropic", "frontier_lab", None, ["anthropic", "앤트로픽", "claude", "클로드", "claude 3", "claude 3.5", "sonnet", "haiku", "opus", "dario amodei"]),
    EntityDefinition("Google DeepMind", "google_deepmind", "frontier_lab", "Alphabet", ["google deepmind", "deepmind", "구글 딥마인드", "딥마인드", "demis hassabis", "gemini", "제미나이", "gemma", "구글 ai", "google ai"]),
    EntityDefinition("Meta AI", "meta_ai", "frontier_lab", "Meta", ["meta ai", "메타 ai", "llama", "라마", "llama 3", "llama 3.1", "llama 3.3", "fair", "yann lecun", "zuckerberg", "저커버그"]),
    EntityDefinition("Microsoft", "microsoft", "frontier_lab", "Microsoft", ["microsoft", "마이크로소프트", "ms", "copilot", "코파일럿", "azure ai", "satya nadella", "나델라"]),
    EntityDefinition("Mistral AI", "mistral", "frontier_lab", None, ["mistral ai", "mistral", "미스트랄", "mixtral", "le chat", "arthur mensch"]),
    EntityDefinition("xAI", "xai", "frontier_lab", None, ["xai", "x.ai", "grok", "grok 2", "grok 3", "elon musk", "일론 머스크"]),
    EntityDefinition("Apple", "apple", "frontier_lab", "Apple", ["apple", "애플", "apple intelligence", "애플 인텔리전스", "siri"]),
    EntityDefinition("Amazon AWS", "amazon_aws", "infra", "Amazon", ["amazon", "aws", "아마존", "bedrock", "베드록", "trainium", "inferentia"]),

    # --- Hardware, Compute & Semiconductors ---
    EntityDefinition("NVIDIA", "nvidia", "hardware", "NVIDIA", ["nvidia", "엔비디아", "jensen huang", "젠슨 황", "blackwell", "블랙웰", "b200", "h100", "h200", "cuda", "쿠다", "dgx"]),
    EntityDefinition("AMD", "amd", "hardware", "AMD", ["amd", "lisa su", "리사 수", "mi300", "mi325", "mi350", "rocm"]),
    EntityDefinition("Intel", "intel", "hardware", "Intel", ["intel", "인텔", "gaudi", "gaudi 3"]),
    EntityDefinition("Qualcomm", "qualcomm", "hardware", "Qualcomm", ["qualcomm", "퀄컴", "snapdragon x", "스냅드래곤"]),
    EntityDefinition("TSMC", "tsmc", "hardware", "TSMC", ["tsmc", "대만 tsmc", "파운드리"]),
    EntityDefinition("SK Hynix", "sk_hynix", "hardware", "SK Group", ["sk hynix", "sk하이닉스", "하이닉스", "hbm", "hbm3e", "hbm4"]),
    EntityDefinition("Samsung Electronics", "samsung", "hardware", "Samsung Group", ["samsung electronics", "samsung", "삼성전자", "삼성", "exynos", "hbm3e"]),
    EntityDefinition("ARM", "arm", "hardware", "SoftBank", ["arm", "arm holdings", "암"]),
    EntityDefinition("Groq", "groq", "hardware", None, ["groq", "lpu", "groqchip"]),
    EntityDefinition("Cerebras", "cerebras", "hardware", None, ["cerebras", "cs-3", "wafer scale engine"]),
    EntityDefinition("Tenstorrent", "tenstorrent", "hardware", None, ["tenstorrent", "텐스토렌트", "jim keller", "짐 켈러"]),
    EntityDefinition("FuriosaAI", "furiosa", "hardware", None, ["furiosaai", "furiosa", "퓨리오사ai", "renegade", "warboy"]),
    EntityDefinition("Rebellions", "rebellions", "hardware", None, ["rebellions", "리벨리온", "atom", "rebel"]),

    # --- Cloud & AI Infrastructure ---
    EntityDefinition("CoreWeave", "coreweave", "infra", None, ["coreweave", "코어위브"]),
    EntityDefinition("Lambda Labs", "lambda_labs", "infra", None, ["lambda labs", "lambda"]),
    EntityDefinition("RunPod", "runpod", "infra", None, ["runpod"]),
    EntityDefinition("Together AI", "together_ai", "infra", None, ["together ai", "together.ai"]),

    # --- AI Tools, Agents & Open-Source ---
    EntityDefinition("Hugging Face", "huggingface", "tools_agent", None, ["hugging face", "huggingface", "허깅페이스", "transformers", "tgi", "hf"]),
    EntityDefinition("Cursor", "cursor", "tools_agent", "Anysphere", ["cursor", "cursor ai", "커서", "anysphere"]),
    EntityDefinition("Cognition", "cognition", "tools_agent", None, ["cognition", "devin", "데빈"]),
    EntityDefinition("Windsurf", "windsurf", "tools_agent", "Codeium", ["windsurf", "codeium", "윈드서프"]),
    EntityDefinition("GitHub Copilot", "github_copilot", "tools_agent", "Microsoft", ["github copilot", "깃허브 코파일럿"]),
    EntityDefinition("LangChain", "langchain", "tools_agent", None, ["langchain", "랭체인", "langgraph"]),
    EntityDefinition("LlamaIndex", "llamaindex", "tools_agent", None, ["llamaindex", "라마인덱스"]),
    EntityDefinition("vLLM", "vllm", "tools_agent", None, ["vllm"]),
    EntityDefinition("Ollama", "ollama", "tools_agent", None, ["ollama", "올라마"]),
    EntityDefinition("Replit", "replit", "tools_agent", None, ["replit", "replit agent"]),
    EntityDefinition("Poolside", "poolside", "tools_agent", None, ["poolside", "poolside ai"]),

    # --- Generative Multimodal & Media ---
    EntityDefinition("Runway", "runway", "multimodal", None, ["runway", "런웨이", "gen-2", "gen-3", "gen-3 alpha"]),
    EntityDefinition("Midjourney", "midjourney", "multimodal", None, ["midjourney", "미드저니", "david holz"]),
    EntityDefinition("Stability AI", "stability_ai", "multimodal", None, ["stability ai", "stable diffusion", "스테이블 디퓨전", "sd3"]),
    EntityDefinition("ElevenLabs", "elevenlabs", "multimodal", None, ["elevenlabs", "일레븐랩스", "voice ai"]),
    EntityDefinition("Perplexity", "perplexity", "multimodal", None, ["perplexity", "퍼플렉시티", "aravind srinivas"]),
    EntityDefinition("Kling AI", "kling_ai", "multimodal", "Kuaishou", ["kling", "kling ai", "클링"]),
    EntityDefinition("Pika", "pika", "multimodal", None, ["pika", "pika labs"]),
    EntityDefinition("Suno", "suno", "multimodal", None, ["suno", "suno ai"]),
    EntityDefinition("Udio", "udio", "multimodal", None, ["udio", "udio ai"]),

    # --- Korean AI Ecosystem ---
    EntityDefinition("Naver", "naver", "korean_ai", "Naver", ["naver", "네이버", "hyperclova", "hyperclova x", "하이퍼클로바x", "하이퍼클로바"]),
    EntityDefinition("Kakao", "kakao", "korean_ai", "Kakao", ["kakao", "카카오", "kanana", "카나나"]),
    EntityDefinition("LG AI Research", "lg_ai", "korean_ai", "LG Group", ["lg ai research", "lg ai연구원", "lg ai", "exaone", "엑사원"]),
    EntityDefinition("Upstage", "upstage", "korean_ai", None, ["upstage", "업스테이지", "solar", "솔라"]),
    EntityDefinition("Wrtn", "wrtn", "korean_ai", None, ["wrtn", "뤼튼", "뤼튼테크놀로지스"]),
    EntityDefinition("Twelve Labs", "twelve_labs", "korean_ai", None, ["twelve labs", "트웰브랩스"]),

    # --- Academic, Research & Standards ---
    EntityDefinition("arXiv", "arxiv", "research", None, ["arxiv", "아카이브"]),
    EntityDefinition("Stanford HAI", "stanford_hai", "research", "Stanford University", ["stanford hai", "스탠퍼드 hai", "stanford university"]),
    EntityDefinition("MIT CSAIL", "mit_csail", "research", "MIT", ["mit csail", "csail", "mit"]),
    EntityDefinition("Berkeley BAIR", "berkeley_bair", "research", "UC Berkeley", ["berkeley bair", "bair", "uc berkeley"]),

    # --- Regulatory & Policy Bodies ---
    EntityDefinition("EU AI Office", "eu_ai_office", "regulator", "European Union", ["eu ai office", "eu ai act", "유럽 ai 법", "european commission", "유럽연합"]),
    EntityDefinition("US NIST AISI", "us_nist_aisi", "regulator", "US Government", ["nist", "aisi", "ai safety institute", "인공지능 안전 연구소", "ftc", "white house"]),
    EntityDefinition("MSIT Korea", "msit_korea", "regulator", "Republic of Korea", ["과학기술정보통신부", "과기정통부", "msit", "개인정보보호위원회", "개보위", "ai기본법"]),
]

_ENTITY_BY_ID: dict[str, EntityDefinition] = {e.canonical_id: e for e in ENTITY_DEFINITIONS}
_ALIAS_TO_CANONICAL: dict[str, EntityDefinition] = {}

for entity in ENTITY_DEFINITIONS:
    _ALIAS_TO_CANONICAL[entity.canonical_name.lower()] = entity
    _ALIAS_TO_CANONICAL[entity.canonical_id.lower()] = entity
    for alias in entity.aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = entity


def get_entity(canonical_id: str) -> EntityDefinition | None:
    return _ENTITY_BY_ID.get(canonical_id)


def resolve_canonical_entity(name_or_alias: str) -> EntityDefinition | None:
    cleaned = name_or_alias.strip().lower()
    return _ALIAS_TO_CANONICAL.get(cleaned)


def find_entities_in_text(text: str) -> list[str]:
    """Find all matching canonical entity names mentioned in the text."""
    if not text:
        return []
    
    text_lower = f" {text.lower()} "
    found_entities: set[str] = set()

    for entity in ENTITY_DEFINITIONS:
        # Check canonical name
        pattern = rf"\b{re.escape(entity.canonical_name.lower())}\b"
        if re.search(pattern, text_lower):
            found_entities.add(entity.canonical_name)
            continue

        # Check aliases
        for alias in entity.aliases:
            # For Korean aliases or short terms, check boundaries or inclusion
            if any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in alias):
                if alias in text_lower:
                    found_entities.add(entity.canonical_name)
                    break
            else:
                pat = rf"\b{re.escape(alias)}\b"
                if re.search(pat, text_lower):
                    found_entities.add(entity.canonical_name)
                    break

    return sorted(found_entities)


def is_same_entity(name1: str, name2: str) -> bool:
    e1 = resolve_canonical_entity(name1)
    e2 = resolve_canonical_entity(name2)
    if e1 and e2:
        return e1.canonical_id == e2.canonical_id
    return name1.strip().lower() == name2.strip().lower()


def is_same_parent_group(name1: str, name2: str) -> bool:
    e1 = resolve_canonical_entity(name1)
    e2 = resolve_canonical_entity(name2)
    if e1 and e2 and e1.parent_group and e2.parent_group:
        return e1.parent_group.lower() == e2.parent_group.lower()
    return False


def get_entity_family(name: str) -> set[str]:
    entity = resolve_canonical_entity(name)
    if not entity:
        return {name}
    family = {entity.canonical_name}
    if entity.parent_group:
        for other in ENTITY_DEFINITIONS:
            if other.parent_group == entity.parent_group:
                family.add(other.canonical_name)
    return family

