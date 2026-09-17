# AI Newsletter & Intelligence Briefing (인공지능 뉴스레터 & 인텔리전스 브리핑)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-46%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AI Newsletter & Intelligence Briefing** is an automated, production-ready AI news curation, multi-dimensional scoring, event clustering, bilingual web dashboard, and email intelligence platform.

Designed as an AI-specialized evolution of [automotive-newsletter](https://github.com/chbaede/automotive-newsletter), it delivers comprehensive, factual, and noise-filtered intelligence across the global AI ecosystem with **first-class bilingual (Korean & English)** support.

---

## Key Highlights (주요 특징)

### 1. First-Class Bilingual Support (완벽한 한/영 이중 언어 지원)
- **Data Models & Summaries**: Every article and clustered event stores bilingual titles, concise summaries, impact explanations (*Why It Matters*), and key takeaway points (`ko` / `en`).
- **Web Dashboard**: Instant one-click toggle between Korean (`KO`) and English (`EN`) across the navigation, categories, badges, modals, and content.
- **Email Delivery**: Independent dispatch support for Korean (`--lang ko`) and English (`--lang en`) editions with responsive, email-client-tested HTML & plaintext templates.

### 2. High-Authority Multi-Source Catalog (26 Curated AI Feeds)
- **Frontier Labs & Big Tech**: OpenAI, Anthropic, Google DeepMind, Meta AI, Microsoft AI, NVIDIA, Hugging Face.
- **Academic & Research**: arXiv CS.AI, arXiv CS.CL (NLP), Stanford HAI, MIT CSAIL.
- **Global Tech Media**: TechCrunch AI, VentureBeat AI, The Verge AI, MarkTechPost, MIT Technology Review.
- **Korean AI & Industry Media**: AI타임스 (AITimes), ZDNet Korea AI, 테크M, 전자신문 AI, GeekNews AI.
- **AI Safety & Policy Regulators**: European AI Office (EU AI Act), US NIST AI Safety Institute (AISI), 대한민국 과학기술정보통신부 (MSIT AI 기본법).
- **Aggregators**: Google News Global & Korean AI streams.
- **Fast Concurrent Fetching**: Multithreaded asynchronous collection retrieves all 26 feeds in ~2 seconds.

### 3. AI Taxonomy & Canonical Entity Registry (분류 체계 및 엔티티 레지스트리)
- **10 Primary AI Categories**:
  - Frontier Models & LLMs (`frontier_models` / 프론티어 모델 & LLM)
  - AI Agents & Coding Tools (`agents_coding` / AI 에이전트 & 코딩)
  - AI Chips & Infrastructure (`hardware_infra` / AI 반도체 & 인프라)
  - Open Source & Weights (`open_source` / 오픈소스 & 가중치)
  - Research & Breakthroughs (`research_breakthroughs` / 연구 & 브레이크스루)
  - Multimodal & Media (`multimodal_media` / 멀티모달 & 미디어)
  - Enterprise & Applications (`enterprise_app` / 엔터프라이즈 & 응용)
  - Policy, Safety & Regulation (`regulation_policy` / 정책, 규제 & 안전)
  - Industry, VC & Business (`industry_vc` / 투자, 시장 & 스타트업)
  - Conferences & Events (`conference` / 학회 & 컨퍼런스)
- **80+ Canonical AI Entities & Aliases**: Canonical mapping for frontier labs, chipmakers, hyperscalers, Korean AI leaders, and startups.
- **Clustering Guards**: Intelligent separation guards prevent distinct model versions (e.g., GPT-4o vs Claude 3.5 Sonnet) or competitors from being improperly merged.

### 4. 5-Dimensional Scoring Engine (다차원 평가 엔진)
Calculates a balanced priority score ($0.0 - 100.0$) using weighted evaluation:
1. **Source Authority Score** ($30\%$): Publisher hierarchy (Regulators: 100, Labs/ArXiv: 95, Top Media: 85-90).
2. **AI Domain Relevance** ($25\%$): Precision scoring based on core AI keywords, entities, and categories.
3. **Impact Score** ($20\%$): Breakthroughs, regulatory enforcement, major funding, benchmarks, and production releases.
4. **Novelty Score** ($15\%$): Freshness against past 3-day coverage.
5. **Recency Score** ($10\%$): Publication recency within the 24-hour cycle.

Articles are prioritized into four operational tiers: **CRITICAL (긴급)**, **HIGH (주요)**, **MEDIUM (일반)**, and **WATCH (모니터링)**.

### 5. Multi-Engine Summarizer (다중 요약 엔진)
- **Template Summarizer**: Instant, zero-cost, zero-hallucination deterministic fallback based on structured title, topic, and entity extraction.
- **Ollama Local LLM Summarizer**: Optional plug-and-play local LLM (`llama3.2` or any local model via Ollama) producing structured JSON grounded strictly in source text.
- **Graceful Fallback**: Automatic fallback to the template engine if LLM service is offline or times out.

### 6. Modern Web Dashboard & Automated Delivery
- Built with **FastAPI** + **Jinja2** + modern responsive CSS (dark/light clean aesthetic).
- Live search bar, category filtering, region filters (Korea, US, Europe, Global), and source type filters.
- Modal views for detailed multi-source event clusters and original source links.
- On-demand admin triggers for collection and email sending.
- Background scheduler daemon with catch-up logic and atomic SQLite run locks.

---

## Project Architecture (시스템 아키텍처)

```text
ai-newsletter/
├── ai_newsletter/
│   ├── __init__.py
│   ├── __main__.py               # CLI entrypoint
│   ├── cli.py                    # Command-line interface
│   ├── config.py                 # Pydantic/dataclass settings & environment loader
│   ├── logging.py                # Structured JSON logging
│   ├── models.py                 # Core dataclasses (Article, Event, Issue, Metrics)
│   ├── entity_registry.py        # 80+ Canonical AI entities & alias mappings
│   ├── taxonomy.py               # 10 AI categories, subcategories & regex matchers
│   ├── sources.py                # 26 Curated AI source feeds & authority scores
│   ├── content_extractor.py      # Polite article text extractor & boilerplate cleaner
│   ├── prompts.py                # Strictly grounded bilingual summarization prompts
│   ├── summarizer.py             # Template, Ollama LLM, and Fallback summarizers
│   ├── scoring.py                # 5-dimensional scoring & weighting engine
│   ├── priority.py               # Priority tiers & executive summary generator
│   ├── clustering.py             # Event clustering with version & competitor guards
│   ├── collector.py              # Parallel feed fetcher, title deduplication, SQLite store
│   ├── store.py                  # SQLite storage & schema migrations
│   ├── presentation.py           # Presentation view models, localized labels & filters
│   ├── mailer.py                 # HTML/Text bilingual email generator & SMTP sender
│   ├── scheduler.py              # Background daemon with atomic execution locks
│   ├── web.py                    # FastAPI web dashboard & admin API
│   ├── templates/                # Jinja2 HTML templates
│   │   ├── base.html             # Layout with bilingual header, nav & modal
│   │   └── index.html            # Main dashboard view
│   └── static/                   # CSS and JavaScript assets
│       ├── styles.css            # Responsive typography & clean dark/light styles
│       └── app.js                # Instant KO/EN toggle, search, filtering & modals
├── tests/                        # Comprehensive test suite (46 passing tests)
├── Dockerfile                    # Production container image
├── deploy.sh                     # Automated deployment script
├── pyproject.toml                # Build system & package dependencies
├── requirements.txt              # Pinned requirements
└── README.md                     # Documentation
```

---

## Quickstart (빠른 시작)

### 1. Installation

```bash
# Clone repository
git clone https://github.com/chbaede/ai-newsletter.git
cd ai-newsletter

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### 2. Configuration

Copy the example environment file and adjust your settings:

```bash
cp .env.example .env
```

Key environment variables:
| Variable | Default | Description |
|---|---|---|
| `PORT` | `8001` | Web dashboard HTTP port (avoids conflict with automotive-newsletter on 8000) |
| `NEWSLETTER_TIMEZONE` | `Asia/Seoul` | Target timezone for daily issues |
| `DAILY_COLLECTION_TIME`| `07:00` | Automated daily collection schedule (HH:MM) |
| `DEFAULT_LANGUAGE` | `ko` | Default web UI and email language (`ko` or `en`) |
| `ENABLE_AI_SUMMARY` | `false` | Enable local Ollama LLM summarization |
| `OLLAMA_BASE_URL` | `http://localhost:11434`| Ollama server address |
| `VERIFY_TLS` | `true` | Set to `false` if behind corporate SSL inspection |
| `SMTP_HOST` | - | Outbound SMTP server address |
| `NEWSLETTER_TO` | - | Comma-separated list of subscriber emails |

### 3. CLI Commands

#### Feed Status & Network Health Check
Check all 26 configured sources and test network connectivity in parallel:
```bash
ai-newsletter sources
```

#### Collect Latest News
Collect articles, score them, cluster into events, and persist to SQLite:
```bash
ai-newsletter collect --force
```

#### Run Web Dashboard
Start the local FastAPI web dashboard (runs on port 8001 by default):
```bash
ai-newsletter serve --reload
```
Open [http://localhost:8001](http://localhost:8001) in your browser.

#### Send Newsletter via Email
Send the collected issue in Korean or English:
```bash
# Korean edition
ai-newsletter send --lang ko

# English edition
ai-newsletter send --lang en
```

#### Run Background Scheduler
Run the standalone daily collection daemon:
```bash
ai-newsletter schedule --time 07:00
```

---

## Running Tests (테스트 실행)

The test suite covers unit and integration tests across clustering, scoring, guards, feeds, extraction, web endpoints, database persistence, and email formatting:

```bash
pytest
```

Output:
```text
======================== 46 passed, 2 warnings in 0.30s ========================
```

---

## Docker Deployment (도커 배포)

Build and run using the production Dockerfile (exposed on port 8001):

```bash
# Build Docker image
docker build -t ai-newsletter:latest .

# Run container
docker run -d \
  -p 8001:8001 \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  --name ai-newsletter \
  ai-newsletter:latest
```

Or deploy using the automated deployment script:
```bash
chmod +x deploy.sh
./deploy.sh
```

---

## License (라이선스)

MIT License. See [LICENSE](LICENSE) for details.

