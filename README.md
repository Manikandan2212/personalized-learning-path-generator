# LearnPath AI — Agentic RAG-Based Personalized Learning System

> A production-grade multi-agent system that generates personalized learning roadmaps, answers questions via RAG, tracks progress, and runs adaptive quizzes — all orchestrated through a structured MCP (Model Context Protocol) context layer.

---

## Architecture Overview

```
User → Frontend (HTML/JS)
         ↓
   Flask REST API
         ↓
 ┌─────────────────────────────────────────┐
 │         Multi-Agent System              │
 │                                         │
 │  PlannerAgent   → roadmap generation    │
 │  ResourceAgent  → course/link fetching  │
 │  ValidationAgent→ guardrails I/O        │
 │  QuizAgent      → adaptive assessment   │
 │  ProgressAgent  → dashboard & SQLite    │
 └─────────────────────────────────────────┘
         ↓
   RAG Pipeline (TF-IDF Vector Store)
         ↓
 Knowledge Base (15 topics, chunked)
         ↓
   Observability Layer (SQLite metrics)
```

---

## How Each Evaluation Criterion Is Met

### 1. RAG (Retrieval Augmented Generation)

**File:** `rag/vector_store.py`, `rag/knowledge_base.py`, `rag/chatbot.py`

- Custom TF-IDF vector store (no external ML deps required)
- 15 knowledge base topics chunked and indexed at startup
- Every chat query retrieves top-k documents **before** generating an answer
- Retrieval results are injected into the MCP context and the response

```python
# rag/chatbot.py — retrieval happens before generation
retrieved = self.vector_store.search(sanitized_query, top_k=5)
mcp_context["retrieved_docs"] = retrieved         # → injected into MCP
raw_answer = rule_based_answer(query, retrieved, mcp_context)  # grounded answer
```

### 2. Agentic Framework

**Files:** `agents/planner_agent.py`, `agents/resource_agent.py`, `agents/validation_agent.py`, `agents/quiz_agent.py`, `agents/progress_agent.py`

All agents expose a standard `run(context: MCPContext)` interface and contain real decision logic:

| Agent | Decision Logic |
|---|---|
| **PlannerAgent** | Detects domain, adjusts path depth by inferred user level, filters known topics, falls back to RAG ordering |
| **ResourceAgent** | Fetches curated resources, computes free/paid split, finds related topics via RAG similarity |
| **ValidationAgent** | Blocks harmful input (regex), checks confidence threshold, validates roadmap structure, restricts uploads |
| **QuizAgent** | Shuffles options, evaluates answers server-side, adapts feedback by score band, falls back to RAG-generated questions |
| **ProgressAgent** | Persists to SQLite, builds MCP context from live progress data, computes overall dashboard metrics |

```python
# agents/planner_agent.py — real decision logic
def run(self, context: MCPContext) -> Dict:
    user_level = self._infer_user_level(context)   # decision: beginner/intermediate/advanced
    domain = self._detect_domain(context.user_goal) # decision: keyword → domain
    path_ids = self._apply_level_filter(            # decision: trim path by level
        LEARNING_PATHS[domain], user_level
    )
    path_ids = [p for p in path_ids              # decision: skip known topics
                if p not in context._known_topics]
    return self._build_steps(path_ids, context.user_goal)
```

### 3. MCP (Model Context Protocol)

**File:** `rag/mcp_context.py`

Structured context payload passed to every agent call and LLM prompt. Instead of passing raw strings, every agent receives a typed `MCPContext` object:

```python
MCPContext(
    user_goal    = "Learn Deep Learning",
    user_name    = "Priya",
    current_step = "Neural Networks",
    progress_pct = 40.0,
    completed_topics = 2,
    total_topics     = 8,
    average_score    = 75.0,
    retrieved_docs   = [...]   # injected at query time by RAG
)
```

The context also serializes to a **prompt block** injected into every LLM/RAG call:

```python
context.to_prompt_block()
# ==> === MCP STRUCTURED CONTEXT ===
#     User: Priya (id=user_123)
#     Goal: Learn Deep Learning
#     Current Step: Neural Networks
#     Progress: 40.0% (2/8 topics)
#     Retrieved Documents: [...]
```

### 4. Guardrails

**File:** `agents/validation_agent.py`

Three layers of protection:

**Input Guardrails** — applied before any agent runs:
```python
def validate_input(query):
    if len(query) < 2: return REJECT         # too short
    if harmful_pattern.match(query): return REJECT  # harmful content
    if len(query) > 1000: truncate(query)    # truncate long input
```

**Output Guardrails** — applied after generation:
```python
def validate_output(response, retrieved_docs, query):
    confidence = compute_confidence(response, retrieved_docs, query)
    if confidence < THRESHOLD:               # 0.05
        return fallback_message()            # no hallucination
```

**Upload Guardrails** — file type, extension, size checks:
```python
def validate_upload(filename, content_type, size_bytes):
    if content_type not in ALLOWED_TYPES: REJECT   # pdf/txt only
    if size_bytes > 5MB: REJECT                    # size limit
```

### 5. Observability

**File:** `observability/metrics.py`

Every agent call, RAG retrieval, and API request is logged to SQLite and structured JSON logs:

```python
# Called in chatbot.py, api/app.py, all agents
metrics.log_request(user_id, endpoint, method, query, agent, response_time_ms, status)
metrics.log_rag(user_id, query, retrieved_docs, response_time_ms)
metrics.log_agent(user_id, agent_name, action, input_summary, output_summary, duration_ms, success)
```

Live metrics are visible at `/api/metrics`:
- Total API requests + avg response time
- RAG retrieval count + avg top score
- Agent call count + success rate
- Recent request log table

### 6. Testing (Unit + Integration)

**Files:** `tests/test_rag.py`, `tests/test_agents.py`, `tests/test_integration.py`

```
91 tests, 0 failures
```

**Unit tests** (`test_rag.py` — 24 tests):
- Tokenizer: stopword removal, punctuation, casing, length filtering
- VectorStore: indexing, TF-IDF scoring, cosine similarity, top-k, metadata
- Chunker: single/multi chunk, overlap

**Unit tests** (`test_agents.py` — 55 tests):
- PlannerAgent: `run()` interface, domain detection, level filtering, known-topic exclusion, MCP recording
- ValidationAgent: input/output guardrails, confidence threshold, file uploads, roadmap validation
- QuizAgent: generation, evaluation, score bands, feedback, MCP context
- ResourceAgent: curated resources, free count, RAG-based related topics
- MCPContext: serialization, prompt block, immutable mutations

**Integration tests** (`test_integration.py` — 12 tests):
- Full pipeline: Input → Validation → Planner → Roadmap (ML, Python, Web Dev, NLP)
- Roadmap → Resource fetch (all steps)
- Quiz generate → evaluate → pass/fail recommendation
- Chat RAG pipeline: query → retrieve → validate → respond
- End-to-end user journey: new user → goal → roadmap → resources → quiz → 100% score

---

## Project Structure

```
learning_system/
├── agents/
│   ├── planner_agent.py      # Roadmap generation with decision logic
│   ├── resource_agent.py     # Resource fetching via RAG
│   ├── validation_agent.py   # Input/output guardrails
│   ├── quiz_agent.py         # Quiz generation + evaluation
│   └── progress_agent.py     # Progress tracking, SQLite persistence
├── rag/
│   ├── vector_store.py       # TF-IDF vector search engine
│   ├── knowledge_base.py     # 15-topic knowledge base + quiz bank
│   ├── mcp_context.py        # MCPContext dataclass (structured protocol)
│   └── chatbot.py            # RAG-grounded Q&A engine
├── observability/
│   └── metrics.py            # Structured logging + SQLite metrics
├── api/
│   └── app.py                # Flask REST API (all routes)
├── static/
│   ├── index.html            # Frontend SPA
│   ├── css/style.css         # UI styles
│   └── js/app.js             # Frontend logic
├── tests/
│   ├── test_rag.py           # 24 RAG unit tests
│   ├── test_agents.py        # 55 agent unit tests
│   └── test_integration.py   # 12 integration tests (full pipeline)
└── data/
    ├── progress.db           # SQLite — user progress
    └── metrics.db            # SQLite — observability logs
```

---

## Running the Project

```bash
# Start the API server
cd learning_system
python3 api/app.py
# → http://localhost:5000

# Run all tests
python3 -m unittest discover -s tests -p "test_*.py" -v
# → 91 tests, 0 failures
```

---

## API Endpoints

| Method | Endpoint | Agent | Description |
|---|---|---|---|
| POST | `/api/user` | ProgressAgent | Create/fetch user |
| POST | `/api/roadmap` | PlannerAgent | Generate learning roadmap |
| GET | `/api/roadmap` | ProgressAgent | Fetch saved roadmap |
| POST | `/api/chat` | ChatbotEngine | RAG-grounded Q&A |
| GET | `/api/quiz/<topic>` | QuizAgent | Generate quiz |
| POST | `/api/quiz/<topic>/submit` | QuizAgent | Evaluate answers |
| GET | `/api/resources/<topic>` | ResourceAgent | Fetch resources |
| PUT | `/api/progress/<topic>` | ProgressAgent | Update topic status |
| GET | `/api/progress` | ProgressAgent | Full dashboard |
| GET | `/api/metrics` | Observability | System metrics |
| GET | `/api/topics` | Knowledge Base | List all topics |

---

## Knowledge Base Topics

| Domain | Topics | Levels |
|---|---|---|
| Mathematics | Linear Algebra, Calculus, Statistics | Beginner |
| Python | Python Intro, Python OOP | Beginner → Intermediate |
| Data Science | Pandas/NumPy, Data Visualization | Beginner → Intermediate |
| Machine Learning | ML Foundations, Neural Networks | Intermediate |
| Deep Learning | Deep Learning | Advanced |
| NLP / AI | Transformers, RAG Systems, Prompt Engineering | Advanced |
| Web Dev | HTML/CSS, JavaScript | Beginner → Intermediate |

---

## Design Decisions

**Why TF-IDF instead of embeddings?** Zero external dependencies — runs offline, no API keys, deterministic. Examiner can run it immediately with just `python3`.

**Why SQLite?** No database server setup needed. Progress and metrics persist between sessions without configuration.

**Why a `run(MCPContext)` interface on every agent?** Standardizes the agent contract — any orchestrator can call any agent with the same structured input. This is the core of the agentic pattern.

**Why server-side answer caching for quizzes?** Prevents client-side cheating — correct answers never leave the server. The API strips `_answers` before the JSON response is sent.
