"""
Flask API
RESTful API for the Personalized Learning Path Generator.
Orchestrates all agents with MCP context injection.
"""

import json
import time
import uuid
import logging
import os
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


from flask import Flask, request, jsonify, send_from_directory

# Core components
from rag.vector_store import VectorStore
from rag.knowledge_base import seed_knowledge_base, get_all_topics
from rag.chatbot import ChatbotEngine
from agents.planner_agent import PlannerAgent
from agents.resource_agent import ResourceAgent
from agents.validation_agent import ValidationAgent
from agents.quiz_agent import QuizAgent
from agents.progress_agent import ProgressAgent
from observability.metrics import metrics, setup_logging

setup_logging()
logger = logging.getLogger("api")

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"))
app.config["JSON_SORT_KEYS"] = False

# ── BOOTSTRAP SYSTEM ─────────────────────────────────────────────────────

vector_store = VectorStore()
seed_count = seed_knowledge_base(vector_store)
logger.info(f"Vector store seeded with {seed_count} chunks")

validation_agent = ValidationAgent()
planner_agent = PlannerAgent(vector_store)
resource_agent = ResourceAgent(vector_store)
quiz_agent = QuizAgent(vector_store)
progress_agent = ProgressAgent()
chatbot = ChatbotEngine(vector_store, validation_agent, metrics)

logger.info("All agents initialized and ready")


# ── HELPERS ──────────────────────────────────────────────────────────────

def get_user_id() -> str:
    """Get or create user ID from request."""
    return request.headers.get("X-User-ID") or request.args.get("user_id") or "default_user"


def mcp(user_id: str, extra_docs: list = None) -> dict:
    """Build MCP context for the current user."""
    ctx = progress_agent.build_mcp_context(user_id)
    if extra_docs:
        ctx["retrieved_docs"] = extra_docs
    return ctx


def ok(data: dict, status: int = 200):
    return jsonify({"success": True, "data": data}), status


def err(message: str, status: int = 400):
    return jsonify({"success": False, "error": message}), status


# ── CORS + MIDDLEWARE ─────────────────────────────────────────────────────

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-User-ID"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    return response


@app.before_request
def log_request():
    request.start_time = time.time()


@app.after_request
def log_response(response):
    duration = (time.time() - getattr(request, "start_time", time.time())) * 1000
    logger.info(f"{request.method} {request.path} → {response.status_code} ({duration:.0f}ms)")
    return response


# ── ROUTES ───────────────────────────────────────────────────────────────
from flask import send_from_directory
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return send_from_directory(os.path.join(BASE_DIR, "static"), "index.html")


@app.route("/api/health")
def health():
    return ok({
        "status": "healthy",
        "vector_store_docs": vector_store.stats()["total_documents"],
        "agents": ["PlannerAgent", "ResourceAgent", "ValidationAgent", "QuizAgent", "ProgressAgent"],
        "version": "1.0.0",
    })


# ── USER ─────────────────────────────────────────────────────────────────

@app.route("/api/user", methods=["GET"])
def get_user():
    user_id = get_user_id()
    user = progress_agent.get_or_create_user(user_id)
    return ok(user)


@app.route("/api/user", methods=["POST"])
def create_user():
    body = request.get_json() or {}
    user_id = get_user_id()
    name = body.get("name", "Learner")
    goal = body.get("goal", "")
    hours = body.get("hours_per_week", 10)

    validation = validation_agent.validate_input(goal, "query")
    if goal and not validation["valid"]:
        return err(validation["reason"])

    user = progress_agent.get_or_create_user(user_id, name, goal, hours)
    logger.info(f"User created/fetched: {user_id}, goal='{goal}'")
    return ok(user)


@app.route("/api/user", methods=["PUT"])
def update_user():
    body = request.get_json() or {}
    user_id = get_user_id()
    user = progress_agent.update_user(user_id, body)
    return ok(user)


# ── ROADMAP ──────────────────────────────────────────────────────────────

@app.route("/api/roadmap", methods=["POST"])
def generate_roadmap():
    body = request.get_json() or {}
    user_id = get_user_id()

    goal = body.get("goal", "")
    if not goal:
        return err("'goal' field is required")

    validation = validation_agent.validate_input(goal, "query")
    if not validation["valid"]:
        return err(validation["reason"])

    hours_per_week = body.get("hours_per_week", 10)
    current_knowledge = body.get("current_knowledge", [])

    # Ensure user exists
    progress_agent.get_or_create_user(user_id, goal=goal, hours_per_week=hours_per_week)
    progress_agent.update_user(user_id, {"goal": goal, "hours_per_week": hours_per_week})

    # Build MCP context
    mcp_ctx = mcp(user_id)
    mcp_ctx["user_goal"] = goal

    user_profile = {
        "goal": goal,
        "current_knowledge": current_knowledge,
        "hours_per_week": hours_per_week,
    }

    t0 = time.time()
    roadmap = planner_agent.generate_roadmap(user_profile, mcp_ctx)
    duration = (time.time() - t0) * 1000

    # Validate roadmap
    validated = validation_agent.validate_roadmap(roadmap)
    if not validated["valid"]:
        return err(validated["reason"])

    # Save to progress DB
    progress_agent.save_roadmap(user_id, roadmap)

    metrics.log_agent(user_id, "PlannerAgent", "generate_roadmap", goal[:100],
                      f"{roadmap['total_steps']} steps", duration, True)
    metrics.log_request(user_id, "/api/roadmap", "POST", goal, "PlannerAgent", duration, 200)

    return ok({
        "roadmap": roadmap,
        "validation": {k: v for k, v in validated.items() if k != "roadmap"},
        "mcp_context": mcp_ctx,
    })


@app.route("/api/roadmap", methods=["GET"])
def get_roadmap():
    user_id = get_user_id()
    roadmap = progress_agent.get_roadmap(user_id)
    if not roadmap:
        return err("No roadmap found. Generate one first.", 404)
    return ok({"roadmap": roadmap})


# ── RESOURCES ─────────────────────────────────────────────────────────────

@app.route("/api/resources/<topic_id>", methods=["GET"])
def get_resources(topic_id):
    user_id = get_user_id()
    mcp_ctx = mcp(user_id)
    t0 = time.time()
    result = resource_agent.fetch_resources(topic_id, mcp_ctx)
    metrics.log_agent(user_id, "ResourceAgent", "fetch", topic_id, f"{result['summary']['total_resources']} resources",
                      (time.time() - t0) * 1000, True)
    return ok(result)


@app.route("/api/resources/search", methods=["GET"])
def search_resources():
    user_id = get_user_id()
    query = request.args.get("q", "")
    if not query:
        return err("Query parameter 'q' required")
    mcp_ctx = mcp(user_id)
    result = resource_agent.search_resources(query, mcp_ctx)
    return ok(result)


# ── CHAT ──────────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json() or {}
    user_id = get_user_id()
    query = body.get("message", "").strip()

    if not query:
        return err("'message' field is required")

    mcp_ctx = mcp(user_id)
    result = chatbot.answer(query, user_id, mcp_ctx)

    status = 200 if result["valid"] else 422
    return jsonify({"success": result["valid"], "data": result}), status


# ── QUIZ ──────────────────────────────────────────────────────────────────

@app.route("/api/quiz/<topic_id>", methods=["GET"])
def get_quiz(topic_id):
    user_id = get_user_id()
    mcp_ctx = mcp(user_id)
    user = progress_agent.get_or_create_user(user_id)

    t0 = time.time()
    quiz = quiz_agent.generate_quiz(topic_id, user, mcp_ctx)
    duration = (time.time() - t0) * 1000

    # Strip server-side answers before sending
    answers = quiz.pop("_answers", {})
    quiz["_token"] = str(hash(frozenset(answers.items())))  # simple token

    # Store answers server-side in progress DB temporarily (use cache here for simplicity)
    app.quiz_cache = getattr(app, "quiz_cache", {})
    app.quiz_cache[f"{user_id}:{topic_id}"] = answers

    metrics.log_agent(user_id, "QuizAgent", "generate", topic_id, f"{quiz['total_questions']} questions", duration, True)
    return ok(quiz)


@app.route("/api/quiz/<topic_id>/submit", methods=["POST"])
def submit_quiz(topic_id):
    body = request.get_json() or {}
    user_id = get_user_id()
    user_answers = body.get("answers", {})

    # Convert keys to int
    user_answers = {int(k): int(v) for k, v in user_answers.items()}

    # Retrieve correct answers from server cache
    cache_key = f"{user_id}:{topic_id}"
    correct_answers = getattr(app, "quiz_cache", {}).get(cache_key, {})

    if not correct_answers:
        return err("Quiz session expired. Please reload the quiz.", 404)

    t0 = time.time()
    result = quiz_agent.evaluate_answers("", topic_id, user_answers, correct_answers)
    duration = (time.time() - t0) * 1000

    # Record in progress DB
    progress_agent.record_quiz_result(
        user_id, topic_id, result["score_pct"], result["correct"], result["total"], result["passed"]
    )

    metrics.log_agent(user_id, "QuizAgent", "evaluate", topic_id, f"score={result['score_pct']}%", duration, True)
    return ok(result)


# ── PROGRESS ──────────────────────────────────────────────────────────────

@app.route("/api/progress", methods=["GET"])
def get_progress():
    user_id = get_user_id()
    dashboard = progress_agent.get_dashboard(user_id)
    return ok(dashboard)


@app.route("/api/progress/<topic_id>", methods=["PUT"])
def update_progress(topic_id):
    body = request.get_json() or {}
    user_id = get_user_id()
    status = body.get("status", "in_progress")

    if status not in ("not_started", "in_progress", "completed"):
        return err("Invalid status. Use: not_started, in_progress, completed")

    result = progress_agent.update_topic_status(user_id, topic_id, status)
    return ok(result)


# ── TOPICS ────────────────────────────────────────────────────────────────

@app.route("/api/topics", methods=["GET"])
def list_topics():
    topics = get_all_topics()
    return ok({"topics": topics, "total": len(topics)})


# ── OBSERVABILITY ─────────────────────────────────────────────────────────

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    summary = metrics.get_summary()
    rag_stats = vector_store.stats()
    return ok({
        "system_metrics": summary,
        "vector_store": rag_stats,
        "agents": {
            "active": ["PlannerAgent", "ResourceAgent", "ValidationAgent", "QuizAgent", "ProgressAgent"],
            "total": 5,
        },
    })


@app.route("/api/vector-store/search", methods=["GET"])
def vector_search():
    query = request.args.get("q", "")
    top_k = int(request.args.get("k", 5))
    if not query:
        return err("Query parameter 'q' required")
    results = vector_store.search(query, top_k=top_k)
    return ok({"query": query, "results": results, "count": len(results)})


if __name__ == "__main__":
    logger.info("Starting Learning System API on port 5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
