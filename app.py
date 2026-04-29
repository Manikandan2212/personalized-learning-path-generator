"""
Flask API
RESTful API for the Personalized Learning Path Generator.
Orchestrates all agents with MCP context injection.
"""

import time
import logging
import os

from flask import Flask, request, jsonify, send_from_directory

# Core components
from rag.vector_store import VectorStore
from rag.knowledge_base import seed_knowledge_base, get_all_topics
from rag.chatbot import ChatbotEngine
from rag.mcp_context import MCPContext  

from agents.planner_agent import PlannerAgent
from agents.resource_agent import ResourceAgent
from agents.validation_agent import ValidationAgent
from agents.quiz_agent import QuizAgent
from agents.progress_agent import ProgressAgent

from observability.metrics import metrics, setup_logging

# Setup
setup_logging()
logger = logging.getLogger("api")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"))
app.config["JSON_SORT_KEYS"] = False

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

def get_user_id():
    return request.headers.get("X-User-ID") or request.args.get("user_id") or "default_user"
@app.route("/api/progress", methods=["GET"])
def get_progress():
    user_id = get_user_id()
    
    try:
        dashboard = progress_agent.get_dashboard(user_id)

        return jsonify({
            "success": True,
            "data": dashboard
        })

    except Exception as e:
        import traceback
        print("ERROR in /api/progress:", str(e))
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    
def build_mcp_context(user_id: str) -> MCPContext:
    progress_data = progress_agent.get_dashboard(user_id)
    roadmap = progress_agent.get_roadmap(user_id)

    return MCPContext.from_progress(
        user_id=user_id,
        progress_data=progress_data,
        roadmap=roadmap
    )


def ok(data, status=200):
    return jsonify({"success": True, "data": data}), status


def err(message, status=400):
    return jsonify({"success": False, "error": message}), status


# ── MIDDLEWARE ──────────────────────────────────────────────

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


# ── ROUTES ─────────────────────────────────────────────────

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


# ── USER ─────────────────────────────────────────────────

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
    return ok(user)


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

    # Ensure user exists
    progress_agent.get_or_create_user(user_id, goal=goal, hours_per_week=hours_per_week)
    progress_agent.update_user(user_id, {"goal": goal, "hours_per_week": hours_per_week})

    ctx = build_mcp_context(user_id)
    ctx = ctx.with_request_type("roadmap")

    # Override goal dynamically
    ctx.user_goal = goal
    ctx.hours_per_week = hours_per_week

    t0 = time.time()
    roadmap = planner_agent.run(ctx)
    duration = (time.time() - t0) * 1000

    # Validate roadmap
    validated = validation_agent.validate_roadmap(roadmap)
    if not validated["valid"]:
        return err(validated["reason"])

    # Save
    progress_agent.save_roadmap(user_id, roadmap)

    metrics.log_agent(user_id, "PlannerAgent", "generate_roadmap",
                      goal[:100], f"{roadmap['total_steps']} steps", duration, True)

    return ok({
        "roadmap": roadmap,
        "mcp_context": ctx.to_dict()  
    })


# ── CHAT (UPDATED MCP) ─────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json() or {}
    user_id = get_user_id()

    query = body.get("message", "").strip()
    if not query:
        return err("'message' field is required")

    ctx = build_mcp_context(user_id)
    ctx = ctx.with_request_type("query")

    result = chatbot.answer(query, user_id, ctx.to_dict())

    return jsonify({
        "success": result["valid"],
        "data": result
    })


# ── RUN ───────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting Learning System API on port 5000")
    app.run(host="0.0.0.0", port=5000, debug=False)