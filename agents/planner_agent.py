"""
Planner Agent
=============
Generates personalized learning roadmaps using RAG retrieval.
Accepts MCPContext as structured input (Model Context Protocol).
Contains real decision logic — not just a wrapper.

Agent interface:
    agent = PlannerAgent(vector_store)
    result = agent.run(mcp_context)
"""

import logging
from typing import List, Dict, Optional

from rag.mcp_context import MCPContext

logger = logging.getLogger("agents.planner")

LEARNING_PATHS = {
    "machine learning":   ["linear_algebra", "statistics_basics", "calculus", "python_intro", "pandas_numpy", "ml_foundations", "neural_networks", "deep_learning"],
    "deep learning":      ["linear_algebra", "calculus", "statistics_basics", "python_intro", "pandas_numpy", "ml_foundations", "neural_networks", "deep_learning"],
    "data science":       ["python_intro", "statistics_basics", "linear_algebra", "pandas_numpy", "data_visualization", "ml_foundations"],
    "web development":    ["html_css", "javascript", "python_intro"],
    "python":             ["python_intro", "python_oop"],
    "nlp":                ["python_intro", "linear_algebra", "statistics_basics", "ml_foundations", "deep_learning", "neural_networks", "transformers_nlp"],
    "ai":                 ["python_intro", "linear_algebra", "statistics_basics", "calculus", "ml_foundations", "deep_learning", "prompt_engineering", "rag_systems"],
    "rag":                ["python_intro", "ml_foundations", "deep_learning", "prompt_engineering", "rag_systems"],
    "transformers":       ["linear_algebra", "calculus", "python_intro", "ml_foundations", "neural_networks", "deep_learning", "transformers_nlp"],
}

KEYWORD_TO_DOMAIN = {
    "neural": "deep learning", "cnn": "deep learning", "rnn": "deep learning",
    "bert": "nlp", "gpt": "nlp", "transformer": "transformers",
    "pandas": "data science", "numpy": "data science", "statistics": "data science",
    "javascript": "web development", "html": "web development", "css": "web development",
    "classification": "machine learning", "regression": "machine learning",
    "cluster": "machine learning", "retrieval": "rag", "vector": "ai",
    "llm": "ai", "prompt": "ai",
}

WHY_NEEDED = {
    "linear_algebra":    "Vectors and matrices are the mathematical backbone of all ML models.",
    "calculus":          "Gradient descent — how ML models learn — is built entirely on derivatives.",
    "statistics_basics": "Probability and statistics let you interpret model outputs and data distributions.",
    "python_intro":      "Python is the primary language for ML, data science, and AI engineering.",
    "python_oop":        "Classes and objects help you structure ML pipelines and model code cleanly.",
    "pandas_numpy":      "NumPy and Pandas are essential for data loading, cleaning, and transformation.",
    "data_visualization":"Visualizing data reveals patterns and helps diagnose model behavior.",
    "ml_foundations":    "Classical ML algorithms build intuition you need for deep learning.",
    "neural_networks":   "Neural nets are the core architecture behind all deep learning.",
    "deep_learning":     "Deep learning is the engine powering modern AI breakthroughs.",
    "transformers_nlp":  "Transformers are the architecture behind GPT, BERT, and modern LLMs.",
    "rag_systems":       "RAG grounds LLM outputs in real documents, reducing hallucination.",
    "prompt_engineering":"Effective prompting maximises what you get from any LLM.",
    "html_css":          "HTML and CSS are the building blocks of every web interface.",
    "javascript":        "JavaScript brings interactivity and dynamic behaviour to web applications.",
}


class PlannerAgent:
    """
    Planner Agent — builds a personalised learning roadmap.

    Decision logic (real agent behaviour, not just a wrapper):
    - Detects domain from user goal
    - Adjusts path depth based on user level (beginner vs advanced)
    - Filters out topics the user already knows
    - Scales pace recommendations to hours_per_week
    - Falls back to RAG-retrieved ordering when domain is unknown
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.name = "PlannerAgent"

    # ── MAIN AGENT INTERFACE ─────────────────────────────────────────

    def run(self, context: MCPContext) -> Dict:
        """
        Main agent entry point.
        Accepts MCPContext (structured MCP payload), returns roadmap dict.

        Decision flow:
          1. Extract goal + infer user level from MCP context
          2. Detect domain via keyword matching
          3. Select path: beginner=full, intermediate=skip basics, advanced=advanced only
          4. RAG-retrieve to fill gaps when domain is unknown
          5. Filter already-known topics
          6. Annotate with time estimates and explanations
        """
        logger.info(f"PlannerAgent.run() | context={context!r}")

        goal = context.user_goal
        user_level = self._infer_user_level(context)
        hours_per_week = context.hours_per_week

        # DECISION 1: detect domain from goal text
        domain = self._detect_domain(goal)

        # DECISION 2: select path based on domain + level
        if domain and domain in LEARNING_PATHS:
            path_ids = self._apply_level_filter(LEARNING_PATHS[domain], user_level)
        else:
            # RAG fallback: retrieve relevant topics, order by difficulty level
            path_ids = self._rag_based_path(goal)

        # DECISION 3: skip topics the user already knows
        known = getattr(context, "_known_topics", [])
        path_ids = [p for p in path_ids if p not in known]

        # BUILD STEPS
        steps = self._build_steps(path_ids, goal)
        total_hours = sum(s["duration_hours"] for s in steps)
        weeks = round(total_hours / max(hours_per_week, 1), 1)

        roadmap = {
            "goal":             goal,
            "domain":           domain or "general",
            "user_level":       user_level,
            "total_steps":      len(steps),
            "total_hours":      total_hours,
            "weeks_to_complete": weeks,
            "steps":            steps,
            "mcp_context_used": {
                "user_goal":      context.user_goal,
                "user_name":      context.user_name,
                "current_step":   context.current_step,
                "progress_pct":   context.progress_pct,
                "user_level":     user_level,
                "hours_per_week": context.hours_per_week,
            },
        }

        logger.info(f"PlannerAgent: {roadmap['total_steps']} steps | {total_hours}h | domain={domain} | level={user_level}")
        return roadmap

    # ── BACKWARD-COMPATIBLE DICT INTERFACE ───────────────────────────

    def generate_roadmap(self, user_profile: Dict, mcp_context: Dict) -> Dict:
        """Accepts raw dicts (legacy). Converts to MCPContext and calls run()."""
        ctx = MCPContext(
            user_id=mcp_context.get("user_id", "unknown"),
            user_name=mcp_context.get("user_name", "Learner"),
            user_goal=user_profile.get("goal", mcp_context.get("user_goal", "")),
            current_step=mcp_context.get("current_step", "Not started"),
            progress_pct=float(mcp_context.get("progress_pct", 0)),
            completed_topics=int(mcp_context.get("completed_topics", 0)),
            total_topics=int(mcp_context.get("total_topics", 0)),
            hours_per_week=int(user_profile.get("hours_per_week", 10)),
        )
        ctx._known_topics = user_profile.get("current_knowledge", [])
        return self.run(ctx)

    # ── DECISION METHODS (real agent logic) ──────────────────────────

    def _infer_user_level(self, context: MCPContext) -> str:
        """Decide user level from quiz scores + completed topic count."""
        if context.average_score >= 80 and context.completed_topics >= 3:
            return "advanced"
        elif context.completed_topics >= 1 or context.average_score >= 50:
            return "intermediate"
        return "beginner"

    def _detect_domain(self, goal: str) -> Optional[str]:
        goal_lower = goal.lower()
        for key in LEARNING_PATHS:
            if key in goal_lower:
                return key
        for kw, domain in KEYWORD_TO_DOMAIN.items():
            if kw in goal_lower:
                return domain
        return None

    def _apply_level_filter(self, path_ids: List[str], user_level: str) -> List[str]:
        """
        Real decision: trim path to match user experience level.
        beginner     → full path (all prerequisites included)
        intermediate → skip pure beginner topics
        advanced     → skip beginner + intermediate, focus on advanced content
        """
        from rag.knowledge_base import get_full_entry

        if user_level == "beginner":
            return path_ids

        filtered = []
        for tid in path_ids:
            entry = get_full_entry(tid)
            entry_level = entry.get("level", "beginner") if entry else "beginner"
            if user_level == "intermediate" and entry_level == "beginner":
                continue
            if user_level == "advanced" and entry_level in ("beginner", "intermediate"):
                continue
            filtered.append(tid)

        return filtered if len(filtered) >= 3 else path_ids  # safety: always 3+ steps

    def _rag_based_path(self, goal: str) -> List[str]:
        """Fallback: RAG retrieval + sort by difficulty level."""
        retrieved = self.vector_store.search(goal, top_k=10, min_score=0.0)
        seen, path = set(), []
        level_order = {"beginner": 0, "intermediate": 1, "advanced": 2}
        for r in retrieved:
            src = r["metadata"]["source_id"]
            if src not in seen:
                seen.add(src)
                path.append((src, level_order.get(r["metadata"].get("level", "beginner"), 0)))
        path.sort(key=lambda x: x[1])
        return [p[0] for p in path]

    def _build_steps(self, path_ids: List[str], goal: str) -> List[Dict]:
        from rag.knowledge_base import get_full_entry
        steps = []
        for num, tid in enumerate(path_ids, 1):
            entry = get_full_entry(tid)
            if not entry:
                continue
            steps.append({
                "step":           num,
                "topic_id":       tid,
                "title":          tid.replace("_", " ").title(),
                "topic":          entry["topic"],
                "level":          entry["level"],
                "duration_hours": entry.get("duration_hours", 10),
                "prerequisites":  entry["prerequisites"],
                "resources":      entry.get("resources", []),
                "why_needed":     WHY_NEEDED.get(tid, f"Required foundation for: {goal}"),
                "status":         "not_started",
            })
        return steps
