"""
Planner Agent
Generates personalized learning roadmaps using RAG retrieval.
Builds a dependency-resolved step-by-step path based on user goal.
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger("agents.planner")

LEVEL_ORDER = {"beginner": 0, "intermediate": 1, "advanced": 2}

DOMAIN_PREREQUISITES = {
    "Machine Learning": ["linear_algebra", "statistics_basics", "python_intro"],
    "Deep Learning": ["ml_foundations", "linear_algebra", "calculus", "python_intro"],
    "NLP": ["deep_learning", "ml_foundations", "python_intro"],
    "AI Engineering": ["ml_foundations", "python_intro"],
    "Data Science": ["python_intro", "statistics_basics"],
    "Web Development": [],
    "Python": [],
    "Mathematics": [],
}

LEARNING_PATHS = {
    "machine learning": ["linear_algebra", "statistics_basics", "calculus", "python_intro", "pandas_numpy", "ml_foundations", "neural_networks", "deep_learning"],
    "deep learning": ["linear_algebra", "calculus", "statistics_basics", "python_intro", "pandas_numpy", "ml_foundations", "neural_networks", "deep_learning"],
    "data science": ["python_intro", "statistics_basics", "linear_algebra", "pandas_numpy", "data_visualization", "ml_foundations"],
    "web development": ["html_css", "javascript", "python_intro"],
    "python": ["python_intro", "python_oop"],
    "nlp": ["python_intro", "linear_algebra", "statistics_basics", "ml_foundations", "deep_learning", "neural_networks", "transformers_nlp"],
    "ai": ["python_intro", "linear_algebra", "statistics_basics", "calculus", "ml_foundations", "deep_learning", "prompt_engineering", "rag_systems"],
    "rag": ["python_intro", "ml_foundations", "deep_learning", "prompt_engineering", "rag_systems"],
    "transformers": ["linear_algebra", "calculus", "python_intro", "ml_foundations", "neural_networks", "deep_learning", "transformers_nlp"],
}


class PlannerAgent:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.name = "PlannerAgent"

    def _detect_goal_domain(self, goal: str) -> Optional[str]:
        goal_lower = goal.lower()
        for key in LEARNING_PATHS:
            if key in goal_lower:
                return key
        keyword_map = {
            "neural": "deep learning",
            "cnn": "deep learning",
            "rnn": "deep learning",
            "bert": "nlp",
            "gpt": "nlp",
            "transformer": "transformers",
            "pandas": "data science",
            "numpy": "data science",
            "statistics": "data science",
            "javascript": "web development",
            "html": "web development",
            "css": "web development",
            "classification": "machine learning",
            "regression": "machine learning",
            "cluster": "machine learning",
            "retrieval": "rag",
            "vector": "ai",
            "llm": "ai",
            "prompt": "ai",
        }
        for kw, domain in keyword_map.items():
            if kw in goal_lower:
                return domain
        return None

    def generate_roadmap(self, user_profile: Dict, mcp_context: Dict) -> Dict:
        goal = user_profile.get("goal", "")
        current_knowledge = user_profile.get("current_knowledge", [])
        available_hours_per_week = user_profile.get("hours_per_week", 10)

        logger.info(f"PlannerAgent: generating roadmap for goal='{goal}'")

        # RAG retrieval for relevant topics
        retrieved = self.vector_store.search(goal, top_k=8, min_score=0.0)

        # Determine domain
        domain = self._detect_goal_domain(goal)
        if domain and domain in LEARNING_PATHS:
            path_ids = LEARNING_PATHS[domain]
        else:
            # Fall back to retrieval-based path
            seen = set()
            path_ids = []
            for r in retrieved:
                src = r["metadata"]["source_id"]
                if src not in seen:
                    seen.add(src)
                    path_ids.append(src)

        # Filter out already known topics
        path_ids = [p for p in path_ids if p not in current_knowledge]

        # Build step objects
        from rag.knowledge_base import get_full_entry
        steps = []
        total_hours = 0
        for step_num, topic_id in enumerate(path_ids, 1):
            entry = get_full_entry(topic_id)
            if not entry:
                continue
            duration = entry.get("duration_hours", 10)
            total_hours += duration
            steps.append({
                "step": step_num,
                "topic_id": topic_id,
                "title": topic_id.replace("_", " ").title(),
                "topic": entry["topic"],
                "level": entry["level"],
                "duration_hours": duration,
                "prerequisites": entry["prerequisites"],
                "resources": entry.get("resources", []),
                "why_needed": self._explain_why(topic_id, goal),
                "status": "not_started",
            })

        weeks_to_complete = round(total_hours / max(available_hours_per_week, 1), 1)

        roadmap = {
            "goal": goal,
            "domain": domain or "general",
            "total_steps": len(steps),
            "total_hours": total_hours,
            "weeks_to_complete": weeks_to_complete,
            "steps": steps,
            "retrieved_context": [
                {"doc_id": r["doc_id"], "score": r["score"], "topic": r["metadata"]["topic"]}
                for r in retrieved[:5]
            ],
            "mcp_context_used": {
                "user_goal": mcp_context.get("user_goal"),
                "current_step": mcp_context.get("current_step"),
                "progress_pct": mcp_context.get("progress_pct"),
            },
        }

        logger.info(f"PlannerAgent: roadmap created — {len(steps)} steps, {total_hours}h total")
        return roadmap

    def _explain_why(self, topic_id: str, goal: str) -> str:
        explanations = {
            "linear_algebra": "Vectors and matrices are the mathematical backbone of all ML models.",
            "calculus": "Gradient descent — how ML models learn — is built on derivatives.",
            "statistics_basics": "Probability and statistics let you interpret model outputs and data distributions.",
            "python_intro": "Python is the primary language for ML, data science, and AI engineering.",
            "python_oop": "Classes and objects help you structure ML pipelines and model code cleanly.",
            "pandas_numpy": "NumPy and Pandas are essential for data loading, cleaning, and transformation.",
            "data_visualization": "Visualizing data reveals patterns and helps diagnose model behavior.",
            "ml_foundations": "Classical ML algorithms build intuition you'll need for deep learning.",
            "neural_networks": "Neural nets are the core architecture behind all deep learning.",
            "deep_learning": "Deep learning is the engine powering modern AI breakthroughs.",
            "transformers_nlp": "Transformers are the architecture behind GPT, BERT, and modern LLMs.",
            "rag_systems": "RAG grounds LLM outputs in real documents, reducing hallucination.",
            "prompt_engineering": "Effective prompting maximizes what you get from any LLM.",
            "html_css": "HTML and CSS are the building blocks of every web interface.",
            "javascript": "JavaScript brings interactivity and dynamic behavior to web applications.",
        }
        return explanations.get(topic_id, f"Required foundation for achieving: {goal}")
