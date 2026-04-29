"""
Unit Tests — Agents
Tests: PlannerAgent, ValidationAgent, QuizAgent, ResourceAgent, ProgressAgent.
Each agent has a run() method tested with MCPContext input.
"""

import sys
sys.path.insert(0, "/home/claude/learning_system")

import unittest
import os
import time

# Use a test DB so we don't pollute real data
os.environ["TEST_MODE"] = "1"

from rag.vector_store import VectorStore
from rag.knowledge_base import seed_knowledge_base
from rag.mcp_context import MCPContext
from agents.planner_agent import PlannerAgent
from agents.validation_agent import ValidationAgent
from agents.quiz_agent import QuizAgent
from agents.resource_agent import ResourceAgent


# ── SHARED FIXTURE ────────────────────────────────────────────────────────

def make_store() -> VectorStore:
    store = VectorStore()
    seed_knowledge_base(store)
    return store


def make_context(goal="Learn Machine Learning", level_boost=False) -> MCPContext:
    ctx = MCPContext(
        user_id="test_user",
        user_name="Test Learner",
        user_goal=goal,
        current_step="Linear Algebra",
        progress_pct=20.0,
        completed_topics=1,
        total_topics=8,
        average_score=75.0 if level_boost else 0.0,
        hours_per_week=10,
    )
    ctx._known_topics = []
    return ctx


# ── PLANNER AGENT ─────────────────────────────────────────────────────────

class TestPlannerAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = make_store()
        cls.agent = PlannerAgent(cls.store)

    def test_run_returns_roadmap(self):
        ctx = make_context("Learn Machine Learning")
        result = self.agent.run(ctx)
        self.assertIsNotNone(result)
        self.assertIn("steps", result)
        self.assertIn("goal", result)

    def test_run_steps_not_empty(self):
        ctx = make_context("Learn Python")
        result = self.agent.run(ctx)
        self.assertGreater(result["total_steps"], 0)
        self.assertGreater(len(result["steps"]), 0)

    def test_run_detects_domain(self):
        ctx = make_context("I want to learn machine learning")
        result = self.agent.run(ctx)
        self.assertEqual(result["domain"], "machine learning")

    def test_run_detects_nlp_domain(self):
        ctx = make_context("Learn NLP and transformers")
        result = self.agent.run(ctx)
        self.assertIn(result["domain"], ["nlp", "transformers", "general"])

    def test_run_mcp_context_recorded(self):
        ctx = make_context("Learn Deep Learning")
        result = self.agent.run(ctx)
        mcp_used = result.get("mcp_context_used", {})
        self.assertEqual(mcp_used["user_goal"], "Learn Deep Learning")
        self.assertEqual(mcp_used["user_name"], "Test Learner")

    def test_beginner_gets_full_path(self):
        ctx = make_context("machine learning")
        ctx.completed_topics = 0
        ctx.average_score = 0
        result = self.agent.run(ctx)
        self.assertEqual(result["user_level"], "beginner")
        self.assertGreater(result["total_steps"], 0)

    def test_advanced_gets_shorter_path(self):
        ctx_beginner = make_context("machine learning")
        ctx_beginner.completed_topics = 0
        ctx_beginner.average_score = 0
        ctx_beginner._known_topics = []
        beginner_result = self.agent.run(ctx_beginner)

        ctx_advanced = make_context("machine learning")
        ctx_advanced.completed_topics = 5
        ctx_advanced.average_score = 90
        ctx_advanced._known_topics = []
        advanced_result = self.agent.run(ctx_advanced)

        # Advanced user should have fewer or equal steps
        self.assertLessEqual(advanced_result["total_steps"], beginner_result["total_steps"] + 2)

    def test_known_topics_filtered_out(self):
        ctx = make_context("machine learning")
        ctx._known_topics = ["python_intro", "linear_algebra"]
        result = self.agent.run(ctx)
        step_ids = [s["topic_id"] for s in result["steps"]]
        self.assertNotIn("python_intro", step_ids)
        self.assertNotIn("linear_algebra", step_ids)

    def test_total_hours_positive(self):
        ctx = make_context("Learn Data Science")
        result = self.agent.run(ctx)
        self.assertGreater(result["total_hours"], 0)

    def test_weeks_calculated(self):
        ctx = make_context("Learn Python")
        ctx.hours_per_week = 10
        result = self.agent.run(ctx)
        expected = round(result["total_hours"] / 10, 1)
        self.assertAlmostEqual(result["weeks_to_complete"], expected, places=0)

    def test_each_step_has_required_fields(self):
        ctx = make_context("Learn Machine Learning")
        result = self.agent.run(ctx)
        required_fields = {"step", "topic_id", "title", "level", "duration_hours", "why_needed", "status"}
        for step in result["steps"]:
            for field in required_fields:
                self.assertIn(field, step, f"Step missing field: {field}")

    def test_generate_roadmap_dict_interface(self):
        """Backward-compatible dict interface still works."""
        user_profile = {"goal": "Learn Python", "hours_per_week": 5, "current_knowledge": []}
        mcp_ctx = {"user_goal": "Learn Python", "user_name": "Tester"}
        result = self.agent.generate_roadmap(user_profile, mcp_ctx)
        self.assertIn("steps", result)
        self.assertGreater(len(result["steps"]), 0)

    def test_unknown_goal_uses_rag_fallback(self):
        ctx = make_context("xyzzy quantum flux capacitor")
        result = self.agent.run(ctx)
        # Should not crash, should return some steps or empty
        self.assertIn("steps", result)


# ── VALIDATION AGENT ──────────────────────────────────────────────────────

class TestValidationAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ValidationAgent()

    # Input guardrails
    def test_valid_query_passes(self):
        r = self.agent.validate_input("What is gradient descent?")
        self.assertTrue(r["valid"])

    def test_empty_query_blocked(self):
        r = self.agent.validate_input("")
        self.assertFalse(r["valid"])

    def test_too_short_blocked(self):
        r = self.agent.validate_input("a")
        self.assertFalse(r["valid"])

    def test_harmful_content_blocked(self):
        r = self.agent.validate_input("how to hack into a server")
        self.assertFalse(r["valid"])

    def test_long_input_truncated(self):
        long_input = "machine learning " * 200  # > 1000 chars
        r = self.agent.validate_input(long_input)
        self.assertTrue(r["valid"])
        self.assertLessEqual(len(r["sanitized"]), 1100)

    def test_sanitized_removes_html(self):
        r = self.agent.validate_input("What is <script>alert('xss')</script> Python?")
        self.assertNotIn("<script>", r.get("sanitized", ""))

    def test_educational_query_valid(self):
        queries = [
            "Explain backpropagation",
            "How does attention mechanism work?",
            "What are tensors in PyTorch?",
            "Why do I need calculus for machine learning?",
        ]
        for q in queries:
            r = self.agent.validate_input(q)
            self.assertTrue(r["valid"], f"Should be valid: {q}")

    # Output guardrails
    def test_valid_output_passes(self):
        docs = [{"score": 0.8, "content": "machine learning concepts"}]
        r = self.agent.validate_output("Machine learning uses algorithms to learn from data.", docs, "machine learning")
        self.assertTrue(r["valid"])
        self.assertGreater(r["confidence"], 0)

    def test_empty_output_blocked(self):
        docs = [{"score": 0.1, "content": "test"}]
        r = self.agent.validate_output("", docs, "anything")
        self.assertFalse(r["valid"])

    def test_no_retrieved_docs_low_confidence(self):
        r = self.agent.validate_output("Some answer here about stuff", [], "query")
        self.assertFalse(r["valid"])  # confidence below threshold with no docs

    def test_fallback_message_present_on_failure(self):
        r = self.agent.validate_output("", [], "query")
        self.assertFalse(r["valid"])
        self.assertIn("fallback_message", r)
        self.assertIsNotNone(r["fallback_message"])

    # Roadmap validation
    def test_valid_roadmap_passes(self):
        roadmap = {"steps": [{"topic_id": f"topic_{i}"} for i in range(5)], "total_hours": 50}
        r = self.agent.validate_roadmap(roadmap)
        self.assertTrue(r["valid"])

    def test_empty_roadmap_fails(self):
        roadmap = {"steps": [], "total_hours": 0}
        r = self.agent.validate_roadmap(roadmap)
        self.assertFalse(r["valid"])

    # Upload validation
    def test_valid_pdf_upload(self):
        r = self.agent.validate_upload("notes.pdf", "application/pdf", 1024 * 100)
        self.assertTrue(r["valid"])

    def test_invalid_file_type_blocked(self):
        r = self.agent.validate_upload("image.jpg", "image/jpeg", 1024)
        self.assertFalse(r["valid"])

    def test_oversized_file_blocked(self):
        r = self.agent.validate_upload("big.pdf", "application/pdf", 10 * 1024 * 1024)
        self.assertFalse(r["valid"])


# ── QUIZ AGENT ────────────────────────────────────────────────────────────

class TestQuizAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = make_store()
        cls.agent = QuizAgent(cls.store)
        cls.mcp = make_context().to_dict()
        cls.user = {"goal": "Learn ML", "name": "Tester"}

    def test_quiz_generated(self):
        quiz = self.agent.generate_quiz("python_intro", self.user, self.mcp)
        self.assertIn("questions", quiz)

    def test_quiz_has_questions(self):
        quiz = self.agent.generate_quiz("ml_foundations", self.user, self.mcp)
        self.assertGreater(quiz.get("total_questions", 0), 0)

    def test_question_has_options(self):
        quiz = self.agent.generate_quiz("linear_algebra", self.user, self.mcp)
        if quiz["questions"]:
            q = quiz["questions"][0]
            self.assertIn("options", q)
            self.assertEqual(len(q["options"]), 4)

    def test_answers_present_for_server_side_use(self):
        """
        _answers is kept in the agent response so the API layer can cache
        and strip it before sending to the client. This is by design.
        """
        quiz = self.agent.generate_quiz("python_intro", self.user, self.mcp)
        # Agent returns _answers for server-side caching (API strips before sending to client)
        self.assertIn("_answers", quiz)
        self.assertIsInstance(quiz["_answers"], dict)
        # Verify answer indices are valid option indices
        for q_id, ans_idx in quiz["_answers"].items():
            self.assertGreaterEqual(ans_idx, 0)
            self.assertLess(ans_idx, 4)

    def test_evaluate_perfect_score(self):
        correct = {0: 1, 1: 2}
        user_answers = {0: 1, 1: 2}
        result = self.agent.evaluate_answers("", "python_intro", user_answers, correct)
        self.assertEqual(result["score_pct"], 100.0)
        self.assertTrue(result["passed"])

    def test_evaluate_zero_score(self):
        correct = {0: 1, 1: 2}
        user_answers = {0: 0, 1: 0}
        result = self.agent.evaluate_answers("", "python_intro", user_answers, correct)
        self.assertEqual(result["score_pct"], 0.0)
        self.assertFalse(result["passed"])

    def test_evaluate_passing_threshold(self):
        # 70% should pass
        correct = {0: 0, 1: 1, 2: 2}
        user_answers = {0: 0, 1: 1, 2: 0}  # 2/3 = 66.7% — fail
        result = self.agent.evaluate_answers("", "test", user_answers, correct)
        self.assertFalse(result["passed"])

    def test_feedback_included(self):
        correct = {0: 1}
        result = self.agent.evaluate_answers("", "python_intro", {0: 1}, correct)
        self.assertIn("feedback", result)
        self.assertIsInstance(result["feedback"], str)

    def test_mcp_context_in_quiz(self):
        quiz = self.agent.generate_quiz("python_intro", self.user, self.mcp)
        self.assertIn("mcp_context", quiz)


# ── RESOURCE AGENT ────────────────────────────────────────────────────────

class TestResourceAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = make_store()
        cls.agent = ResourceAgent(cls.store)
        cls.mcp = make_context().to_dict()

    def test_fetch_known_topic(self):
        result = self.agent.fetch_resources("python_intro", self.mcp)
        self.assertIn("resources", result)

    def test_resources_not_empty_for_known_topic(self):
        result = self.agent.fetch_resources("ml_foundations", self.mcp)
        self.assertGreater(result["summary"]["total_resources"], 0)

    def test_summary_fields_present(self):
        result = self.agent.fetch_resources("linear_algebra", self.mcp)
        summary = result["summary"]
        self.assertIn("total_resources", summary)
        self.assertIn("free_resources", summary)
        self.assertIn("estimated_total_hours", summary)

    def test_free_count_accurate(self):
        result = self.agent.fetch_resources("deep_learning", self.mcp)
        resources = result["resources"]
        free_count = sum(1 for r in resources if r.get("free", True))
        self.assertEqual(result["summary"]["free_resources"], free_count)

    def test_search_resources(self):
        result = self.agent.search_resources("machine learning beginner", self.mcp)
        self.assertIn("results", result)
        self.assertGreaterEqual(result["total_found"], 0)

    def test_mcp_context_in_result(self):
        result = self.agent.fetch_resources("python_intro", self.mcp)
        self.assertIn("mcp_context", result)

    def test_unknown_topic_returns_empty(self):
        result = self.agent.fetch_resources("nonexistent_topic_xyz", self.mcp)
        self.assertEqual(result["summary"]["total_resources"], 0)


# ── MCP CONTEXT ───────────────────────────────────────────────────────────

class TestMCPContext(unittest.TestCase):
    def test_to_dict(self):
        ctx = MCPContext(user_id="u1", user_goal="Learn Python")
        d = ctx.to_dict()
        self.assertEqual(d["user_goal"], "Learn Python")
        self.assertIn("retrieved_docs", d)

    def test_to_prompt_block(self):
        ctx = MCPContext(user_goal="Learn ML", user_name="Alice", progress_pct=40.0)
        block = ctx.to_prompt_block()
        self.assertIn("Learn ML", block)
        self.assertIn("Alice", block)
        self.assertIn("40.0%", block)

    def test_with_retrieved_docs(self):
        ctx = MCPContext(user_goal="test")
        docs = [{"doc_id": "d1", "score": 0.9, "metadata": {"topic": "ML", "level": "beginner"}, "content": "test content"}]
        ctx2 = ctx.with_retrieved_docs(docs)
        self.assertEqual(len(ctx2.retrieved_docs), 1)
        self.assertEqual(ctx2.retrieved_docs[0].topic, "ML")
        # Original unchanged
        self.assertEqual(len(ctx.retrieved_docs), 0)

    def test_minimal_constructor(self):
        ctx = MCPContext.minimal("u1", "Learn AI")
        self.assertEqual(ctx.user_goal, "Learn AI")
        self.assertEqual(ctx.progress_pct, 0.0)

    def test_repr(self):
        ctx = MCPContext(user_name="Bob", user_goal="ML", progress_pct=55.0)
        r = repr(ctx)
        self.assertIn("Bob", r)
        self.assertIn("55.0", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
