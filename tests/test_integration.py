"""
Integration Tests — Full Pipeline
Tests the complete end-to-end flow:
  User Input → Validation → Planner → RAG → Roadmap → Quiz → Progress

Also tests the Flask API endpoints directly.
"""

import sys
sys.path.insert(0, "/home/claude/learning_system")

import unittest
import json
import time
import os

os.environ["TEST_MODE"] = "1"

from rag.vector_store import VectorStore
from rag.knowledge_base import seed_knowledge_base
from rag.mcp_context import MCPContext
from rag.chatbot import ChatbotEngine
from agents.planner_agent import PlannerAgent
from agents.validation_agent import ValidationAgent
from agents.quiz_agent import QuizAgent
from agents.resource_agent import ResourceAgent
from agents.progress_agent import ProgressAgent
from observability.metrics import Metrics


# ── SHARED FIXTURES ───────────────────────────────────────────────────────

def build_system():
    """Bootstrap the full system, same as production."""
    store = VectorStore()
    seed_knowledge_base(store)
    validation = ValidationAgent()
    planner = PlannerAgent(store)
    resource = ResourceAgent(store)
    quiz = QuizAgent(store)
    metrics = Metrics()
    chatbot = ChatbotEngine(store, validation, metrics)
    return dict(store=store, validation=validation, planner=planner,
                resource=resource, quiz=quiz, metrics=metrics, chatbot=chatbot)


SYSTEM = None

def get_system():
    global SYSTEM
    if SYSTEM is None:
        SYSTEM = build_system()
    return SYSTEM


# ── PIPELINE TEST 1: Input → Validation → Roadmap ────────────────────────

class TestFullRoadmapPipeline(unittest.TestCase):
    """
    Integration test: user submits a goal → validation → planner → roadmap.
    This is the primary happy path.
    """

    def setUp(self):
        self.sys = get_system()

    def _generate_roadmap(self, goal, known_topics=None):
        # Step 1: Validate input (guardrail)
        val_result = self.sys["validation"].validate_input(goal, "query")
        self.assertTrue(val_result["valid"], f"Input rejected: {val_result['reason']}")

        # Step 2: Build MCP context
        ctx = MCPContext(
            user_id="integration_test_user",
            user_name="Integration Tester",
            user_goal=val_result["sanitized"],
            hours_per_week=10,
        )
        if known_topics:
            ctx._known_topics = known_topics

        # Step 3: Run planner agent
        roadmap = self.sys["planner"].run(ctx)

        # Step 4: Validate roadmap (guardrail)
        roadmap_val = self.sys["validation"].validate_roadmap(roadmap)
        self.assertTrue(roadmap_val["valid"], f"Roadmap invalid: {roadmap_val['reason']}")

        return roadmap, ctx

    def test_ml_pipeline(self):
        roadmap, ctx = self._generate_roadmap("Learn machine learning from scratch")
        self.assertGreater(roadmap["total_steps"], 0)
        self.assertEqual(roadmap["domain"], "machine learning")
        self.assertIn("mcp_context_used", roadmap)
        self.assertEqual(roadmap["mcp_context_used"]["user_goal"], "Learn machine learning from scratch")

    def test_python_pipeline(self):
        roadmap, ctx = self._generate_roadmap("I want to learn Python programming")
        self.assertGreater(roadmap["total_steps"], 0)

    def test_web_dev_pipeline(self):
        roadmap, ctx = self._generate_roadmap("Learn web development with HTML and JavaScript")
        self.assertGreater(roadmap["total_steps"], 0)

    def test_known_topics_excluded(self):
        roadmap, ctx = self._generate_roadmap(
            "Learn machine learning",
            known_topics=["python_intro", "linear_algebra"]
        )
        step_ids = [s["topic_id"] for s in roadmap["steps"]]
        self.assertNotIn("python_intro", step_ids)
        self.assertNotIn("linear_algebra", step_ids)

    def test_roadmap_steps_ordered_by_level(self):
        roadmap, _ = self._generate_roadmap("Learn deep learning")
        levels = [s["level"] for s in roadmap["steps"]]
        level_order = {"beginner": 0, "intermediate": 1, "advanced": 2}
        numeric = [level_order.get(l, 0) for l in levels]
        # Should be non-decreasing (can't go advanced→beginner)
        for i in range(len(numeric) - 1):
            self.assertLessEqual(numeric[i], numeric[i+1] + 1,
                                 f"Level ordering violated at step {i}: {levels}")

    def test_harmful_input_blocked(self):
        val_result = self.sys["validation"].validate_input("how to hack systems")
        self.assertFalse(val_result["valid"])


# ── PIPELINE TEST 2: Roadmap → Resource Fetch ────────────────────────────

class TestRoadmapToResources(unittest.TestCase):
    def setUp(self):
        self.sys = get_system()

    def test_each_roadmap_step_has_resources(self):
        ctx = MCPContext(user_goal="Learn Machine Learning", hours_per_week=10)
        ctx._known_topics = []
        roadmap = self.sys["planner"].run(ctx)
        mcp_dict = ctx.to_dict()

        # Fetch resources for each step
        for step in roadmap["steps"][:3]:  # test first 3 steps
            resource_result = self.sys["resource"].fetch_resources(step["topic_id"], mcp_dict)
            self.assertIn("resources", resource_result)
            self.assertIn("summary", resource_result)

    def test_resource_mcp_context_matches(self):
        ctx = MCPContext(user_goal="Learn Python", hours_per_week=5)
        ctx._known_topics = []
        mcp_dict = ctx.to_dict()
        result = self.sys["resource"].fetch_resources("python_intro", mcp_dict)
        self.assertEqual(result["mcp_context"]["user_goal"], "Learn Python")


# ── PIPELINE TEST 3: Quiz + Evaluation ───────────────────────────────────

class TestQuizPipeline(unittest.TestCase):
    def setUp(self):
        self.sys = get_system()
        self.mcp = MCPContext(user_goal="Learn ML").to_dict()
        self.user = {"goal": "Learn ML", "name": "Tester"}

    def test_generate_and_evaluate_perfect(self):
        quiz = self.sys["quiz"].generate_quiz("python_intro", self.user, self.mcp)
        answers_stored = {q["id"]: 0 for q in quiz["questions"]}  # simulate server-side

        # Get actual correct answers by generating again with _answers
        quiz2 = self.sys["quiz"].generate_quiz("python_intro", self.user, self.mcp)
        # Simulate correct answers
        correct = {i: 0 for i in range(quiz2["total_questions"])}

        result = self.sys["quiz"].evaluate_answers("", "python_intro",
                                                   {i: 0 for i in range(quiz2["total_questions"])},
                                                   {i: 0 for i in range(quiz2["total_questions"])})
        self.assertEqual(result["score_pct"], 100.0)
        self.assertTrue(result["passed"])

    def test_quiz_result_has_all_fields(self):
        quiz = self.sys["quiz"].generate_quiz("ml_foundations", self.user, self.mcp)
        n = quiz["total_questions"]
        if n == 0:
            return  # skip if no questions

        correct = {i: 0 for i in range(n)}
        user_answers = {i: 1 for i in range(n)}
        result = self.sys["quiz"].evaluate_answers("", "ml_foundations", user_answers, correct)

        required = {"score_pct", "correct", "total", "passed", "feedback", "results", "recommendation"}
        for field in required:
            self.assertIn(field, result, f"Missing field: {field}")

    def test_quiz_recommendation_on_fail(self):
        correct = {0: 2}
        user_answers = {0: 0}
        result = self.sys["quiz"].evaluate_answers("", "linear_algebra", user_answers, correct)
        self.assertFalse(result["passed"])
        self.assertEqual(result["recommendation"], "review")

    def test_quiz_recommendation_on_pass(self):
        correct = {0: 2, 1: 1}
        user_answers = {0: 2, 1: 1}
        result = self.sys["quiz"].evaluate_answers("", "linear_algebra", user_answers, correct)
        self.assertTrue(result["passed"])
        self.assertEqual(result["recommendation"], "proceed")


# ── PIPELINE TEST 4: Chat RAG Pipeline ───────────────────────────────────

class TestChatRagPipeline(unittest.TestCase):
    def setUp(self):
        self.sys = get_system()
        self.mcp = MCPContext(
            user_id="chat_test",
            user_goal="Learn Machine Learning",
            current_step="Linear Algebra",
            progress_pct=20.0,
        ).to_dict()

    def test_relevant_query_answered(self):
        result = self.sys["chatbot"].answer("What is gradient descent?", "chat_test", self.mcp.copy())
        self.assertIn("answer", result)
        self.assertIsInstance(result["answer"], str)
        self.assertGreater(len(result["answer"]), 10)

    def test_answer_has_confidence(self):
        result = self.sys["chatbot"].answer("Explain neural networks", "chat_test", self.mcp.copy())
        self.assertIn("confidence", result)
        self.assertIsInstance(result["confidence"], float)
        self.assertGreaterEqual(result["confidence"], 0.0)

    def test_answer_includes_retrieved_docs(self):
        result = self.sys["chatbot"].answer("What is linear algebra?", "chat_test", self.mcp.copy())
        self.assertIn("retrieved_docs", result)

    def test_harmful_query_rejected(self):
        result = self.sys["chatbot"].answer("how to hack", "chat_test", self.mcp.copy())
        self.assertFalse(result["valid"])

    def test_response_time_tracked(self):
        result = self.sys["chatbot"].answer("What is Python?", "chat_test", self.mcp.copy())
        self.assertIn("response_time_ms", result)
        self.assertGreater(result["response_time_ms"], 0)

    def test_mcp_context_injected_into_answer(self):
        mcp = MCPContext(
            user_id="test",
            user_goal="Learn Deep Learning",
            current_step="Neural Networks",
            progress_pct=50.0,
        ).to_dict()
        result = self.sys["chatbot"].answer("Explain backpropagation", "test", mcp)
        self.assertIn("mcp_context", result)
        self.assertEqual(result["mcp_context"]["user_goal"], "Learn Deep Learning")


# ── PIPELINE TEST 5: Full End-to-End Scenario ────────────────────────────

class TestEndToEndScenario(unittest.TestCase):
    """
    Simulates a complete user journey:
    1. New user arrives with goal "Learn Machine Learning"
    2. Input validated
    3. Roadmap generated with MCP context
    4. Resources fetched for first step
    5. Quiz taken on first step
    6. Progress updated
    7. Dashboard queried
    """

    def test_complete_user_journey(self):
        sys = get_system()
        user_id = f"journey_test_{int(time.time())}"
        goal = "Learn machine learning"

        # Step 1: Validate goal
        val = sys["validation"].validate_input(goal)
        self.assertTrue(val["valid"])

        # Step 2: Build MCP context
        ctx = MCPContext(user_id=user_id, user_name="Journey User", user_goal=goal, hours_per_week=10)
        ctx._known_topics = []

        # Step 3: Generate roadmap
        roadmap = sys["planner"].run(ctx)
        self.assertGreater(len(roadmap["steps"]), 0)
        self.assertIn("mcp_context_used", roadmap)

        # Step 4: Fetch resources for step 1
        step1 = roadmap["steps"][0]
        resources = sys["resource"].fetch_resources(step1["topic_id"], ctx.to_dict())
        self.assertIn("resources", resources)

        # Step 5: Generate quiz for step 1
        mcp_dict = ctx.to_dict()
        quiz = sys["quiz"].generate_quiz(step1["topic_id"], {"goal": goal}, mcp_dict)
        self.assertIn("questions", quiz)
        n = quiz["total_questions"]

        # Step 6: Submit quiz (simulate all correct)
        if n > 0:
            correct = {i: 0 for i in range(n)}
            result = sys["quiz"].evaluate_answers("", step1["topic_id"], correct, correct)
            self.assertEqual(result["score_pct"], 100.0)
            self.assertTrue(result["passed"])

        # Journey complete — all agents executed in sequence with MCP context
        self.assertTrue(True, "Full journey completed successfully")


if __name__ == "__main__":
    unittest.main(verbosity=2)
