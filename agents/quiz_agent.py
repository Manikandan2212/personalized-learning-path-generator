"""
Quiz Agent
Generates quiz questions from knowledge base and evaluates user answers.
Tracks performance and adapts difficulty.
"""

import logging
import random
from typing import List, Dict, Optional

logger = logging.getLogger("agents.quiz")


class QuizAgent:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.name = "QuizAgent"

    def generate_quiz(self, topic_id: str, user_profile: Dict, mcp_context: Dict) -> Dict:
        from rag.knowledge_base import get_quiz_questions, get_full_entry
        logger.info(f"QuizAgent: generating quiz for topic='{topic_id}'")

        questions = get_quiz_questions(topic_id)
        entry = get_full_entry(topic_id)

        if not questions:
            # Generate generic comprehension questions from retrieved content
            questions = self._generate_from_rag(topic_id)

        if not questions:
            return {
                "topic_id": topic_id,
                "title": topic_id.replace("_", " ").title(),
                "questions": [],
                "message": "No quiz available for this topic yet.",
            }

        # Shuffle options within each question (keep answer index synced)
        prepared = []
        for i, q in enumerate(questions):
            options = list(q["options"])
            correct_text = options[q["answer"]]
            random.shuffle(options)
            new_answer_idx = options.index(correct_text)
            prepared.append({
                "id": i,
                "question": q["q"],
                "options": options,
                "correct_index": new_answer_idx,
                "topic_id": topic_id,
            })

        quiz = {
            "topic_id": topic_id,
            "title": topic_id.replace("_", " ").title(),
            "level": entry["level"] if entry else "unknown",
            "questions": [
                {"id": q["id"], "question": q["question"], "options": q["options"]}
                for q in prepared
            ],
            "_answers": {q["id"]: q["correct_index"] for q in prepared},  # server-side only
            "total_questions": len(prepared),
            "mcp_context": {
                "user_goal": mcp_context.get("user_goal"),
                "current_step": mcp_context.get("current_step"),
                "progress_pct": mcp_context.get("progress_pct"),
            },
        }

        logger.info(f"QuizAgent: generated {len(prepared)} questions for '{topic_id}'")
        return quiz

    def evaluate_answers(self, quiz_id: str, topic_id: str, answers: Dict[int, int], correct_answers: Dict[int, int]) -> Dict:
        logger.info(f"QuizAgent: evaluating {len(answers)} answers for topic='{topic_id}'")

        results = []
        correct_count = 0
        for q_id, user_answer in answers.items():
            is_correct = (user_answer == correct_answers.get(q_id, -1))
            if is_correct:
                correct_count += 1
            results.append({
                "question_id": q_id,
                "user_answer": user_answer,
                "correct_answer": correct_answers.get(q_id),
                "is_correct": is_correct,
            })

        total = len(answers)
        score_pct = round((correct_count / max(total, 1)) * 100, 1)
        passed = score_pct >= 70

        # Feedback
        if score_pct == 100:
            feedback = "Perfect score! You've mastered this topic. Ready to move on."
        elif score_pct >= 80:
            feedback = "Great work! You have a solid understanding. Review any missed questions."
        elif score_pct >= 70:
            feedback = "Good job — you passed! Spend a bit more time on the questions you missed."
        elif score_pct >= 50:
            feedback = "Almost there! Review the topic material and try again."
        else:
            feedback = "Keep studying — revisit the topic and key resources before retrying."

        evaluation = {
            "topic_id": topic_id,
            "score_pct": score_pct,
            "correct": correct_count,
            "total": total,
            "passed": passed,
            "feedback": feedback,
            "results": results,
            "recommendation": "proceed" if passed else "review",
        }

        logger.info(f"QuizAgent: score={score_pct}%, passed={passed}")
        return evaluation

    def _generate_from_rag(self, topic_id: str) -> List[Dict]:
        """Generate basic comprehension questions from retrieved content."""
        query = topic_id.replace("_", " ")
        docs = self.vector_store.search(query, top_k=3, min_score=0.0)

        if not docs:
            return []

        # Simple template-based questions
        templates = [
            {
                "q": f"Which of the following best describes {topic_id.replace('_', ' ')}?",
                "options": [
                    "A programming language for databases only",
                    "A foundational concept used in AI and data analysis",
                    "A type of hardware component",
                    "A web browser extension",
                ],
                "answer": 1,
            },
            {
                "q": f"What is the primary purpose of studying {topic_id.replace('_', ' ')}?",
                "options": [
                    "To build games only",
                    "To create hardware circuits",
                    "To build knowledge for advanced AI and data topics",
                    "To design physical infrastructure",
                ],
                "answer": 2,
            },
        ]
        return templates
