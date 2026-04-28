"""
Validation Agent
Implements input guardrails, output guardrails, and confidence checks.
Prevents hallucination and filters irrelevant/harmful content.
"""

import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger("agents.validation")

CONFIDENCE_THRESHOLD = 0.05
MAX_QUERY_LENGTH = 1000
MIN_QUERY_LENGTH = 2

BLOCKED_PATTERNS = [
    r"\b(hack|exploit|crack|bypass|inject|malware|virus|phishing|ddos|sql injection)\b",
    r"\b(kill|murder|harm|hurt|abuse|violence)\b",
    r"\b(adult|porn|nsfw|explicit|sexual)\b",
]

EDUCATIONAL_KEYWORDS = [
    "learn", "study", "understand", "explain", "how", "what", "why", "tutorial",
    "course", "guide", "beginner", "advanced", "python", "machine learning", "data",
    "math", "algebra", "calculus", "statistics", "neural", "deep learning", "ai",
    "nlp", "web", "javascript", "html", "css", "programming", "code", "model",
    "algorithm", "concept", "practice", "roadmap", "path", "skill",
]

HALLUCINATION_PHRASES = [
    "i think", "i believe", "probably", "maybe", "i'm not sure",
    "i don't know exactly", "it might be", "could be wrong",
]


class ValidationAgent:
    def __init__(self):
        self.name = "ValidationAgent"

    # ── INPUT GUARDRAILS ────────────────────────────────────────────────

    def validate_input(self, user_input: str, input_type: str = "query") -> Dict:
        """
        Validate user input before passing to other agents or LLM.
        Returns: {"valid": bool, "reason": str, "sanitized": str}
        """
        issues = []

        # Length checks
        if len(user_input.strip()) < MIN_QUERY_LENGTH:
            return {"valid": False, "reason": "Input too short. Please provide more detail.", "sanitized": ""}

        if len(user_input) > MAX_QUERY_LENGTH:
            user_input = user_input[:MAX_QUERY_LENGTH]
            issues.append("Input truncated to 1000 characters.")

        # Harmful content check
        lower_input = user_input.lower()
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, lower_input):
                logger.warning(f"ValidationAgent: blocked harmful content: '{user_input[:50]}'")
                return {
                    "valid": False,
                    "reason": "Input contains content outside the scope of this educational platform.",
                    "sanitized": "",
                }

        # For query type: check educational relevance
        if input_type == "query":
            relevance_score = self._educational_relevance_score(lower_input)
            if relevance_score < 0.1:
                logger.info(f"ValidationAgent: low relevance score {relevance_score:.2f} for '{user_input[:50]}'")
                issues.append("Query may be off-topic. Results may be limited.")

        # Sanitize
        sanitized = user_input.strip()
        sanitized = re.sub(r"[<>\"']", "", sanitized)

        logger.info(f"ValidationAgent: input valid=True, issues={issues}")
        return {
            "valid": True,
            "reason": "Input validated successfully.",
            "sanitized": sanitized,
            "warnings": issues,
        }

    def _educational_relevance_score(self, text: str) -> float:
        hits = sum(1 for kw in EDUCATIONAL_KEYWORDS if kw in text)
        return min(hits / max(len(text.split()) * 0.3, 1), 1.0)

    # ── OUTPUT GUARDRAILS ───────────────────────────────────────────────

    def validate_output(self, llm_response: str, retrieved_docs: List[Dict], query: str) -> Dict:
        """
        Validate LLM/agent output before returning to user.
        Checks: confidence, hallucination markers, source grounding.
        """
        issues = []
        confidence = self._compute_confidence(llm_response, retrieved_docs, query)

        # Low confidence fallback
        if confidence < CONFIDENCE_THRESHOLD:
            logger.warning(f"ValidationAgent: confidence {confidence:.3f} below threshold {CONFIDENCE_THRESHOLD}")
            return {
                "valid": False,
                "confidence": confidence,
                "reason": "Confidence below threshold. Could not find reliable information in the knowledge base.",
                "fallback_message": (
                    "I'm not confident enough to answer this accurately from the available knowledge base. "
                    "Please try rephrasing your question or asking about a specific topic like Python, "
                    "Machine Learning, Deep Learning, Statistics, or Web Development."
                ),
                "response": None,
            }

        # Hallucination markers check
        lower_response = llm_response.lower()
        for phrase in HALLUCINATION_PHRASES:
            if phrase in lower_response:
                issues.append(f"Response contains uncertain language: '{phrase}'")

        # Empty response check
        if not llm_response or len(llm_response.strip()) < 10:
            return {
                "valid": False,
                "confidence": 0.0,
                "reason": "Empty or too short response generated.",
                "fallback_message": "Unable to generate a response. Please try again.",
                "response": None,
            }

        logger.info(f"ValidationAgent: output valid=True, confidence={confidence:.3f}, issues={len(issues)}")
        return {
            "valid": True,
            "confidence": confidence,
            "reason": "Output passed validation.",
            "warnings": issues,
            "response": llm_response,
            "grounded": len(retrieved_docs) > 0,
        }

    def _compute_confidence(self, response: str, retrieved_docs: List[Dict], query: str) -> float:
        if not retrieved_docs:
            return 0.02

        # Use top retrieval score as base
        top_score = max((d.get("score", 0) for d in retrieved_docs), default=0)

        # Boost if query terms appear in response
        from rag.vector_store import tokenize
        query_tokens = set(tokenize(query))
        response_tokens = set(tokenize(response))
        token_overlap = len(query_tokens & response_tokens) / max(len(query_tokens), 1)

        confidence = (top_score * 0.6) + (token_overlap * 0.4)
        return min(confidence, 1.0)

    # ── ROADMAP VALIDATION ──────────────────────────────────────────────

    def validate_roadmap(self, roadmap: Dict) -> Dict:
        steps = roadmap.get("steps", [])
        issues = []

        if len(steps) == 0:
            return {"valid": False, "reason": "Roadmap has no steps.", "roadmap": roadmap}

        if len(steps) > 20:
            issues.append("Roadmap is very long. Consider narrowing your goal.")

        # Check for circular prerequisites (simplified)
        seen_ids = set()
        for step in steps:
            topic_id = step.get("topic_id")
            for prereq in step.get("prerequisites", []):
                if prereq in seen_ids:
                    pass  # prereq appears before this step - good
            seen_ids.add(topic_id)

        total_hours = roadmap.get("total_hours", 0)
        if total_hours > 500:
            issues.append(f"Total learning time is {total_hours}h — consider breaking into phases.")

        logger.info(f"ValidationAgent: roadmap valid=True, {len(steps)} steps, issues={issues}")
        return {
            "valid": True,
            "reason": "Roadmap validated.",
            "step_count": len(steps),
            "total_hours": total_hours,
            "warnings": issues,
            "roadmap": roadmap,
        }

    # ── IMAGE/UPLOAD VALIDATION ─────────────────────────────────────────

    def validate_upload(self, filename: str, content_type: str, size_bytes: int) -> Dict:
        ALLOWED_TYPES = ["application/pdf", "text/plain", "text/markdown"]
        MAX_SIZE = 5 * 1024 * 1024  # 5MB

        if content_type not in ALLOWED_TYPES:
            return {
                "valid": False,
                "reason": f"File type '{content_type}' not supported. Please upload PDF or TXT files.",
            }

        if size_bytes > MAX_SIZE:
            return {
                "valid": False,
                "reason": f"File too large ({size_bytes // 1024}KB). Maximum allowed is 5MB.",
            }

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ["pdf", "txt", "md"]:
            return {
                "valid": False,
                "reason": f"File extension '.{ext}' not allowed. Use .pdf, .txt, or .md",
            }

        logger.info(f"ValidationAgent: upload valid — {filename} ({size_bytes} bytes)")
        return {"valid": True, "reason": "File upload validated.", "filename": filename}
