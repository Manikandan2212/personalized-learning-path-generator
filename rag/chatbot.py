"""
Chatbot Q&A Engine
Answers user questions using RAG retrieval + MCP context injection.
Implements output guardrails and confidence thresholding.
"""

import logging
import time
from typing import Dict, List

logger = logging.getLogger("rag.chatbot")


SYSTEM_PROMPT_TEMPLATE = """You are an expert personalized learning assistant.
Answer only from the retrieved knowledge base context below.
If the answer is not in the context, say: "I couldn't find that in the knowledge base. Try asking about Python, Machine Learning, Deep Learning, Statistics, or Web Development."

User Context (MCP):
- Goal: {user_goal}
- Current Step: {current_step}
- Progress: {progress_pct}% complete
- Completed Topics: {completed_topics} of {total_topics}

Retrieved Knowledge:
{retrieved_context}

Instructions:
- Be concise and educational
- Relate answer to the user's current learning step when relevant
- Do NOT invent information not present in the retrieved context
- End with a practical next step or tip
"""


def format_context(retrieved_docs: List[Dict]) -> str:
    if not retrieved_docs:
        return "No relevant documents retrieved."
    parts = []
    for i, doc in enumerate(retrieved_docs[:4], 1):
        topic = doc["metadata"].get("topic", "")
        level = doc["metadata"].get("level", "")
        parts.append(f"[{i}] ({topic} - {level}):\n{doc['content'][:500]}")
    return "\n\n".join(parts)


def rule_based_answer(query: str, retrieved_docs: List[Dict], mcp_context: Dict) -> str:
    """
    Rule-based answer generation from retrieved documents.
    Used when no external LLM API is available.
    Synthesizes a coherent response from top retrieved chunks.
    """
    if not retrieved_docs:
        return (
            "I couldn't find relevant information for your query in the knowledge base. "
            "Try asking about: Python, Machine Learning, Deep Learning, Linear Algebra, "
            "Statistics, Neural Networks, Transformers, RAG, or Web Development."
        )

    top_doc = retrieved_docs[0]
    topic = top_doc["metadata"].get("topic", "this topic")
    level = top_doc["metadata"].get("level", "")
    content = top_doc["content"]

    # Build synthesized answer
    user_goal = mcp_context.get("user_goal", "")
    current_step = mcp_context.get("current_step", "")

    answer_parts = [f"**{topic}** ({level} level)\n"]
    answer_parts.append(content[:600])

    if len(retrieved_docs) > 1:
        answer_parts.append(f"\n\n**Additional context:**")
        for doc in retrieved_docs[1:3]:
            extra_topic = doc["metadata"].get("topic", "")
            if extra_topic != topic:
                answer_parts.append(f"\n• *{extra_topic}*: {doc['content'][:200]}")

    # Add context-aware next step
    if current_step and current_step != "Not started":
        answer_parts.append(f"\n\n**Your current step:** {current_step}")

    resources = top_doc["metadata"].get("resources", [])
    if resources:
        answer_parts.append(f"\n\n**Recommended resource:** [{resources[0]['title']}]({resources[0]['url']})")

    prereqs = top_doc["metadata"].get("prerequisites", [])
    if prereqs:
        prereq_names = [p.replace("_", " ").title() for p in prereqs]
        answer_parts.append(f"\n\n**Prerequisites:** {', '.join(prereq_names)}")

    return "\n".join(answer_parts)


class ChatbotEngine:
    def __init__(self, vector_store, validation_agent, metrics):
        self.vector_store = vector_store
        self.validation_agent = validation_agent
        self.metrics = metrics
        self.name = "ChatbotEngine"

    def answer(self, query: str, user_id: str, mcp_context: Dict) -> Dict:
        t0 = time.time()
        logger.info(f"ChatbotEngine: query='{query[:60]}' user='{user_id}'")

        # Input validation
        validation = self.validation_agent.validate_input(query, input_type="query")
        if not validation["valid"]:
            return {
                "answer": validation["reason"],
                "valid": False,
                "confidence": 0.0,
                "retrieved_docs": [],
                "mcp_context": mcp_context,
            }

        sanitized_query = validation["sanitized"]

        # RAG retrieval
        rag_t0 = time.time()
        retrieved = self.vector_store.search(sanitized_query, top_k=5, min_score=0.0)
        rag_time = (time.time() - rag_t0) * 1000

        # Log RAG metrics
        self.metrics.log_rag(user_id, sanitized_query, retrieved, rag_time)

        # Inject retrieved docs into MCP context
        mcp_context["retrieved_docs"] = [
            {"doc_id": r["doc_id"], "score": r["score"], "topic": r["metadata"].get("topic")}
            for r in retrieved
        ]

        # Generate answer using rule-based RAG synthesis
        raw_answer = rule_based_answer(sanitized_query, retrieved, mcp_context)

        # Output validation
        output_validation = self.validation_agent.validate_output(raw_answer, retrieved, sanitized_query)
        if not output_validation["valid"]:
            final_answer = output_validation["fallback_message"]
            confidence = 0.0
        else:
            final_answer = output_validation["response"]
            confidence = output_validation["confidence"]

        total_time = (time.time() - t0) * 1000
        self.metrics.log_request(user_id, "/chat", "POST", sanitized_query, "ChatbotEngine", total_time, 200)
        self.metrics.log_agent(user_id, "ChatbotEngine", "answer", sanitized_query[:100],
                               final_answer[:100], total_time, True)

        logger.info(f"ChatbotEngine: answered in {total_time:.0f}ms, confidence={confidence:.3f}")

        return {
            "answer": final_answer,
            "valid": True,
            "confidence": round(confidence, 3),
            "retrieved_docs": [
                {
                    "doc_id": r["doc_id"],
                    "topic": r["metadata"].get("topic"),
                    "level": r["metadata"].get("level"),
                    "score": r["score"],
                    "snippet": r["content"][:120] + "...",
                }
                for r in retrieved[:3]
            ],
            "mcp_context": mcp_context,
            "response_time_ms": round(total_time, 1),
            "warnings": validation.get("warnings", []),
        }
