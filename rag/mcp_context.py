"""
MCP — Model Context Protocol
============================
Defines the structured context payload passed to every agent and LLM call.
This is the single source of truth for what context is available to all agents.

When a user asks "Why am I learning CNN?", we don't just send the question.
We send the full structured context:

{
    "user_goal": "Learn Deep Learning",
    "user_name": "Priya",
    "current_step": "Linear Algebra",
    "progress_pct": 40.0,
    "completed_topics": 2,
    "total_topics": 8,
    "average_score": 85.0,
    "retrieved_docs": [
        {"doc_id": "linear_algebra_chunk_0", "score": 0.91, "topic": "Mathematics"}
    ]
}

This structured payload = MCP usage.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import time


@dataclass
class RetrievedDoc:
    doc_id: str
    score: float
    topic: str
    level: str = "unknown"
    snippet: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MCPContext:
    """
    Structured context injected into every agent call and LLM prompt.
    Follows the Model Context Protocol pattern: pass rich, typed context
    rather than raw strings.
    """
    # User identity & goal
    user_id: str = "anonymous"
    user_name: str = "Learner"
    user_goal: str = ""

    # Current learning position
    current_step: str = "Not started"
    current_topic_id: str = ""
    progress_pct: float = 0.0

    # Performance stats
    completed_topics: int = 0
    total_topics: int = 0
    average_score: float = 0.0
    hours_per_week: int = 10

    # RAG: filled at query time by the chatbot/agents
    retrieved_docs: List[RetrievedDoc] = field(default_factory=list)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    request_type: str = "query"   # query | roadmap | quiz | resource

    # ── BUILDERS ────────────────────────────────────────────────────

    @classmethod
    def from_progress(cls, user_id: str, progress_data: Dict, roadmap: Optional[Dict] = None) -> "MCPContext":
        """Build MCP context from progress agent dashboard data."""
        user = progress_data.get("user", {})
        summary = progress_data.get("summary", {})
        current_topic_id = progress_data.get("current_topic", "")

        # Resolve human-readable step name from roadmap
        current_step = current_topic_id.replace("_", " ").title() if current_topic_id else "Not started"
        if roadmap:
            for step in roadmap.get("steps", []):
                if step["topic_id"] == current_topic_id:
                    current_step = step["title"]
                    break

        return cls(
            user_id=user_id,
            user_name=user.get("name", "Learner"),
            user_goal=user.get("goal", ""),
            current_step=current_step,
            current_topic_id=current_topic_id,
            progress_pct=summary.get("overall_progress_pct", 0.0),
            completed_topics=summary.get("completed", 0),
            total_topics=summary.get("total_topics", 0),
            average_score=summary.get("average_quiz_score", 0.0),
            hours_per_week=user.get("hours_per_week", 10),
        )

    @classmethod
    def minimal(cls, user_id: str, goal: str) -> "MCPContext":
        """Minimal context for new users before any progress exists."""
        return cls(user_id=user_id, user_goal=goal)

    # ── MUTATION ────────────────────────────────────────────────────

    def with_retrieved_docs(self, docs: List[Dict]) -> "MCPContext":
        """Return a new context with retrieved docs injected (immutable-style)."""
        import copy
        ctx = copy.copy(self)
        ctx.retrieved_docs = [
            RetrievedDoc(
                doc_id=d.get("doc_id", ""),
                score=d.get("score", 0.0),
                topic=d.get("metadata", d).get("topic", d.get("topic", "")),
                level=d.get("metadata", d).get("level", d.get("level", "unknown")),
                snippet=d.get("content", d.get("snippet", ""))[:150],
            )
            for d in docs
        ]
        return ctx

    def with_request_type(self, request_type: str) -> "MCPContext":
        import copy
        ctx = copy.copy(self)
        ctx.request_type = request_type
        return ctx

    # ── SERIALIZATION ────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        """Serialize to plain dict for JSON responses and logging."""
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "user_goal": self.user_goal,
            "current_step": self.current_step,
            "current_topic_id": self.current_topic_id,
            "progress_pct": self.progress_pct,
            "completed_topics": self.completed_topics,
            "total_topics": self.total_topics,
            "average_score": self.average_score,
            "hours_per_week": self.hours_per_week,
            "retrieved_docs": [d.to_dict() for d in self.retrieved_docs],
            "timestamp": self.timestamp,
            "request_type": self.request_type,
        }

    def to_prompt_block(self) -> str:
        """
        Format MCP context as a structured block for LLM system prompts.
        This is injected before every LLM call.
        """
        docs_block = ""
        if self.retrieved_docs:
            docs_block = "\nRetrieved Knowledge Base Documents:\n"
            for i, doc in enumerate(self.retrieved_docs[:4], 1):
                docs_block += f"  [{i}] (topic={doc.topic}, level={doc.level}, score={doc.score:.3f}): {doc.snippet}\n"
        else:
            docs_block = "\nRetrieved Documents: none\n"

        return f"""
=== MCP STRUCTURED CONTEXT ===
User: {self.user_name} (id={self.user_id})
Goal: {self.user_goal}
Current Step: {self.current_step}
Progress: {self.progress_pct}% ({self.completed_topics}/{self.total_topics} topics)
Average Quiz Score: {self.average_score}%
Request Type: {self.request_type}
{docs_block}
================================"""

    def __repr__(self) -> str:
        return (
            f"MCPContext(user={self.user_name!r}, goal={self.user_goal!r}, "
            f"step={self.current_step!r}, progress={self.progress_pct}%)"
        )
