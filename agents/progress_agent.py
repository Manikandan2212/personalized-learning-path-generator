"""
Progress Agent
Tracks user progress across topics, updates performance metrics,
and persists state to SQLite database.
"""

import sqlite3
import json
import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger("agents.progress")

DB_PATH = "progress.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            goal TEXT,
            hours_per_week INTEGER DEFAULT 10,
            created_at REAL,
            updated_at REAL
        );

        CREATE TABLE IF NOT EXISTS topic_progress (
            user_id TEXT,
            topic_id TEXT,
            status TEXT DEFAULT 'not_started',
            score_pct REAL DEFAULT 0,
            attempts INTEGER DEFAULT 0,
            started_at REAL,
            completed_at REAL,
            PRIMARY KEY (user_id, topic_id)
        );

        CREATE TABLE IF NOT EXISTS roadmaps (
            user_id TEXT PRIMARY KEY,
            roadmap_json TEXT,
            created_at REAL
        );

        CREATE TABLE IF NOT EXISTS quiz_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            topic_id TEXT,
            score_pct REAL,
            correct INTEGER,
            total INTEGER,
            passed INTEGER,
            taken_at REAL
        );
    """)
    conn.commit()
    conn.close()
    logger.info("ProgressAgent: database initialized")


class ProgressAgent:
    def __init__(self):
        self.name = "ProgressAgent"
        init_db()

    # ── USER MANAGEMENT ─────────────────────────────────────────────────

    def get_or_create_user(self, user_id: str, name: str = "Learner", goal: str = "", hours_per_week: int = 10) -> Dict:
        conn = get_connection()
        cur = conn.cursor()
        now = time.time()

        cur.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cur.fetchone()

        if row:
            conn.close()
            return dict(row)

        cur.execute(
            "INSERT INTO users (user_id, name, goal, hours_per_week, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (user_id, name, goal, hours_per_week, now, now),
        )
        conn.commit()
        conn.close()
        logger.info(f"ProgressAgent: created user '{user_id}'")
        return {"user_id": user_id, "name": name, "goal": goal, "hours_per_week": hours_per_week, "created_at": now, "updated_at": now}

    def update_user(self, user_id: str, updates: Dict) -> Dict:
        conn = get_connection()
        cur = conn.cursor()
        allowed = {"name", "goal", "hours_per_week"}
        fields = {k: v for k, v in updates.items() if k in allowed}
        fields["updated_at"] = time.time()

        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [user_id]
        cur.execute(f"UPDATE users SET {set_clause} WHERE user_id = ?", values)
        conn.commit()
        cur.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else {}

    # ── ROADMAP ─────────────────────────────────────────────────────────

    def save_roadmap(self, user_id: str, roadmap: Dict):
        conn = get_connection()
        cur = conn.cursor()
        now = time.time()
        cur.execute(
            "INSERT OR REPLACE INTO roadmaps (user_id, roadmap_json, created_at) VALUES (?,?,?)",
            (user_id, json.dumps(roadmap), now),
        )
        # Init topic progress rows
        for step in roadmap.get("steps", []):
            topic_id = step["topic_id"]
            cur.execute(
                "INSERT OR IGNORE INTO topic_progress (user_id, topic_id, status, score_pct, attempts) VALUES (?,?,'not_started',0,0)",
                (user_id, topic_id),
            )
        conn.commit()
        conn.close()
        logger.info(f"ProgressAgent: roadmap saved for user '{user_id}'")

    def get_roadmap(self, user_id: str) -> Optional[Dict]:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT roadmap_json FROM roadmaps WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            return json.loads(row["roadmap_json"])
        return None

    # ── TOPIC PROGRESS ──────────────────────────────────────────────────

    def update_topic_status(self, user_id: str, topic_id: str, status: str) -> Dict:
        conn = get_connection()
        cur = conn.cursor()
        now = time.time()

        cur.execute(
            "INSERT OR IGNORE INTO topic_progress (user_id, topic_id, status) VALUES (?,?,?)",
            (user_id, topic_id, "not_started"),
        )

        if status == "in_progress":
            cur.execute(
                "UPDATE topic_progress SET status=?, started_at=? WHERE user_id=? AND topic_id=?",
                (status, now, user_id, topic_id),
            )
        elif status == "completed":
            cur.execute(
                "UPDATE topic_progress SET status=?, completed_at=? WHERE user_id=? AND topic_id=?",
                (status, now, user_id, topic_id),
            )
        else:
            cur.execute(
                "UPDATE topic_progress SET status=? WHERE user_id=? AND topic_id=?",
                (status, user_id, topic_id),
            )
        conn.commit()
        conn.close()
        logger.info(f"ProgressAgent: {user_id} / {topic_id} → {status}")
        return self.get_topic_progress(user_id, topic_id)

    def get_topic_progress(self, user_id: str, topic_id: str) -> Dict:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM topic_progress WHERE user_id=? AND topic_id=?", (user_id, topic_id))
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else {"user_id": user_id, "topic_id": topic_id, "status": "not_started", "score_pct": 0, "attempts": 0}

    def record_quiz_result(self, user_id: str, topic_id: str, score_pct: float, correct: int, total: int, passed: bool):
        conn = get_connection()
        cur = conn.cursor()
        now = time.time()

        cur.execute(
            "INSERT INTO quiz_history (user_id, topic_id, score_pct, correct, total, passed, taken_at) VALUES (?,?,?,?,?,?,?)",
            (user_id, topic_id, score_pct, correct, total, int(passed), now),
        )
        cur.execute(
            "INSERT OR IGNORE INTO topic_progress (user_id, topic_id, status) VALUES (?,?,'not_started')",
            (user_id, topic_id),
        )
        cur.execute(
            "UPDATE topic_progress SET score_pct=?, attempts=attempts+1 WHERE user_id=? AND topic_id=?",
            (score_pct, user_id, topic_id),
        )
        if passed:
            cur.execute(
                "UPDATE topic_progress SET status='completed', completed_at=? WHERE user_id=? AND topic_id=?",
                (now, user_id, topic_id),
            )
        conn.commit()
        conn.close()
        logger.info(f"ProgressAgent: quiz recorded {user_id}/{topic_id} score={score_pct}%")

    # ── DASHBOARD ───────────────────────────────────────────────────────

    def get_dashboard(self, user_id: str) -> Dict:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
        user = cur.fetchone()

        cur.execute("SELECT * FROM topic_progress WHERE user_id=?", (user_id,))
        topic_rows = cur.fetchall()

        cur.execute("SELECT * FROM quiz_history WHERE user_id=? ORDER BY taken_at DESC LIMIT 10", (user_id,))
        quiz_rows = cur.fetchall()

        conn.close()

        topics = [dict(r) for r in topic_rows]
        completed = [t for t in topics if t["status"] == "completed"]
        in_progress = [t for t in topics if t["status"] == "in_progress"]
        not_started = [t for t in topics if t["status"] == "not_started"]
        total = len(topics)

        avg_score = 0.0
        if completed:
            scores = [t["score_pct"] for t in completed if t["score_pct"] > 0]
            avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0

        overall_pct = round((len(completed) / max(total, 1)) * 100, 1)

        return {
            "user": dict(user) if user else {"user_id": user_id},
            "summary": {
                "total_topics": total,
                "completed": len(completed),
                "in_progress": len(in_progress),
                "not_started": len(not_started),
                "overall_progress_pct": overall_pct,
                "average_quiz_score": avg_score,
            },
            "topics": topics,
            "recent_quiz_history": [dict(r) for r in quiz_rows],
            "current_topic": in_progress[0]["topic_id"] if in_progress else (
                not_started[0]["topic_id"] if not_started else None
            ),
        }

    def build_mcp_context(self, user_id: str) -> Dict:
        """Build the MCP structured context for LLM calls."""
        dashboard = self.get_dashboard(user_id)
        user = dashboard.get("user", {})
        summary = dashboard.get("summary", {})

        current_topic = dashboard.get("current_topic")
        roadmap = self.get_roadmap(user_id)

        current_step_title = None
        if current_topic and roadmap:
            for step in roadmap.get("steps", []):
                if step["topic_id"] == current_topic:
                    current_step_title = step["title"]
                    break

        return {
            "user_goal": user.get("goal", ""),
            "user_name": user.get("name", "Learner"),
            "current_step": current_step_title or current_topic or "Not started",
            "progress_pct": summary.get("overall_progress_pct", 0),
            "completed_topics": summary.get("completed", 0),
            "total_topics": summary.get("total_topics", 0),
            "average_score": summary.get("average_quiz_score", 0),
            "retrieved_docs": [],  # filled in by RAG at query time
        }
