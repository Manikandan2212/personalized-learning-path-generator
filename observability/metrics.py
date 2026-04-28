"""
Observability Layer
Structured logging, metrics collection, and request/response tracking.
"""

import logging
import json
import time
import sqlite3
import os
from typing import Dict, Optional
from functools import wraps

LOG_PATH = "/home/claude/learning_system/data/system.log"
METRICS_DB = "/home/claude/learning_system/data/metrics.db"

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


# ── LOGGING SETUP ────────────────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        return json.dumps(log_data)


def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(console)

    # File handler (JSON structured logs)
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(JSONFormatter())
    root.addHandler(file_handler)


# ── METRICS DATABASE ─────────────────────────────────────────────────────

def init_metrics_db():
    conn = sqlite3.connect(METRICS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS request_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            user_id TEXT,
            endpoint TEXT,
            method TEXT,
            query TEXT,
            agent TEXT,
            response_time_ms REAL,
            status_code INTEGER,
            error TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rag_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            user_id TEXT,
            query TEXT,
            num_results INTEGER,
            top_score REAL,
            retrieved_doc_ids TEXT,
            response_time_ms REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            user_id TEXT,
            agent_name TEXT,
            action TEXT,
            input_summary TEXT,
            output_summary TEXT,
            duration_ms REAL,
            success INTEGER
        )
    """)
    conn.commit()
    conn.close()


class Metrics:
    def __init__(self):
        init_metrics_db()
        self.logger = logging.getLogger("observability.metrics")

    def log_request(self, user_id: str, endpoint: str, method: str, query: str,
                    agent: str, response_time_ms: float, status_code: int, error: str = None):
        try:
            conn = sqlite3.connect(METRICS_DB)
            conn.execute(
                "INSERT INTO request_logs (timestamp, user_id, endpoint, method, query, agent, response_time_ms, status_code, error) VALUES (?,?,?,?,?,?,?,?,?)",
                (time.time(), user_id, endpoint, method, query[:200], agent, response_time_ms, status_code, error),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to log request metric: {e}")

    def log_rag(self, user_id: str, query: str, results: list, response_time_ms: float):
        try:
            top_score = max((r.get("score", 0) for r in results), default=0)
            doc_ids = json.dumps([r.get("doc_id") for r in results[:5]])
            conn = sqlite3.connect(METRICS_DB)
            conn.execute(
                "INSERT INTO rag_logs (timestamp, user_id, query, num_results, top_score, retrieved_doc_ids, response_time_ms) VALUES (?,?,?,?,?,?,?)",
                (time.time(), user_id, query[:200], len(results), top_score, doc_ids, response_time_ms),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to log RAG metric: {e}")

    def log_agent(self, user_id: str, agent_name: str, action: str,
                  input_summary: str, output_summary: str, duration_ms: float, success: bool):
        try:
            conn = sqlite3.connect(METRICS_DB)
            conn.execute(
                "INSERT INTO agent_logs (timestamp, user_id, agent_name, action, input_summary, output_summary, duration_ms, success) VALUES (?,?,?,?,?,?,?,?)",
                (time.time(), user_id, agent_name, action, input_summary[:200], output_summary[:200], duration_ms, int(success)),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to log agent metric: {e}")

    def get_summary(self) -> Dict:
        try:
            conn = sqlite3.connect(METRICS_DB)
            conn.row_factory = sqlite3.Row

            req_count = conn.execute("SELECT COUNT(*) as c FROM request_logs").fetchone()["c"]
            avg_rt = conn.execute("SELECT AVG(response_time_ms) as a FROM request_logs").fetchone()["a"]
            rag_count = conn.execute("SELECT COUNT(*) as c FROM rag_logs").fetchone()["c"]
            avg_score = conn.execute("SELECT AVG(top_score) as a FROM rag_logs").fetchone()["a"]
            agent_count = conn.execute("SELECT COUNT(*) as c FROM agent_logs").fetchone()["c"]
            success_rate = conn.execute("SELECT AVG(success)*100 as a FROM agent_logs").fetchone()["a"]

            recent_requests = conn.execute(
                "SELECT endpoint, response_time_ms, status_code, timestamp FROM request_logs ORDER BY timestamp DESC LIMIT 10"
            ).fetchall()

            conn.close()
            return {
                "requests": {
                    "total": req_count,
                    "avg_response_time_ms": round(avg_rt or 0, 2),
                },
                "rag": {
                    "total_retrievals": rag_count,
                    "avg_top_score": round(avg_score or 0, 4),
                },
                "agents": {
                    "total_calls": agent_count,
                    "success_rate_pct": round(success_rate or 0, 1),
                },
                "recent_requests": [dict(r) for r in recent_requests],
            }
        except Exception as e:
            return {"error": str(e)}


# ── TIMING DECORATOR ─────────────────────────────────────────────────────

def timed(metrics: Metrics, agent_name: str, action: str, user_id_key: str = "user_id"):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            success = True
            result = None
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = (time.time() - t0) * 1000
                uid = kwargs.get(user_id_key, "unknown")
                out_summary = str(result)[:100] if result else ""
                metrics.log_agent(uid, agent_name, action, str(args)[:100], out_summary, duration, success)
        return wrapper
    return decorator


# Singleton
setup_logging()
metrics = Metrics()
