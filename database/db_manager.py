"""SQLite persistence for AURA chat history and analysis history."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from config.settings import DB_PATH
from src.utils import now_iso, to_jsonable


class DatabaseError(RuntimeError):
    """Raised when a database operation fails."""


class DatabaseManager:
    """Thin wrapper around sqlite3 with explicit connection handling."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_schema()
        except (OSError, sqlite3.Error) as exc:
            raise DatabaseError(f"Could not initialise the AURA database: {exc}") from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except sqlite3.Error as exc:
            connection.rollback()
            raise DatabaseError(str(exc)) from exc
        finally:
            connection.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL DEFAULT 'default',
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT '',
                    tool_trace TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chat_session
                    ON chat_messages (session_id, id);

                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    action TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    details TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_analysis_created
                    ON analysis_history (id DESC);
                """
            )

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------
    def add_message(
        self,
        role: str,
        content: str,
        mode: str = "",
        tool_trace: list[dict[str, Any]] | None = None,
        session_id: str = "default",
    ) -> int:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        trace_json = json.dumps(to_jsonable(tool_trace)) if tool_trace else None
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, mode, tool_trace, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, role, content, mode, trace_json, now_iso()),
            )
            return int(cursor.lastrowid or 0)

    def get_messages(self, session_id: str = "default", limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM (SELECT * FROM chat_messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                (session_id, int(limit)),
            ).fetchall()
        messages: list[dict[str, Any]] = []
        for row in rows:
            trace = None
            if row["tool_trace"]:
                try:
                    trace = json.loads(row["tool_trace"])
                except json.JSONDecodeError:
                    trace = None
            messages.append(
                {
                    "id": row["id"],
                    "role": row["role"],
                    "content": row["content"],
                    "mode": row["mode"],
                    "tool_trace": trace,
                    "created_at": row["created_at"],
                }
            )
        return messages

    def clear_messages(self, session_id: str = "default") -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))

    # ------------------------------------------------------------------
    # Analysis history
    # ------------------------------------------------------------------
    def log_analysis(
        self,
        dataset_name: str,
        action: str,
        summary: str,
        details: dict[str, Any] | None = None,
    ) -> int:
        details_json = json.dumps(to_jsonable(details)) if details else None
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO analysis_history (dataset_name, action, summary, details, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (dataset_name or "unnamed", action, summary, details_json, now_iso()),
            )
            return int(cursor.lastrowid or 0)

    def get_analysis_history(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM analysis_history ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        history: list[dict[str, Any]] = []
        for row in rows:
            details = None
            if row["details"]:
                try:
                    details = json.loads(row["details"])
                except json.JSONDecodeError:
                    details = None
            history.append(
                {
                    "id": row["id"],
                    "dataset_name": row["dataset_name"],
                    "action": row["action"],
                    "summary": row["summary"],
                    "details": details,
                    "created_at": row["created_at"],
                }
            )
        return history

    def clear_analysis_history(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM analysis_history")