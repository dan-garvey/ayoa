from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from app.schemas.image_generation import (
    GeneratedImageArtifact,
    ImageDeliveryKind,
    ImageGenerationJob,
    ImageGenerationRequest,
    ImageGenerationStatus,
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS image_jobs (
    job_id              TEXT PRIMARY KEY,
    dedupe_key          TEXT NOT NULL UNIQUE,
    session_id          TEXT NOT NULL,
    checkpoint_id       TEXT NOT NULL,
    checkpoint_sha256   TEXT NOT NULL,
    turn_index          INTEGER NOT NULL,
    actor_character_id  TEXT NOT NULL,
    delivery_kind       TEXT NOT NULL,
    status              TEXT NOT NULL,
    request_json        TEXT NOT NULL,
    artifact_json       TEXT NOT NULL DEFAULT '',
    error_code          TEXT NOT NULL DEFAULT '',
    attempts            INTEGER NOT NULL DEFAULT 0,
    delivery_attempts   INTEGER NOT NULL DEFAULT 0,
    next_delivery_at    REAL NOT NULL DEFAULT 0,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL,
    started_at          REAL,
    completed_at        REAL,
    delivered_at        REAL
);

CREATE INDEX IF NOT EXISTS image_jobs_queue_idx
ON image_jobs(status, created_at, job_id);

CREATE INDEX IF NOT EXISTS image_jobs_rewind_idx
ON image_jobs(session_id, turn_index, status);

CREATE TABLE IF NOT EXISTS image_eligible_beats (
    session_id          TEXT NOT NULL,
    actor_character_id  TEXT NOT NULL,
    checkpoint_sha256   TEXT NOT NULL,
    turn_index          INTEGER NOT NULL,
    ordinal             INTEGER NOT NULL,
    created_at          REAL NOT NULL,
    PRIMARY KEY (session_id, actor_character_id, checkpoint_sha256),
    UNIQUE (session_id, actor_character_id, ordinal)
);
"""


class ImageJobStore:
    """Small durable queue for presentation-only generation jobs."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        _chmod_private(self.db_path.parent, directory=True)
        with self._connect() as db:
            db.executescript(_SCHEMA)
            self._ensure_columns(db)
        _chmod_private(self.db_path, directory=False)

    def enqueue(self, request: ImageGenerationRequest) -> ImageGenerationJob:
        now = time.time()
        job_id = f"img_{request.dedupe_key[:32]}"
        request_json = request.model_dump_json()
        with self._connect() as db:
            db.execute(
                """
                INSERT OR IGNORE INTO image_jobs (
                    job_id, dedupe_key, session_id, checkpoint_id,
                    checkpoint_sha256, turn_index, actor_character_id,
                    delivery_kind, status, request_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    request.dedupe_key,
                    request.session_id,
                    request.checkpoint_id,
                    request.checkpoint_sha256,
                    request.turn_index,
                    request.actor_character_id,
                    request.delivery_kind.value,
                    ImageGenerationStatus.queued.value,
                    request_json,
                    now,
                    now,
                ),
            )
            row = db.execute(
                "SELECT * FROM image_jobs WHERE dedupe_key = ?",
                (request.dedupe_key,),
            ).fetchone()
        if row is None:
            raise RuntimeError("image job enqueue did not produce a row")
        return _job_from_row(row)

    def get(self, job_id: str) -> ImageGenerationJob | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM image_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return _job_from_row(row) if row is not None else None

    def active_count(self) -> int:
        with self._connect() as db:
            row = db.execute(
                "SELECT COUNT(*) FROM image_jobs WHERE status IN (?, ?)",
                (
                    ImageGenerationStatus.queued.value,
                    ImageGenerationStatus.running.value,
                ),
            ).fetchone()
        return int(row[0]) if row else 0

    def recover_interrupted(self) -> int:
        now = time.time()
        with self._connect() as db:
            running = db.execute(
                """
                UPDATE image_jobs
                SET status = ?, updated_at = ?, started_at = NULL,
                    error_code = ''
                WHERE status = ?
                """,
                (
                    ImageGenerationStatus.queued.value,
                    now,
                    ImageGenerationStatus.running.value,
                ),
            )
        return running.rowcount

    def recover_expired_deliveries(self, *, lease_seconds: float = 300) -> int:
        now = time.time()
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE image_jobs
                SET status = ?, updated_at = ?, next_delivery_at = ?
                WHERE status = ? AND updated_at < ?
                """,
                (
                    ImageGenerationStatus.succeeded.value,
                    now,
                    now,
                    ImageGenerationStatus.delivering.value,
                    now - max(1.0, lease_seconds),
                ),
            )
        return cursor.rowcount

    def claim_next(self) -> ImageGenerationJob | None:
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """
                SELECT * FROM image_jobs
                WHERE status = ?
                ORDER BY created_at, job_id
                LIMIT 1
                """,
                (ImageGenerationStatus.queued.value,),
            ).fetchone()
            if row is None:
                db.commit()
                return None
            cursor = db.execute(
                """
                UPDATE image_jobs
                SET status = ?, attempts = attempts + 1,
                    started_at = ?, updated_at = ?, error_code = ''
                WHERE job_id = ? AND status = ?
                """,
                (
                    ImageGenerationStatus.running.value,
                    now,
                    now,
                    row["job_id"],
                    ImageGenerationStatus.queued.value,
                ),
            )
            if cursor.rowcount != 1:
                db.rollback()
                return None
            claimed = db.execute(
                "SELECT * FROM image_jobs WHERE job_id = ?",
                (row["job_id"],),
            ).fetchone()
            db.commit()
        return _job_from_row(claimed) if claimed is not None else None

    def mark_succeeded(
        self,
        job_id: str,
        artifact: GeneratedImageArtifact,
    ) -> ImageGenerationJob | None:
        now = time.time()
        with self._connect() as db:
            db.execute(
                """
                UPDATE image_jobs
                SET status = ?, artifact_json = ?, error_code = '',
                    completed_at = ?, updated_at = ?
                WHERE job_id = ? AND status = ?
                """,
                (
                    ImageGenerationStatus.succeeded.value,
                    artifact.model_dump_json(),
                    now,
                    now,
                    job_id,
                    ImageGenerationStatus.running.value,
                ),
            )
        return self.get(job_id)

    def mark_failed(self, job_id: str, error_code: str) -> ImageGenerationJob | None:
        now = time.time()
        with self._connect() as db:
            db.execute(
                """
                UPDATE image_jobs
                SET status = ?, error_code = ?, completed_at = ?, updated_at = ?
                WHERE job_id = ? AND status IN (?, ?)
                """,
                (
                    ImageGenerationStatus.failed.value,
                    str(error_code or "generation_failed").strip(),
                    now,
                    now,
                    job_id,
                    ImageGenerationStatus.queued.value,
                    ImageGenerationStatus.running.value,
                ),
            )
        return self.get(job_id)

    def mark_delivered(self, job_id: str) -> ImageGenerationJob | None:
        now = time.time()
        with self._connect() as db:
            db.execute(
                """
                UPDATE image_jobs
                SET status = ?, delivered_at = ?, updated_at = ?
                WHERE job_id = ? AND status = ?
                """,
                (
                    ImageGenerationStatus.delivered.value,
                    now,
                    now,
                    job_id,
                    ImageGenerationStatus.delivering.value,
                ),
            )
        return self.get(job_id)

    def claim_delivery(self, job_id: str) -> ImageGenerationJob | None:
        now = time.time()
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE image_jobs
                SET status = ?, updated_at = ?,
                    delivery_attempts = delivery_attempts + 1
                WHERE job_id = ? AND status = ? AND next_delivery_at <= ?
                """,
                (
                    ImageGenerationStatus.delivering.value,
                    now,
                    job_id,
                    ImageGenerationStatus.succeeded.value,
                    now,
                ),
            )
            if cursor.rowcount != 1:
                return None
        return self.get(job_id)

    def release_delivery(self, job_id: str) -> ImageGenerationJob | None:
        now = time.time()
        with self._connect() as db:
            row = db.execute(
                "SELECT delivery_attempts FROM image_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            attempts = int(row["delivery_attempts"] or 1) if row else 1
            retry_at = now + min(300.0, float(2 ** min(attempts, 8)))
            db.execute(
                """
                UPDATE image_jobs
                SET status = ?, updated_at = ?, next_delivery_at = ?
                WHERE job_id = ? AND status = ?
                """,
                (
                    ImageGenerationStatus.succeeded.value,
                    now,
                    retry_at,
                    job_id,
                    ImageGenerationStatus.delivering.value,
                ),
            )
        return self.get(job_id)

    def heartbeat_delivery(self, job_id: str) -> bool:
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE image_jobs SET updated_at = ?
                WHERE job_id = ? AND status = ?
                """,
                (
                    time.time(),
                    job_id,
                    ImageGenerationStatus.delivering.value,
                ),
            )
        return cursor.rowcount == 1

    def requeue_retryable(self, job_id: str) -> ImageGenerationJob | None:
        now = time.time()
        with self._connect() as db:
            row = db.execute(
                "SELECT artifact_json FROM image_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            status = (
                ImageGenerationStatus.succeeded.value
                if str(row["artifact_json"] or "")
                else ImageGenerationStatus.queued.value
            )
            db.execute(
                """
                UPDATE image_jobs
                SET status = ?, error_code = '', updated_at = ?,
                    started_at = NULL,
                    completed_at = CASE WHEN ? = ? THEN completed_at ELSE NULL END,
                    delivered_at = NULL, next_delivery_at = 0
                WHERE job_id = ? AND status IN (?, ?)
                """,
                (
                    status,
                    now,
                    status,
                    ImageGenerationStatus.succeeded.value,
                    job_id,
                    ImageGenerationStatus.failed.value,
                    ImageGenerationStatus.cancelled.value,
                ),
            )
        return self.get(job_id)

    def mark_cancelled(
        self,
        job_id: str,
        error_code: str = "cancelled",
    ) -> ImageGenerationJob | None:
        now = time.time()
        with self._connect() as db:
            db.execute(
                """
                UPDATE image_jobs
                SET status = ?, error_code = ?, completed_at = ?, updated_at = ?
                WHERE job_id = ? AND status IN (?, ?, ?, ?)
                """,
                (
                    ImageGenerationStatus.cancelled.value,
                    str(error_code or "cancelled").strip(),
                    now,
                    now,
                    job_id,
                    ImageGenerationStatus.queued.value,
                    ImageGenerationStatus.running.value,
                    ImageGenerationStatus.succeeded.value,
                    ImageGenerationStatus.delivering.value,
                ),
            )
        return self.get(job_id)

    def cancel_after(self, session_id: str, turn_index: int) -> int:
        now = time.time()
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE image_jobs
                SET status = ?, error_code = 'rewound', completed_at = ?,
                    updated_at = ?
                WHERE session_id = ? AND turn_index > ?
                  AND status IN (?, ?, ?, ?, ?)
                """,
                (
                    ImageGenerationStatus.cancelled.value,
                    now,
                    now,
                    session_id,
                    turn_index,
                    ImageGenerationStatus.queued.value,
                    ImageGenerationStatus.running.value,
                    ImageGenerationStatus.succeeded.value,
                    ImageGenerationStatus.delivering.value,
                    ImageGenerationStatus.delivered.value,
                ),
            )
            db.execute(
                """
                DELETE FROM image_eligible_beats
                WHERE session_id = ? AND turn_index > ?
                """,
                (session_id, turn_index),
            )
        return cursor.rowcount

    def cancel_session(self, session_id: str) -> int:
        now = time.time()
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE image_jobs
                SET status = ?, error_code = 'session_ended', completed_at = ?,
                    updated_at = ?
                WHERE session_id = ?
                  AND status IN (?, ?, ?, ?)
                """,
                (
                    ImageGenerationStatus.cancelled.value,
                    now,
                    now,
                    session_id,
                    ImageGenerationStatus.queued.value,
                    ImageGenerationStatus.running.value,
                    ImageGenerationStatus.succeeded.value,
                    ImageGenerationStatus.delivering.value,
                ),
            )
        return cursor.rowcount

    def cancel_discord_destination(
        self,
        *,
        session_id: str,
        session_channel_id: int,
    ) -> int:
        now = time.time()
        cancelled = 0
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT job_id, request_json FROM image_jobs
                WHERE session_id = ? AND delivery_kind = ?
                  AND status IN (?, ?, ?, ?)
                """,
                (
                    session_id,
                    ImageDeliveryKind.discord.value,
                    ImageGenerationStatus.queued.value,
                    ImageGenerationStatus.running.value,
                    ImageGenerationStatus.succeeded.value,
                    ImageGenerationStatus.delivering.value,
                ),
            ).fetchall()
            for row in rows:
                request = ImageGenerationRequest.model_validate_json(
                    row["request_json"]
                )
                try:
                    target_channel = int(
                        request.delivery.get("session_channel_id")
                    )
                except (TypeError, ValueError):
                    continue
                if target_channel != int(session_channel_id):
                    continue
                cursor = db.execute(
                    """
                    UPDATE image_jobs
                    SET status = ?, error_code = 'session_detached',
                        completed_at = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (
                        ImageGenerationStatus.cancelled.value,
                        now,
                        now,
                        row["job_id"],
                    ),
                )
                cancelled += cursor.rowcount
        return cancelled

    def register_eligible_beat(
        self,
        *,
        session_id: str,
        actor_character_id: str,
        checkpoint_sha256: str,
        turn_index: int,
    ) -> int:
        """Return a stable one-based ordinal for an eligible POV beat."""

        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                """
                SELECT ordinal FROM image_eligible_beats
                WHERE session_id = ? AND actor_character_id = ?
                  AND checkpoint_sha256 = ?
                """,
                (session_id, actor_character_id, checkpoint_sha256),
            ).fetchone()
            if existing is not None:
                db.commit()
                return int(existing["ordinal"])
            row = db.execute(
                """
                SELECT COALESCE(MAX(ordinal), 0) + 1
                FROM image_eligible_beats
                WHERE session_id = ? AND actor_character_id = ?
                """,
                (session_id, actor_character_id),
            ).fetchone()
            ordinal = int(row[0])
            db.execute(
                """
                INSERT INTO image_eligible_beats (
                    session_id, actor_character_id, checkpoint_sha256,
                    turn_index, ordinal, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    actor_character_id,
                    checkpoint_sha256,
                    turn_index,
                    ordinal,
                    now,
                ),
            )
            db.commit()
        return ordinal

    def succeeded_undelivered(
        self,
        delivery_kind: ImageDeliveryKind,
        *,
        session_id: str | None = None,
        actor_character_id: str | None = None,
    ) -> list[ImageGenerationJob]:
        where = ["status = ?", "delivery_kind = ?", "next_delivery_at <= ?"]
        params: list[object] = [
            ImageGenerationStatus.succeeded.value,
            delivery_kind.value,
            time.time(),
        ]
        if session_id is not None:
            where.append("session_id = ?")
            params.append(session_id)
        if actor_character_id is not None:
            where.append("actor_character_id = ?")
            params.append(actor_character_id)
        with self._connect() as db:
            rows = db.execute(
                f"""
                SELECT * FROM image_jobs
                WHERE {' AND '.join(where)}
                ORDER BY completed_at, created_at, job_id
                """,
                params,
            ).fetchall()
        return [_job_from_row(row) for row in rows]

    def pending_delivery(
        self,
        delivery_kind: ImageDeliveryKind,
        *,
        session_id: str,
    ) -> list[ImageGenerationJob]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT * FROM image_jobs
                WHERE delivery_kind = ? AND session_id = ?
                  AND status IN (?, ?, ?)
                ORDER BY created_at, job_id
                """,
                (
                    delivery_kind.value,
                    session_id,
                    ImageGenerationStatus.queued.value,
                    ImageGenerationStatus.running.value,
                    ImageGenerationStatus.succeeded.value,
                ),
            ).fetchall()
        return [_job_from_row(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _ensure_columns(db: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in db.execute("PRAGMA table_info(image_jobs)").fetchall()
        }
        if "delivery_attempts" not in columns:
            db.execute(
                "ALTER TABLE image_jobs ADD COLUMN "
                "delivery_attempts INTEGER NOT NULL DEFAULT 0"
            )
        if "next_delivery_at" not in columns:
            db.execute(
                "ALTER TABLE image_jobs ADD COLUMN "
                "next_delivery_at REAL NOT NULL DEFAULT 0"
            )
        beat_columns = {
            str(row["name"])
            for row in db.execute(
                "PRAGMA table_info(image_eligible_beats)"
            ).fetchall()
        }
        if "turn_index" not in beat_columns:
            db.execute(
                "ALTER TABLE image_eligible_beats ADD COLUMN "
                "turn_index INTEGER NOT NULL DEFAULT 0"
            )


def _job_from_row(row: sqlite3.Row) -> ImageGenerationJob:
    request = ImageGenerationRequest.model_validate_json(row["request_json"])
    artifact_raw = str(row["artifact_json"] or "")
    artifact = (
        GeneratedImageArtifact.model_validate_json(artifact_raw)
        if artifact_raw
        else None
    )
    return ImageGenerationJob(
        job_id=row["job_id"],
        request=request,
        status=ImageGenerationStatus(row["status"]),
        artifact=artifact,
        error_code=row["error_code"] or "",
        attempts=int(row["attempts"] or 0),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        started_at=(
            float(row["started_at"]) if row["started_at"] is not None else None
        ),
        completed_at=(
            float(row["completed_at"])
            if row["completed_at"] is not None
            else None
        ),
        delivered_at=(
            float(row["delivered_at"])
            if row["delivered_at"] is not None
            else None
        ),
    )


def _chmod_private(path: Path, *, directory: bool) -> None:
    try:
        path.chmod(0o700 if directory else 0o600)
    except OSError:
        return
