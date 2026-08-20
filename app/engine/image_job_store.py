from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Sequence

from app.engine.image_director import DurableDirectorRun, VisibleEventProjection
from app.schemas.image_director import ImageDirectorOutput
from app.schemas.image_generation import (
    IMAGE_JOB_SCHEMA_VERSION,
    FrozenReferenceInput,
    GeneratedImageArtifact,
    IdentityReferenceCandidate,
    IdentityReferenceStatus,
    ImageDelivery,
    ImageDeliveryKind,
    ImageDeliveryStatus,
    ImageGenerationJob,
    ImageGenerationRequest,
    ImageGenerationStatus,
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS image_store_meta (
    key                 TEXT PRIMARY KEY,
    value               TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS image_transactions (
    transaction_id      TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL,
    source_turn_index   INTEGER NOT NULL,
    source_checkpoint_sha256 TEXT NOT NULL,
    target_checkpoint_sha256 TEXT NOT NULL DEFAULT '',
    lineage_bound       INTEGER NOT NULL DEFAULT 1,
    status              TEXT NOT NULL,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS image_transactions_session_idx
ON image_transactions(session_id, status);

CREATE TABLE IF NOT EXISTS image_director_runs (
    run_id              TEXT PRIMARY KEY,
    projection_key      TEXT NOT NULL,
    session_id          TEXT NOT NULL,
    transaction_id      TEXT NOT NULL,
    source_event_id     TEXT NOT NULL,
    source_event_fingerprint TEXT NOT NULL,
    source_event_sequence INTEGER NOT NULL,
    source_turn_index   INTEGER NOT NULL,
    projection_json     TEXT NOT NULL,
    status              TEXT NOT NULL,
    output_json         TEXT NOT NULL DEFAULT '',
    error_code          TEXT NOT NULL DEFAULT '',
    attempts            INTEGER NOT NULL DEFAULT 0,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL,
    UNIQUE(transaction_id, source_event_id, projection_key),
    FOREIGN KEY(transaction_id) REFERENCES image_transactions(transaction_id)
);

CREATE INDEX IF NOT EXISTS image_director_queue_idx
ON image_director_runs(status, source_event_sequence, created_at);

CREATE TABLE IF NOT EXISTS image_jobs (
    job_id              TEXT PRIMARY KEY,
    dedupe_key          TEXT NOT NULL UNIQUE,
    session_id          TEXT NOT NULL,
    transaction_id      TEXT NOT NULL,
    source_event_id     TEXT NOT NULL,
    source_event_fingerprint TEXT NOT NULL,
    source_event_sequence INTEGER NOT NULL,
    source_turn_index   INTEGER NOT NULL,
    status              TEXT NOT NULL,
    request_json        TEXT NOT NULL,
    artifact_json       TEXT NOT NULL DEFAULT '',
    error_code          TEXT NOT NULL DEFAULT '',
    attempts            INTEGER NOT NULL DEFAULT 0,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL,
    started_at          REAL,
    completed_at        REAL,
    FOREIGN KEY(transaction_id) REFERENCES image_transactions(transaction_id)
);

CREATE INDEX IF NOT EXISTS image_jobs_queue_idx
ON image_jobs(status, created_at, job_id);

CREATE INDEX IF NOT EXISTS image_jobs_lineage_idx
ON image_jobs(session_id, source_turn_index, source_event_id);

CREATE TABLE IF NOT EXISTS image_prose_gates (
    transaction_id      TEXT NOT NULL,
    source_event_id     TEXT NOT NULL,
    pov_character_id    TEXT NOT NULL,
    opened_at           REAL NOT NULL,
    PRIMARY KEY (transaction_id, source_event_id, pov_character_id),
    FOREIGN KEY(transaction_id) REFERENCES image_transactions(transaction_id)
);

CREATE TABLE IF NOT EXISTS image_prose_receipts (
    session_id           TEXT NOT NULL,
    source_event_id      TEXT NOT NULL,
    pov_character_id     TEXT NOT NULL,
    opened_at            REAL NOT NULL,
    PRIMARY KEY (session_id, source_event_id, pov_character_id)
);

CREATE TABLE IF NOT EXISTS image_deliveries (
    delivery_id         TEXT PRIMARY KEY,
    job_id              TEXT NOT NULL,
    session_id          TEXT NOT NULL,
    transaction_id      TEXT NOT NULL,
    source_event_id     TEXT NOT NULL,
    source_turn_index   INTEGER NOT NULL,
    pov_character_id    TEXT NOT NULL,
    delivery_kind       TEXT NOT NULL,
    delivery_json       TEXT NOT NULL,
    status              TEXT NOT NULL,
    prose_rendered      INTEGER NOT NULL DEFAULT 0,
    attempts            INTEGER NOT NULL DEFAULT 0,
    next_attempt_at     REAL NOT NULL DEFAULT 0,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL,
    delivered_at        REAL,
    FOREIGN KEY(job_id) REFERENCES image_jobs(job_id),
    FOREIGN KEY(transaction_id) REFERENCES image_transactions(transaction_id),
    UNIQUE(job_id, pov_character_id, delivery_kind, delivery_json)
);

CREATE INDEX IF NOT EXISTS image_deliveries_queue_idx
ON image_deliveries(status, prose_rendered, next_attempt_at, created_at);

CREATE INDEX IF NOT EXISTS image_deliveries_destination_idx
ON image_deliveries(session_id, delivery_kind, status);

CREATE TABLE IF NOT EXISTS image_identity_candidates (
    candidate_id        TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL,
    character_id        TEXT NOT NULL,
    job_id              TEXT NOT NULL UNIQUE,
    artifact_json       TEXT NOT NULL,
    status              TEXT NOT NULL,
    active              INTEGER NOT NULL,
    reminder_required   INTEGER NOT NULL,
    reroll_of_reference_id TEXT NOT NULL DEFAULT '',
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL,
    FOREIGN KEY(job_id) REFERENCES image_jobs(job_id)
);

CREATE INDEX IF NOT EXISTS image_identity_active_idx
ON image_identity_candidates(session_id, character_id, active);

CREATE TABLE IF NOT EXISTS image_reviewed_references (
    session_id          TEXT NOT NULL,
    reference_id        TEXT NOT NULL,
    frozen_json         TEXT NOT NULL,
    purpose             TEXT NOT NULL,
    scope               TEXT NOT NULL,
    updated_at          REAL NOT NULL,
    PRIMARY KEY (session_id, reference_id)
);

CREATE TABLE IF NOT EXISTS image_reviewed_identity_bindings (
    session_id          TEXT NOT NULL,
    character_id        TEXT NOT NULL,
    reference_id        TEXT NOT NULL,
    priority            INTEGER NOT NULL,
    updated_at          REAL NOT NULL,
    PRIMARY KEY (session_id, character_id, reference_id),
    FOREIGN KEY(session_id, reference_id)
        REFERENCES image_reviewed_references(session_id, reference_id)
);

CREATE TABLE IF NOT EXISTS image_reviewed_location_bindings (
    session_id          TEXT NOT NULL,
    location_label      TEXT NOT NULL,
    reference_id        TEXT NOT NULL,
    priority            INTEGER NOT NULL,
    updated_at          REAL NOT NULL,
    PRIMARY KEY (session_id, location_label, reference_id),
    FOREIGN KEY(session_id, reference_id)
        REFERENCES image_reviewed_references(session_id, reference_id)
);

CREATE TABLE IF NOT EXISTS image_identity_policies (
    session_id          TEXT NOT NULL,
    character_id        TEXT NOT NULL,
    blocked             INTEGER NOT NULL,
    minimum_source_turn INTEGER NOT NULL,
    retirement_source_turn INTEGER NOT NULL DEFAULT 0,
    prior_candidate_id  TEXT NOT NULL DEFAULT '',
    prior_status        TEXT NOT NULL DEFAULT '',
    prior_reminder_required INTEGER NOT NULL DEFAULT 0,
    updated_at          REAL NOT NULL,
    PRIMARY KEY (session_id, character_id)
);
"""

_CURRENT_SCHEMA_COLUMNS = {
    "image_transactions": {
        "source_turn_index",
        "target_checkpoint_sha256",
        "lineage_bound",
        "status",
    },
    "image_director_runs": {
        "projection_key",
        "source_event_fingerprint",
        "projection_json",
        "output_json",
    },
    "image_jobs": {
        "dedupe_key",
        "transaction_id",
        "source_event_fingerprint",
        "request_json",
        "artifact_json",
    },
    "image_deliveries": {
        "transaction_id",
        "prose_rendered",
        "next_attempt_at",
        "delivery_json",
    },
    "image_identity_candidates": {
        "artifact_json",
        "active",
        "reminder_required",
        "reroll_of_reference_id",
    },
    "image_identity_policies": {
        "minimum_source_turn",
        "retirement_source_turn",
        "prior_candidate_id",
    },
    "image_reviewed_references": {
        "frozen_json",
        "purpose",
        "scope",
    },
    "image_reviewed_identity_bindings": {
        "character_id",
        "reference_id",
        "priority",
    },
    "image_reviewed_location_bindings": {
        "location_label",
        "reference_id",
        "priority",
    },
}


class ImageJobStore:
    """Durable event-provenance generation, delivery, and identity store."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        _chmod_private(self.db_path.parent, directory=True)
        with self._connect() as db:
            self._migrate_directly(db)
            db.executescript(_SCHEMA)
            db.execute(
                """
                INSERT OR REPLACE INTO image_store_meta(key, value)
                VALUES ('schema_version', ?)
                """,
                (IMAGE_JOB_SCHEMA_VERSION,),
            )
        _chmod_private(self.db_path, directory=False)

    def begin_transaction(
        self,
        *,
        transaction_id: str,
        session_id: str,
        source_turn_index: int,
        source_checkpoint_sha256: str,
        lineage_bound: bool = True,
    ) -> None:
        now = time.time()
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO image_transactions (
                    transaction_id, session_id, source_turn_index,
                    source_checkpoint_sha256, lineage_bound, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'speculative', ?, ?)
                ON CONFLICT(transaction_id) DO NOTHING
                """,
                (
                    transaction_id,
                    session_id,
                    max(0, int(source_turn_index)),
                    source_checkpoint_sha256,
                    int(lineage_bound),
                    now,
                    now,
                ),
            )

    def commit_transaction(
        self,
        transaction_id: str,
        *,
        target_checkpoint_sha256: str,
    ) -> bool:
        now = time.time()
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE image_transactions
                SET status = 'committed', target_checkpoint_sha256 = ?,
                    updated_at = ?
                WHERE transaction_id = ? AND status = 'speculative'
                """,
                (target_checkpoint_sha256, now, transaction_id),
            )
        return cursor.rowcount == 1

    def cancel_transaction(
        self,
        transaction_id: str,
        *,
        reason: str = "transaction_aborted",
    ) -> int:
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            transaction = db.execute(
                """
                SELECT session_id FROM image_transactions
                WHERE transaction_id = ?
                """,
                (transaction_id,),
            ).fetchone()
            db.execute(
                """
                UPDATE image_transactions
                SET status = 'cancelled', updated_at = ?
                WHERE transaction_id = ? AND status != 'cancelled'
                """,
                (now, transaction_id),
            )
            cursor = db.execute(
                """
                UPDATE image_jobs
                SET status = ?, error_code = ?, completed_at = ?,
                    updated_at = ?
                WHERE transaction_id = ?
                  AND status IN (?, ?, ?)
                """,
                (
                    ImageGenerationStatus.cancelled.value,
                    _clean_reason(reason),
                    now,
                    now,
                    transaction_id,
                    ImageGenerationStatus.queued.value,
                    ImageGenerationStatus.running.value,
                    ImageGenerationStatus.succeeded.value,
                ),
            )
            db.execute(
                """
                UPDATE image_deliveries
                SET status = ?, updated_at = ?
                WHERE transaction_id = ? AND status IN (?, ?)
                """,
                (
                    ImageDeliveryStatus.cancelled.value,
                    now,
                    transaction_id,
                    ImageDeliveryStatus.pending.value,
                    ImageDeliveryStatus.delivering.value,
                ),
            )
            db.execute(
                """
                UPDATE image_director_runs
                SET status = 'cancelled', error_code = ?, updated_at = ?
                WHERE transaction_id = ? AND status IN ('queued', 'running')
                """,
                (_clean_reason(reason), now, transaction_id),
            )
            if transaction is not None:
                db.execute(
                    """
                    DELETE FROM image_prose_receipts
                    WHERE session_id = ? AND source_event_id IN (
                        SELECT source_event_id FROM image_director_runs
                        WHERE transaction_id = ?
                        UNION
                        SELECT source_event_id FROM image_jobs
                        WHERE transaction_id = ?
                    )
                    """,
                    (
                        str(transaction["session_id"]),
                        transaction_id,
                        transaction_id,
                    ),
                )
                self._retire_cancelled_candidates(
                    db,
                    session_id=str(transaction["session_id"]),
                )
            db.commit()
        return cursor.rowcount

    def enqueue(self, request: ImageGenerationRequest) -> ImageGenerationJob:
        now = time.time()
        job_id = f"img_{request.dedupe_key[:32]}"
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            transaction = db.execute(
                """
                SELECT session_id, status FROM image_transactions
                WHERE transaction_id = ?
                """,
                (request.transaction_id,),
            ).fetchone()
            if transaction is None:
                raise RuntimeError(
                    "image request transaction was not registered"
                )
            if transaction["session_id"] != request.session_id:
                raise ValueError("image request session/transaction mismatch")
            if transaction["status"] == "cancelled":
                raise RuntimeError("cannot enqueue into a cancelled transaction")
            if request.reroll_of_reference_id:
                candidate = db.execute(
                    """
                    SELECT session_id, character_id, status
                    FROM image_identity_candidates
                    WHERE candidate_id = ?
                    """,
                    (request.reroll_of_reference_id,),
                ).fetchone()
                reviewed = db.execute(
                    """
                    SELECT b.session_id, b.character_id,
                           COALESCE(p.blocked, 0) AS blocked
                    FROM image_reviewed_identity_bindings AS b
                    LEFT JOIN image_identity_policies AS p
                      ON p.session_id = b.session_id
                     AND p.character_id = b.character_id
                    WHERE b.session_id = ? AND b.reference_id = ?
                    """,
                    (
                        request.session_id,
                        request.reroll_of_reference_id,
                    ),
                ).fetchone()
                candidate_current = bool(
                    candidate is not None
                    and candidate["session_id"] == request.session_id
                    and candidate["status"]
                    != IdentityReferenceStatus.retired.value
                )
                reviewed_current = bool(
                    reviewed is not None
                    and not bool(reviewed["blocked"])
                )
                if not candidate_current and not reviewed_current:
                    raise RuntimeError(
                        "identity reroll reference is no longer current"
                    )
                character_id = (
                    str(candidate["character_id"])
                    if candidate_current
                    else str(reviewed["character_id"])
                )
                if request.subject_character_ids != [character_id]:
                    raise RuntimeError("identity reroll subject mismatch")
            db.execute(
                """
                UPDATE image_transactions SET updated_at = ?
                WHERE transaction_id = ?
                """,
                (now, request.transaction_id),
            )
            db.execute(
                """
                INSERT OR IGNORE INTO image_jobs (
                    job_id, dedupe_key, session_id, transaction_id,
                    source_event_id, source_event_fingerprint,
                    source_event_sequence, source_turn_index, status,
                    request_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    request.dedupe_key,
                    request.session_id,
                    request.transaction_id,
                    request.source_event_id,
                    request.source_event_fingerprint,
                    request.source_event_sequence,
                    request.source_turn_index,
                    ImageGenerationStatus.queued.value,
                    request.model_dump_json(),
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

    def enqueue_director_run(
        self,
        projection: VisibleEventProjection,
    ) -> DurableDirectorRun:
        projection_key = projection.grouping_key()
        identity = "|".join(
            (
                projection.transaction_id,
                projection.event_id,
                projection_key,
            )
        )
        import hashlib

        run_id = "imgdir_" + hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:32]
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            transaction = db.execute(
                """
                SELECT status FROM image_transactions
                WHERE transaction_id = ?
                """,
                (projection.transaction_id,),
            ).fetchone()
            if transaction is None or transaction["status"] == "cancelled":
                raise RuntimeError(
                    "director run requires an active image transaction"
                )
            db.execute(
                """
                UPDATE image_transactions SET updated_at = ?
                WHERE transaction_id = ?
                """,
                (now, projection.transaction_id),
            )
            db.execute(
                """
                INSERT OR IGNORE INTO image_director_runs (
                    run_id, projection_key, session_id, transaction_id,
                    source_event_id, source_event_fingerprint,
                    source_event_sequence, source_turn_index,
                    projection_json, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    run_id,
                    projection_key,
                    projection.session_id,
                    projection.transaction_id,
                    projection.event_id,
                    projection.event_fingerprint,
                    projection.event_sequence,
                    projection.source_turn_index,
                    json.dumps(
                        projection.to_storage_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    now,
                    now,
                ),
            )
            row = db.execute(
                """
                SELECT * FROM image_director_runs WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("director run insert did not produce a row")
        return _director_run_from_row(row)

    def claim_next_director_run(self) -> DurableDirectorRun | None:
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """
                SELECT r.* FROM image_director_runs AS r
                JOIN image_transactions AS t
                  ON t.transaction_id = r.transaction_id
                WHERE r.status = 'queued' AND t.status != 'cancelled'
                ORDER BY r.source_event_sequence, r.created_at, r.run_id
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                db.commit()
                return None
            cursor = db.execute(
                """
                UPDATE image_director_runs
                SET status = 'running', attempts = attempts + 1,
                    error_code = '', updated_at = ?
                WHERE run_id = ? AND status = 'queued'
                """,
                (now, row["run_id"]),
            )
            if cursor.rowcount != 1:
                db.rollback()
                return None
            claimed = db.execute(
                """
                SELECT * FROM image_director_runs WHERE run_id = ?
                """,
                (row["run_id"],),
            ).fetchone()
            db.commit()
        return (
            _director_run_from_row(claimed)
            if claimed is not None
            else None
        )

    def complete_director_run(
        self,
        run_id: str,
        output: ImageDirectorOutput,
    ) -> DurableDirectorRun | None:
        with self._connect() as db:
            db.execute(
                """
                UPDATE image_director_runs
                SET status = 'succeeded', output_json = ?,
                    error_code = '', updated_at = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (output.model_dump_json(), time.time(), run_id),
            )
            row = db.execute(
                """
                SELECT * FROM image_director_runs WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        return _director_run_from_row(row) if row is not None else None

    def finalize_director_materialization(
        self,
        run_id: str,
        output: ImageDirectorOutput,
    ) -> DurableDirectorRun | None:
        """Record only requests that were admitted to the generation queue."""
        with self._connect() as db:
            db.execute(
                """
                UPDATE image_director_runs
                SET output_json = ?, updated_at = ?
                WHERE run_id = ? AND status = 'succeeded'
                """,
                (output.model_dump_json(), time.time(), run_id),
            )
            row = db.execute(
                """
                SELECT * FROM image_director_runs WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        return _director_run_from_row(row) if row is not None else None

    def heartbeat_director_run(self, run_id: str) -> bool:
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE image_director_runs SET updated_at = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (time.time(), run_id),
            )
        return cursor.rowcount == 1

    def fail_director_run(
        self,
        run_id: str,
        error_code: str,
    ) -> DurableDirectorRun | None:
        with self._connect() as db:
            db.execute(
                """
                UPDATE image_director_runs
                SET status = 'failed', error_code = ?, updated_at = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (_clean_reason(error_code), time.time(), run_id),
            )
            row = db.execute(
                """
                SELECT * FROM image_director_runs WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        return _director_run_from_row(row) if row is not None else None

    def recover_expired_director_runs(
        self,
        *,
        lease_seconds: float = 300,
    ) -> int:
        now = time.time()
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE image_director_runs
                SET status = 'queued', error_code = '', updated_at = ?
                WHERE status = 'running' AND updated_at < ?
                """,
                (now, now - max(1.0, lease_seconds)),
            )
        return cursor.rowcount

    def add_delivery(
        self,
        *,
        job_id: str,
        session_id: str,
        source_turn_index: int,
        pov_character_id: str,
        delivery_kind: ImageDeliveryKind,
        delivery: dict[str, object],
    ) -> ImageDelivery:
        job = self.get(job_id)
        if job is None:
            raise KeyError(f"unknown image job: {job_id}")
        if job.request.session_id != session_id:
            raise ValueError("image delivery session/job mismatch")
        delivery_json = json.dumps(
            delivery,
            sort_keys=True,
            separators=(",", ":"),
        )
        identity = "|".join(
            (
                job_id,
                pov_character_id,
                delivery_kind.value,
                delivery_json,
            )
        )
        import hashlib

        delivery_id = "imgdel_" + hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:32]
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state = db.execute(
                """
                SELECT j.status AS job_status, t.status AS transaction_status
                FROM image_jobs AS j
                JOIN image_transactions AS t
                  ON t.transaction_id = j.transaction_id
                WHERE j.job_id = ?
                """,
                (job_id,),
            ).fetchone()
            if (
                state is None
                or state["job_status"]
                == ImageGenerationStatus.cancelled.value
                or state["transaction_status"] == "cancelled"
            ):
                raise RuntimeError(
                    "image delivery requires current event lineage"
                )
            gate = db.execute(
                """
                SELECT 1 FROM image_prose_gates
                WHERE transaction_id = ? AND source_event_id = ?
                  AND pov_character_id = ?
                UNION
                SELECT 1 FROM image_prose_receipts
                WHERE session_id = ? AND source_event_id = ?
                  AND pov_character_id = ?
                """,
                (
                    job.request.transaction_id,
                    job.request.source_event_id,
                    pov_character_id,
                    session_id,
                    job.request.source_event_id,
                    pov_character_id,
                ),
            ).fetchone()
            db.execute(
                """
                INSERT OR IGNORE INTO image_deliveries (
                    delivery_id, job_id, session_id, transaction_id,
                    source_event_id, source_turn_index, pov_character_id,
                    delivery_kind, delivery_json, status, prose_rendered,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    delivery_id,
                    job_id,
                    session_id,
                    job.request.transaction_id,
                    job.request.source_event_id,
                    source_turn_index,
                    pov_character_id,
                    delivery_kind.value,
                    delivery_json,
                    ImageDeliveryStatus.pending.value,
                    1 if gate is not None else 0,
                    now,
                    now,
                ),
            )
            row = db.execute(
                "SELECT * FROM image_deliveries WHERE delivery_id = ?",
                (delivery_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("image delivery insert did not produce a row")
        return _delivery_from_row(row)

    def get(self, job_id: str) -> ImageGenerationJob | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM image_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return _job_from_row(row) if row is not None else None

    def get_delivery(self, delivery_id: str) -> ImageDelivery | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM image_deliveries WHERE delivery_id = ?",
                (delivery_id,),
            ).fetchone()
        return _delivery_from_row(row) if row is not None else None

    def deliveries_for_job(self, job_id: str) -> list[ImageDelivery]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT * FROM image_deliveries
                WHERE job_id = ? ORDER BY created_at, delivery_id
                """,
                (job_id,),
            ).fetchall()
        return [_delivery_from_row(row) for row in rows]

    def active_count(self) -> int:
        with self._connect() as db:
            row = db.execute(
                """
                SELECT COUNT(*) FROM image_jobs
                WHERE status IN (?, ?)
                """,
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

    def speculative_transactions(self) -> list[dict[str, object]]:
        with self._connect() as db:
            transactions = db.execute(
                """
                SELECT transaction_id, session_id, updated_at
                FROM image_transactions
                WHERE status = 'speculative'
                ORDER BY created_at, transaction_id
                """
            ).fetchall()
            result: list[dict[str, object]] = []
            for transaction in transactions:
                event_rows = db.execute(
                    """
                    SELECT source_event_id, source_event_fingerprint,
                           source_turn_index
                    FROM image_director_runs
                    WHERE transaction_id = ?
                    UNION
                    SELECT source_event_id, source_event_fingerprint,
                           source_turn_index
                    FROM image_jobs
                    WHERE transaction_id = ?
                    """,
                    (
                        transaction["transaction_id"],
                        transaction["transaction_id"],
                    ),
                ).fetchall()
                result.append(
                    {
                        "transaction_id": str(transaction["transaction_id"]),
                        "session_id": str(transaction["session_id"]),
                        "updated_at": float(transaction["updated_at"]),
                        "events": [
                            (
                                str(row["source_event_id"]),
                                str(row["source_event_fingerprint"]),
                                int(row["source_turn_index"]),
                            )
                            for row in event_rows
                        ],
                    }
                )
        return result

    def recover_expired_deliveries(
        self,
        *,
        lease_seconds: float = 300,
    ) -> int:
        now = time.time()
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE image_deliveries
                SET status = ?, updated_at = ?, next_attempt_at = ?
                WHERE status = ? AND updated_at < ?
                """,
                (
                    ImageDeliveryStatus.pending.value,
                    now,
                    now,
                    ImageDeliveryStatus.delivering.value,
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
                SELECT j.* FROM image_jobs AS j
                JOIN image_transactions AS t
                  ON t.transaction_id = j.transaction_id
                WHERE j.status = ? AND t.status != 'cancelled'
                ORDER BY j.source_event_sequence,
                         CAST(json_extract(
                             j.request_json,
                             '$.request_ordinal'
                         ) AS INTEGER),
                         j.created_at, j.job_id
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

    def mark_failed(
        self,
        job_id: str,
        error_code: str,
    ) -> ImageGenerationJob | None:
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
                    _clean_reason(error_code or "generation_failed"),
                    now,
                    now,
                    job_id,
                    ImageGenerationStatus.queued.value,
                    ImageGenerationStatus.running.value,
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
            db.execute("BEGIN IMMEDIATE")
            session = db.execute(
                "SELECT session_id FROM image_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            db.execute(
                """
                UPDATE image_jobs
                SET status = ?, error_code = ?, completed_at = ?, updated_at = ?
                WHERE job_id = ? AND status IN (?, ?, ?)
                """,
                (
                    ImageGenerationStatus.cancelled.value,
                    _clean_reason(error_code),
                    now,
                    now,
                    job_id,
                    ImageGenerationStatus.queued.value,
                    ImageGenerationStatus.running.value,
                    ImageGenerationStatus.succeeded.value,
                ),
            )
            db.execute(
                """
                UPDATE image_deliveries
                SET status = ?, updated_at = ?
                WHERE job_id = ? AND status IN (?, ?)
                """,
                (
                    ImageDeliveryStatus.cancelled.value,
                    now,
                    job_id,
                    ImageDeliveryStatus.pending.value,
                    ImageDeliveryStatus.delivering.value,
                ),
            )
            if session is not None:
                self._retire_cancelled_candidates(
                    db,
                    session_id=str(session["session_id"]),
                )
            db.commit()
        return self.get(job_id)

    def requeue_retryable(
        self,
        job_id: str,
    ) -> ImageGenerationJob | None:
        now = time.time()
        with self._connect() as db:
            db.execute(
                """
                UPDATE image_jobs
                SET status = ?, error_code = '', updated_at = ?,
                    started_at = NULL, completed_at = NULL
                WHERE job_id = ? AND status = ?
                """,
                (
                    ImageGenerationStatus.queued.value,
                    now,
                    job_id,
                    ImageGenerationStatus.failed.value,
                ),
            )
        return self.get(job_id)

    def open_prose_gate(
        self,
        *,
        transaction_id: str,
        source_event_id: str,
        pov_character_ids: Iterable[str],
    ) -> int:
        ids = tuple(dict.fromkeys(item for item in pov_character_ids if item))
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as db:
            db.executemany(
                """
                INSERT OR IGNORE INTO image_prose_gates (
                    transaction_id, source_event_id, pov_character_id,
                    opened_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    (
                        transaction_id,
                        source_event_id,
                        pov_character_id,
                        time.time(),
                    )
                    for pov_character_id in ids
                ),
            )
            cursor = db.execute(
                f"""
                UPDATE image_deliveries
                SET prose_rendered = 1, updated_at = ?
                WHERE transaction_id = ? AND source_event_id = ?
                  AND pov_character_id IN ({placeholders})
                  AND status = ?
                """,
                (
                    time.time(),
                    transaction_id,
                    source_event_id,
                    *ids,
                    ImageDeliveryStatus.pending.value,
                ),
            )
        return cursor.rowcount

    def open_prose_gate_for_session_event(
        self,
        *,
        session_id: str,
        source_event_id: str,
        pov_character_ids: Iterable[str],
    ) -> int:
        ids = tuple(
            dict.fromkeys(item for item in pov_character_ids if item)
        )
        if not ids:
            return 0
        with self._connect() as db:
            db.executemany(
                """
                INSERT OR IGNORE INTO image_prose_receipts (
                    session_id, source_event_id, pov_character_id, opened_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    (
                        session_id,
                        source_event_id,
                        pov_character_id,
                        time.time(),
                    )
                    for pov_character_id in ids
                ),
            )
            transaction_ids = {
                str(row["transaction_id"])
                for row in db.execute(
                    """
                    SELECT DISTINCT transaction_id FROM image_director_runs
                    WHERE session_id = ? AND source_event_id = ?
                    UNION
                    SELECT DISTINCT transaction_id FROM image_jobs
                    WHERE session_id = ? AND source_event_id = ?
                    """,
                    (
                        session_id,
                        source_event_id,
                        session_id,
                        source_event_id,
                    ),
                ).fetchall()
            }
        return sum(
            self.open_prose_gate(
                transaction_id=transaction_id,
                source_event_id=source_event_id,
                pov_character_ids=ids,
            )
            for transaction_id in transaction_ids
        )

    def claim_next_delivery(
        self,
        kind: ImageDeliveryKind,
    ) -> tuple[ImageDelivery, ImageGenerationJob] | None:
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """
                SELECT d.* FROM image_deliveries AS d
                JOIN image_jobs AS j ON j.job_id = d.job_id
                JOIN image_transactions AS t
                  ON t.transaction_id = d.transaction_id
                WHERE d.status = ? AND d.delivery_kind = ?
                  AND d.prose_rendered = 1 AND d.next_attempt_at <= ?
                  AND j.status = ? AND t.status = 'committed'
                ORDER BY d.created_at, d.delivery_id
                LIMIT 1
                """,
                (
                    ImageDeliveryStatus.pending.value,
                    kind.value,
                    now,
                    ImageGenerationStatus.succeeded.value,
                ),
            ).fetchone()
            if row is None:
                db.commit()
                return None
            cursor = db.execute(
                """
                UPDATE image_deliveries
                SET status = ?, attempts = attempts + 1, updated_at = ?
                WHERE delivery_id = ? AND status = ?
                """,
                (
                    ImageDeliveryStatus.delivering.value,
                    now,
                    row["delivery_id"],
                    ImageDeliveryStatus.pending.value,
                ),
            )
            if cursor.rowcount != 1:
                db.rollback()
                return None
            delivery_row = db.execute(
                """
                SELECT * FROM image_deliveries WHERE delivery_id = ?
                """,
                (row["delivery_id"],),
            ).fetchone()
            job_row = db.execute(
                "SELECT * FROM image_jobs WHERE job_id = ?",
                (row["job_id"],),
            ).fetchone()
            db.commit()
        if delivery_row is None or job_row is None:
            return None
        return _delivery_from_row(delivery_row), _job_from_row(job_row)

    def heartbeat_delivery(self, delivery_id: str) -> bool:
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE image_deliveries SET updated_at = ?
                WHERE delivery_id = ? AND status = ?
                """,
                (
                    time.time(),
                    delivery_id,
                    ImageDeliveryStatus.delivering.value,
                ),
            )
        return cursor.rowcount == 1

    def release_delivery(self, delivery_id: str) -> ImageDelivery | None:
        now = time.time()
        with self._connect() as db:
            row = db.execute(
                """
                SELECT attempts FROM image_deliveries WHERE delivery_id = ?
                """,
                (delivery_id,),
            ).fetchone()
            attempts = int(row["attempts"] or 1) if row else 1
            retry_at = now + min(300.0, float(2 ** min(attempts, 8)))
            db.execute(
                """
                UPDATE image_deliveries
                SET status = ?, updated_at = ?, next_attempt_at = ?
                WHERE delivery_id = ? AND status = ?
                """,
                (
                    ImageDeliveryStatus.pending.value,
                    now,
                    retry_at,
                    delivery_id,
                    ImageDeliveryStatus.delivering.value,
                ),
            )
        return self.get_delivery(delivery_id)

    def mark_delivered(self, delivery_id: str) -> ImageDelivery | None:
        now = time.time()
        with self._connect() as db:
            db.execute(
                """
                UPDATE image_deliveries
                SET status = ?, delivered_at = ?, updated_at = ?
                WHERE delivery_id = ? AND status = ?
                """,
                (
                    ImageDeliveryStatus.delivered.value,
                    now,
                    now,
                    delivery_id,
                    ImageDeliveryStatus.delivering.value,
                ),
            )
        return self.get_delivery(delivery_id)

    def cancel_delivery(self, delivery_id: str) -> ImageDelivery | None:
        with self._connect() as db:
            db.execute(
                """
                UPDATE image_deliveries
                SET status = ?, updated_at = ?
                WHERE delivery_id = ? AND status IN (?, ?)
                """,
                (
                    ImageDeliveryStatus.cancelled.value,
                    time.time(),
                    delivery_id,
                    ImageDeliveryStatus.pending.value,
                    ImageDeliveryStatus.delivering.value,
                ),
            )
        return self.get_delivery(delivery_id)

    def delivery_is_current(self, delivery_id: str) -> bool:
        with self._connect() as db:
            row = db.execute(
                """
                SELECT 1 FROM image_deliveries AS d
                JOIN image_jobs AS j ON j.job_id = d.job_id
                JOIN image_transactions AS t
                  ON t.transaction_id = d.transaction_id
                WHERE d.delivery_id = ? AND d.status = ?
                  AND j.status = ? AND t.status = 'committed'
                """,
                (
                    delivery_id,
                    ImageDeliveryStatus.delivering.value,
                    ImageGenerationStatus.succeeded.value,
                ),
            ).fetchone()
        return row is not None

    def cancel_after(self, session_id: str, turn_index: int) -> int:
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """
                UPDATE image_transactions
                SET status = 'cancelled', updated_at = ?
                WHERE session_id = ? AND source_turn_index > ?
                  AND status != 'cancelled'
                """,
                (now, session_id, turn_index),
            )
            cursor = db.execute(
                """
                UPDATE image_jobs
                SET status = ?, error_code = 'rewound',
                    completed_at = ?, updated_at = ?
                WHERE session_id = ? AND source_turn_index > ?
                  AND status IN (?, ?, ?)
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
                ),
            )
            db.execute(
                """
                UPDATE image_deliveries
                SET status = ?, updated_at = ?
                WHERE session_id = ? AND source_turn_index > ?
                  AND status IN (?, ?)
                """,
                (
                    ImageDeliveryStatus.cancelled.value,
                    now,
                    session_id,
                    turn_index,
                    ImageDeliveryStatus.pending.value,
                    ImageDeliveryStatus.delivering.value,
                ),
            )
            db.execute(
                """
                UPDATE image_director_runs
                SET status = 'cancelled', error_code = 'rewound',
                    updated_at = ?
                WHERE session_id = ? AND source_turn_index > ?
                  AND status IN ('queued', 'running', 'succeeded')
                """,
                (now, session_id, turn_index),
            )
            db.execute(
                """
                DELETE FROM image_prose_receipts
                WHERE session_id = ? AND source_event_id IN (
                    SELECT source_event_id FROM image_director_runs
                    WHERE session_id = ? AND source_turn_index > ?
                    UNION
                    SELECT source_event_id FROM image_jobs
                    WHERE session_id = ? AND source_turn_index > ?
                )
                """,
                (
                    session_id,
                    session_id,
                    turn_index,
                    session_id,
                    turn_index,
                ),
            )
            db.execute(
                """
                DELETE FROM image_prose_gates
                WHERE transaction_id IN (
                    SELECT transaction_id FROM image_transactions
                    WHERE session_id = ? AND source_turn_index > ?
                )
                """,
                (session_id, turn_index),
            )
            self._retire_cancelled_candidates(db, session_id=session_id)
            self._restore_rewound_identity_retirements(
                db,
                session_id=session_id,
                turn_index=turn_index,
            )
            db.commit()
        return cursor.rowcount

    def reconcile_lineage(
        self,
        *,
        session_id: str,
        event_fingerprints: Iterable[str],
        event_ids: Iterable[str] = (),
    ) -> int:
        fingerprints = set(event_fingerprints)
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT j.job_id, j.source_event_fingerprint
                FROM image_jobs AS j
                JOIN image_transactions AS t
                  ON t.transaction_id = j.transaction_id
                WHERE j.session_id = ? AND t.lineage_bound = 1
                  AND j.status IN (?, ?, ?)
                """,
                (
                    session_id,
                    ImageGenerationStatus.queued.value,
                    ImageGenerationStatus.running.value,
                    ImageGenerationStatus.succeeded.value,
                ),
            ).fetchall()
        stale_ids = [
            row["job_id"]
            for row in rows
            if row["source_event_fingerprint"] not in fingerprints
        ]
        for job_id in stale_ids:
            self.mark_cancelled(job_id, "event_not_in_lineage")
        with self._connect() as db:
            if fingerprints:
                placeholders = ",".join("?" for _ in fingerprints)
                db.execute(
                    f"""
                    UPDATE image_director_runs
                    SET status = 'cancelled',
                        error_code = 'event_not_in_lineage',
                        updated_at = ?
                    WHERE session_id = ?
                      AND source_event_fingerprint NOT IN ({placeholders})
                      AND status IN ('queued', 'running', 'succeeded')
                    """,
                    (time.time(), session_id, *sorted(fingerprints)),
                )
            else:
                db.execute(
                    """
                    UPDATE image_director_runs
                    SET status = 'cancelled',
                        error_code = 'event_not_in_lineage',
                        updated_at = ?
                    WHERE session_id = ?
                      AND status IN ('queued', 'running', 'succeeded')
                    """,
                    (time.time(), session_id),
                )
            current_ids = set(event_ids)
            if current_ids:
                placeholders = ",".join("?" for _ in current_ids)
                db.execute(
                    f"""
                    DELETE FROM image_prose_receipts
                    WHERE session_id = ?
                      AND source_event_id NOT IN ({placeholders})
                    """,
                    (session_id, *sorted(current_ids)),
                )
            else:
                db.execute(
                    """
                    DELETE FROM image_prose_receipts WHERE session_id = ?
                    """,
                    (session_id,),
                )
        return len(stale_ids)

    def cancel_session(self, session_id: str) -> int:
        with self._connect() as db:
            transaction_ids = [
                row["transaction_id"]
                for row in db.execute(
                    """
                    SELECT transaction_id FROM image_transactions
                    WHERE session_id = ? AND status != 'cancelled'
                    """,
                    (session_id,),
                ).fetchall()
            ]
        cancelled = sum(
            self.cancel_transaction(
                transaction_id,
                reason="session_ended",
            )
            for transaction_id in transaction_ids
        )
        with self._connect() as db:
            db.execute(
                "DELETE FROM image_reviewed_identity_bindings "
                "WHERE session_id = ?",
                (session_id,),
            )
            db.execute(
                "DELETE FROM image_reviewed_location_bindings "
                "WHERE session_id = ?",
                (session_id,),
            )
            db.execute(
                "DELETE FROM image_reviewed_references WHERE session_id = ?",
                (session_id,),
            )
            db.execute(
                "DELETE FROM image_prose_receipts WHERE session_id = ?",
                (session_id,),
            )
            db.execute(
                "DELETE FROM image_identity_policies WHERE session_id = ?",
                (session_id,),
            )
        return cancelled

    def cancel_discord_destination(
        self,
        *,
        session_id: str,
        session_channel_id: int,
    ) -> int:
        # Generated deliveries resolve the session's current private channel
        # at send time; no stale channel id is frozen into the durable row.
        # Unbinding that session therefore invalidates every pending Discord
        # destination created under the old binding.
        del session_channel_id
        now = time.time()
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE image_deliveries
                SET status = ?, updated_at = ?
                WHERE session_id = ? AND delivery_kind = ?
                  AND status IN (?, ?)
                """,
                (
                    ImageDeliveryStatus.cancelled.value,
                    now,
                    session_id,
                    ImageDeliveryKind.discord.value,
                    ImageDeliveryStatus.pending.value,
                    ImageDeliveryStatus.delivering.value,
                ),
            )
        return cursor.rowcount

    def recent_illustrations(
        self,
        session_id: str,
        *,
        viewer_character_ids: Sequence[str],
        limit: int = 8,
    ) -> list[str]:
        viewers = tuple(
            dict.fromkeys(
                character_id
                for character_id in viewer_character_ids
                if character_id
            )
        )
        if not viewers or limit <= 0:
            return []
        placeholders = ",".join("?" for _ in viewers)
        with self._connect() as db:
            rows = db.execute(
                f"""
                SELECT j.request_json FROM image_jobs AS j
                JOIN image_deliveries AS d ON d.job_id = j.job_id
                WHERE j.session_id = ? AND j.status = ?
                  AND d.pov_character_id IN ({placeholders})
                GROUP BY j.job_id
                HAVING COUNT(DISTINCT d.pov_character_id) = ?
                ORDER BY j.completed_at DESC LIMIT ?
                """,
                (
                    session_id,
                    ImageGenerationStatus.succeeded.value,
                    *viewers,
                    len(viewers),
                    max(0, limit),
                ),
            ).fetchall()
        return [
            _recent_illustration_summary(
                ImageGenerationRequest.model_validate_json(
                    row["request_json"]
                )
            )
            for row in rows
        ]

    def rendered_event_image_status(
        self,
        *,
        session_id: str,
        rendered_event_ids_by_pov: dict[str, Sequence[str]],
    ) -> tuple[bool, bool]:
        """Return ``(has_work, ready)`` for event images tied to rendered prose."""

        povs_by_event: dict[str, set[str]] = {}
        for pov_character_id, event_ids in rendered_event_ids_by_pov.items():
            pov = str(pov_character_id or "").strip()
            if not pov:
                continue
            for event_id in event_ids:
                event = str(event_id or "").strip()
                if event:
                    povs_by_event.setdefault(event, set()).add(pov)
        if not povs_by_event:
            return False, True

        placeholders = ",".join("?" for _ in povs_by_event)
        with self._connect() as db:
            run_rows = db.execute(
                f"""
                SELECT * FROM image_director_runs
                WHERE session_id = ?
                  AND source_event_id IN ({placeholders})
                  AND status != 'cancelled'
                """,
                (session_id, *povs_by_event),
            ).fetchall()
            job_rows = db.execute(
                f"""
                SELECT * FROM image_jobs
                WHERE session_id = ?
                  AND source_event_id IN ({placeholders})
                  AND status != ?
                """,
                (
                    session_id,
                    *povs_by_event,
                    ImageGenerationStatus.cancelled.value,
                ),
            ).fetchall()

        relevant_runs: list[DurableDirectorRun] = []
        for row in run_rows:
            run = _director_run_from_row(row)
            povs = povs_by_event.get(run.projection.event_id, set())
            if povs.intersection(run.projection.viewer_character_ids):
                relevant_runs.append(run)
        if not relevant_runs:
            return False, True

        jobs_by_run: dict[tuple[str, str, str], list[ImageGenerationJob]] = {}
        for row in job_rows:
            job = _job_from_row(row)
            key = (
                job.request.transaction_id,
                job.request.source_event_id,
                job.request.source_event_fingerprint,
            )
            jobs_by_run.setdefault(key, []).append(job)

        for run in relevant_runs:
            if run.status in {"queued", "running"}:
                return True, False
            if run.status != "succeeded" or run.output is None:
                continue
            expected_count = len(run.output.requests)
            if expected_count == 0:
                continue
            key = (
                run.projection.transaction_id,
                run.projection.event_id,
                run.projection.event_fingerprint,
            )
            jobs = jobs_by_run.get(key, [])
            if len({
                job.request.request_ordinal for job in jobs
            }) < expected_count:
                return True, False
            if any(
                job.status in {
                    ImageGenerationStatus.queued,
                    ImageGenerationStatus.running,
                }
                for job in jobs
            ):
                return True, False
        return True, True

    def create_identity_candidate(
        self,
        *,
        job: ImageGenerationJob,
        character_id: str,
    ) -> IdentityReferenceCandidate | None:
        if job.artifact is None:
            raise ValueError("identity candidates require a successful artifact")
        if job.request.kind != "portrait":
            raise ValueError("only individual portraits establish identity")
        if job.request.subject_character_ids != [character_id]:
            raise ValueError("portrait identity subject mismatch")
        candidate_id = f"imgref_{job.job_id.removeprefix('img_')}"
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            policy = db.execute(
                """
                SELECT blocked, minimum_source_turn
                FROM image_identity_policies
                WHERE session_id = ? AND character_id = ?
                """,
                (job.request.session_id, character_id),
            ).fetchone()
            if policy is not None and (
                bool(policy["blocked"])
                or job.request.source_turn_index
                < int(policy["minimum_source_turn"])
            ):
                db.rollback()
                return None
            prior = db.execute(
                """
                SELECT candidate_id FROM image_identity_candidates
                WHERE session_id = ? AND character_id = ? AND active = 1
                ORDER BY created_at DESC LIMIT 1
                """,
                (job.request.session_id, character_id),
            ).fetchone()
            reroll_of = job.request.reroll_of_reference_id or (
                prior["candidate_id"] if prior else ""
            )
            db.execute(
                """
                UPDATE image_identity_candidates
                SET active = 0,
                    status = CASE
                        WHEN status = ? THEN ?
                        ELSE status
                    END,
                    updated_at = ?
                WHERE session_id = ? AND character_id = ? AND active = 1
                """,
                (
                    IdentityReferenceStatus.provisional.value,
                    IdentityReferenceStatus.retained.value,
                    now,
                    job.request.session_id,
                    character_id,
                ),
            )
            db.execute(
                """
                INSERT OR IGNORE INTO image_identity_candidates (
                    candidate_id, session_id, character_id, job_id,
                    artifact_json, status, active, reminder_required,
                    reroll_of_reference_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 1, 1, ?, ?, ?)
                """,
                (
                    candidate_id,
                    job.request.session_id,
                    character_id,
                    job.job_id,
                    job.artifact.model_dump_json(),
                    IdentityReferenceStatus.provisional.value,
                    reroll_of,
                    now,
                    now,
                ),
            )
            row = db.execute(
                """
                SELECT * FROM image_identity_candidates
                WHERE candidate_id = ?
                """,
                (candidate_id,),
            ).fetchone()
            db.commit()
        if row is None:
            raise RuntimeError("identity candidate insert failed")
        return _candidate_from_row(row)

    def get_identity_candidate(
        self,
        candidate_id: str,
    ) -> IdentityReferenceCandidate | None:
        with self._connect() as db:
            row = db.execute(
                """
                SELECT * FROM image_identity_candidates
                WHERE candidate_id = ?
                """,
                (candidate_id,),
            ).fetchone()
        return _candidate_from_row(row) if row is not None else None

    def active_identity_candidate(
        self,
        *,
        session_id: str,
        character_id: str,
    ) -> IdentityReferenceCandidate | None:
        with self._connect() as db:
            row = db.execute(
                """
                SELECT * FROM image_identity_candidates
                WHERE session_id = ? AND character_id = ? AND active = 1
                ORDER BY created_at DESC LIMIT 1
                """,
                (session_id, character_id),
            ).fetchone()
        return _candidate_from_row(row) if row is not None else None

    def active_identity_character_ids(self, session_id: str) -> set[str]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT DISTINCT character_id FROM image_identity_candidates
                WHERE session_id = ? AND active = 1
                """,
                (session_id,),
            ).fetchall()
            reviewed_rows = db.execute(
                """
                SELECT DISTINCT b.character_id
                FROM image_reviewed_identity_bindings AS b
                LEFT JOIN image_identity_policies AS p
                  ON p.session_id = b.session_id
                 AND p.character_id = b.character_id
                WHERE b.session_id = ?
                  AND COALESCE(p.blocked, 0) = 0
                """,
                (session_id,),
            ).fetchall()
        return {
            str(row["character_id"])
            for row in (*rows, *reviewed_rows)
        }

    def replace_reviewed_references(
        self,
        *,
        session_id: str,
        references: dict[
            str,
            tuple[FrozenReferenceInput, str, str],
        ],
        identity_bindings: dict[str, Sequence[str]],
        location_bindings: dict[str, Sequence[str]],
    ) -> None:
        """Atomically mirror checkpoint-owned reviewed bindings into runtime."""

        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "DELETE FROM image_reviewed_identity_bindings "
                "WHERE session_id = ?",
                (session_id,),
            )
            db.execute(
                "DELETE FROM image_reviewed_location_bindings "
                "WHERE session_id = ?",
                (session_id,),
            )
            db.execute(
                "DELETE FROM image_reviewed_references WHERE session_id = ?",
                (session_id,),
            )
            for reference_id, (
                frozen,
                purpose,
                scope,
            ) in references.items():
                if frozen.reference_id != reference_id:
                    raise ValueError("reviewed reference id mismatch")
                db.execute(
                    """
                    INSERT INTO image_reviewed_references (
                        session_id, reference_id, frozen_json, purpose,
                        scope, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        reference_id,
                        frozen.model_dump_json(),
                        purpose,
                        scope,
                        now,
                    ),
                )
            for character_id, reference_ids in identity_bindings.items():
                for priority, reference_id in enumerate(reference_ids):
                    if reference_id not in references:
                        raise ValueError(
                            "reviewed identity binding has no frozen reference"
                        )
                    db.execute(
                        """
                        INSERT INTO image_reviewed_identity_bindings (
                            session_id, character_id, reference_id,
                            priority, updated_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            character_id,
                            reference_id,
                            priority,
                            now,
                        ),
                    )
            for location_label, reference_ids in location_bindings.items():
                for priority, reference_id in enumerate(reference_ids):
                    if reference_id not in references:
                        raise ValueError(
                            "reviewed location binding has no frozen reference"
                        )
                    db.execute(
                        """
                        INSERT INTO image_reviewed_location_bindings (
                            session_id, location_label, reference_id,
                            priority, updated_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            location_label,
                            reference_id,
                            priority,
                            now,
                        ),
                    )
            db.commit()

    def reviewed_identity_reference(
        self,
        *,
        session_id: str,
        character_id: str,
    ) -> FrozenReferenceInput | None:
        with self._connect() as db:
            row = db.execute(
                """
                SELECT r.frozen_json
                FROM image_reviewed_identity_bindings AS b
                JOIN image_reviewed_references AS r
                  ON r.session_id = b.session_id
                 AND r.reference_id = b.reference_id
                LEFT JOIN image_identity_policies AS p
                  ON p.session_id = b.session_id
                 AND p.character_id = b.character_id
                WHERE b.session_id = ? AND b.character_id = ?
                  AND COALESCE(p.blocked, 0) = 0
                ORDER BY b.priority, b.reference_id
                LIMIT 1
                """,
                (session_id, character_id),
            ).fetchone()
        return (
            FrozenReferenceInput.model_validate_json(row["frozen_json"])
            if row is not None
            else None
        )

    def reviewed_reference(
        self,
        *,
        session_id: str,
        reference_id: str,
    ) -> FrozenReferenceInput | None:
        with self._connect() as db:
            row = db.execute(
                """
                SELECT frozen_json
                FROM image_reviewed_references
                WHERE session_id = ? AND reference_id = ?
                """,
                (session_id, reference_id),
            ).fetchone()
        return (
            FrozenReferenceInput.model_validate_json(row["frozen_json"])
            if row is not None
            else None
        )

    def suppress_reviewed_identity_binding(
        self,
        *,
        session_id: str,
        character_id: str,
    ) -> int:
        """Remove one session binding without deleting its shared file."""

        with self._connect() as db:
            cursor = db.execute(
                """
                DELETE FROM image_reviewed_identity_bindings
                WHERE session_id = ? AND character_id = ?
                """,
                (session_id, character_id),
            )
        return cursor.rowcount

    def reviewed_identity_binding(
        self,
        *,
        session_id: str,
        reference_id: str,
    ) -> tuple[str, FrozenReferenceInput] | None:
        with self._connect() as db:
            row = db.execute(
                """
                SELECT b.character_id, r.frozen_json
                FROM image_reviewed_identity_bindings AS b
                JOIN image_reviewed_references AS r
                  ON r.session_id = b.session_id
                 AND r.reference_id = b.reference_id
                LEFT JOIN image_identity_policies AS p
                  ON p.session_id = b.session_id
                 AND p.character_id = b.character_id
                WHERE b.session_id = ? AND b.reference_id = ?
                  AND COALESCE(p.blocked, 0) = 0
                """,
                (session_id, reference_id),
            ).fetchone()
        if row is None:
            return None
        return (
            str(row["character_id"]),
            FrozenReferenceInput.model_validate_json(row["frozen_json"]),
        )

    def reviewed_location_references(
        self,
        *,
        session_id: str,
        location_label: str,
    ) -> list[FrozenReferenceInput]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT r.frozen_json
                FROM image_reviewed_location_bindings AS b
                JOIN image_reviewed_references AS r
                  ON r.session_id = b.session_id
                 AND r.reference_id = b.reference_id
                WHERE b.session_id = ? AND b.location_label = ?
                ORDER BY b.priority, b.reference_id
                """,
                (session_id, location_label),
            ).fetchall()
        return [
            FrozenReferenceInput.model_validate_json(row["frozen_json"])
            for row in rows
        ]

    def active_reviewed_location_labels(self, session_id: str) -> set[str]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT DISTINCT location_label
                FROM image_reviewed_location_bindings
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchall()
        return {str(row["location_label"]) for row in rows}

    def lock_identity_candidate(
        self,
        *,
        session_id: str,
        candidate_id: str,
    ) -> IdentityReferenceCandidate:
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """
                SELECT * FROM image_identity_candidates
                WHERE candidate_id = ? AND session_id = ?
                """,
                (candidate_id, session_id),
            ).fetchone()
            if row is None:
                db.rollback()
                raise KeyError(f"unknown identity candidate: {candidate_id}")
            if row["status"] == IdentityReferenceStatus.retired.value:
                db.rollback()
                raise ValueError("retired identity candidates cannot be locked")
            policy = db.execute(
                """
                SELECT blocked FROM image_identity_policies
                WHERE session_id = ? AND character_id = ?
                """,
                (session_id, row["character_id"]),
            ).fetchone()
            if policy is not None and bool(policy["blocked"]):
                db.rollback()
                raise ValueError(
                    "culled character identities cannot be locked"
                )
            source_job = db.execute(
                "SELECT status FROM image_jobs WHERE job_id = ?",
                (row["job_id"],),
            ).fetchone()
            if (
                source_job is None
                or source_job["status"]
                != ImageGenerationStatus.succeeded.value
            ):
                db.rollback()
                raise ValueError(
                    "identity candidate source is no longer current"
                )
            db.execute(
                """
                UPDATE image_identity_candidates
                SET active = 0, updated_at = ?
                WHERE session_id = ? AND character_id = ?
                """,
                (now, session_id, row["character_id"]),
            )
            db.execute(
                """
                UPDATE image_identity_candidates
                SET status = ?, active = 1, reminder_required = 0,
                    updated_at = ?
                WHERE candidate_id = ?
                """,
                (
                    IdentityReferenceStatus.locked.value,
                    now,
                    candidate_id,
                ),
            )
            locked = db.execute(
                """
                SELECT * FROM image_identity_candidates
                WHERE candidate_id = ?
                """,
                (candidate_id,),
            ).fetchone()
            db.commit()
        return _candidate_from_row(locked)

    def retire_character_identity(
        self,
        *,
        session_id: str,
        character_id: str,
        source_turn_index: int,
    ) -> int:
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                """
                SELECT candidate_id, status, reminder_required
                FROM image_identity_candidates
                WHERE session_id = ? AND character_id = ? AND active = 1
                ORDER BY created_at DESC LIMIT 1
                """,
                (session_id, character_id),
            ).fetchone()
            cursor = db.execute(
                """
                UPDATE image_identity_candidates
                SET status = ?, active = 0, reminder_required = 0,
                    updated_at = ?
                WHERE session_id = ? AND character_id = ?
                  AND status != ?
                """,
                (
                    IdentityReferenceStatus.retired.value,
                    now,
                    session_id,
                    character_id,
                    IdentityReferenceStatus.retired.value,
                ),
            )
            db.execute(
                """
                INSERT INTO image_identity_policies (
                    session_id, character_id, blocked,
                    minimum_source_turn, retirement_source_turn,
                    prior_candidate_id, prior_status,
                    prior_reminder_required, updated_at
                ) VALUES (?, ?, 1, 0, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, character_id) DO UPDATE SET
                    blocked = 1,
                    minimum_source_turn = 0,
                    retirement_source_turn = CASE
                        WHEN image_identity_policies.blocked = 1
                        THEN image_identity_policies.retirement_source_turn
                        ELSE excluded.retirement_source_turn
                    END,
                    prior_candidate_id = CASE
                        WHEN image_identity_policies.blocked = 1
                        THEN image_identity_policies.prior_candidate_id
                        ELSE excluded.prior_candidate_id
                    END,
                    prior_status = CASE
                        WHEN image_identity_policies.blocked = 1
                        THEN image_identity_policies.prior_status
                        ELSE excluded.prior_status
                    END,
                    prior_reminder_required = CASE
                        WHEN image_identity_policies.blocked = 1
                        THEN image_identity_policies.prior_reminder_required
                        ELSE excluded.prior_reminder_required
                    END,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    character_id,
                    max(0, int(source_turn_index)),
                    str(prior["candidate_id"]) if prior is not None else "",
                    str(prior["status"]) if prior is not None else "",
                    (
                        int(prior["reminder_required"])
                        if prior is not None
                        else 0
                    ),
                    now,
                ),
            )
        return cursor.rowcount

    def allow_character_identity_after(
        self,
        *,
        session_id: str,
        character_id: str,
        minimum_source_turn: int,
    ) -> None:
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO image_identity_policies (
                    session_id, character_id, blocked,
                    minimum_source_turn, retirement_source_turn,
                    prior_candidate_id, prior_status,
                    prior_reminder_required, updated_at
                ) VALUES (?, ?, 0, ?, 0, '', '', 0, ?)
                ON CONFLICT(session_id, character_id) DO UPDATE SET
                    blocked = 0,
                    minimum_source_turn = excluded.minimum_source_turn,
                    retirement_source_turn = 0,
                    prior_candidate_id = '',
                    prior_status = '',
                    prior_reminder_required = 0,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    character_id,
                    max(0, int(minimum_source_turn)),
                    time.time(),
                ),
            )

    def all_jobs(self) -> list[ImageGenerationJob]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM image_jobs ORDER BY created_at, job_id"
            ).fetchall()
        return [_job_from_row(row) for row in rows]

    def _retire_cancelled_candidates(
        self,
        db: sqlite3.Connection,
        *,
        session_id: str,
    ) -> None:
        now = time.time()
        db.execute(
            """
            UPDATE image_identity_candidates
            SET status = ?, active = 0, updated_at = ?
            WHERE session_id = ? AND job_id IN (
                SELECT job_id FROM image_jobs WHERE status = ?
            )
            """,
            (
                IdentityReferenceStatus.retired.value,
                now,
                session_id,
                ImageGenerationStatus.cancelled.value,
            ),
        )
        characters = [
            row["character_id"]
            for row in db.execute(
                """
                SELECT DISTINCT character_id FROM image_identity_candidates
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchall()
        ]
        for character_id in characters:
            active = db.execute(
                """
                SELECT 1 FROM image_identity_candidates
                WHERE session_id = ? AND character_id = ? AND active = 1
                """,
                (session_id, character_id),
            ).fetchone()
            if active is not None:
                continue
            prior = db.execute(
                """
                SELECT candidate_id FROM image_identity_candidates
                WHERE session_id = ? AND character_id = ?
                  AND status != ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (
                    session_id,
                    character_id,
                    IdentityReferenceStatus.retired.value,
                ),
            ).fetchone()
            if prior is not None:
                db.execute(
                    """
                    UPDATE image_identity_candidates
                    SET active = 1, updated_at = ? WHERE candidate_id = ?
                    """,
                    (now, prior["candidate_id"]),
                )

    def _restore_rewound_identity_retirements(
        self,
        db: sqlite3.Connection,
        *,
        session_id: str,
        turn_index: int,
    ) -> None:
        rows = db.execute(
            """
            SELECT character_id, prior_candidate_id, prior_status,
                   prior_reminder_required
            FROM image_identity_policies
            WHERE session_id = ? AND blocked = 1
              AND retirement_source_turn > ?
            """,
            (session_id, turn_index),
        ).fetchall()
        now = time.time()
        for row in rows:
            candidate_id = str(row["prior_candidate_id"] or "")
            if candidate_id:
                candidate = db.execute(
                    """
                    SELECT c.candidate_id, j.status AS job_status
                    FROM image_identity_candidates AS c
                    JOIN image_jobs AS j ON j.job_id = c.job_id
                    WHERE c.candidate_id = ?
                    """,
                    (candidate_id,),
                ).fetchone()
                if (
                    candidate is not None
                    and candidate["job_status"]
                    == ImageGenerationStatus.succeeded.value
                ):
                    db.execute(
                        """
                        UPDATE image_identity_candidates
                        SET active = 0, updated_at = ?
                        WHERE session_id = ? AND character_id = ?
                        """,
                        (now, session_id, row["character_id"]),
                    )
                    db.execute(
                        """
                        UPDATE image_identity_candidates
                        SET status = ?, active = 1,
                            reminder_required = ?, updated_at = ?
                        WHERE candidate_id = ?
                        """,
                        (
                            str(row["prior_status"]),
                            int(row["prior_reminder_required"]),
                            now,
                            candidate_id,
                        ),
                    )
            db.execute(
                """
                DELETE FROM image_identity_policies
                WHERE session_id = ? AND character_id = ?
                """,
                (session_id, row["character_id"]),
            )

    def _migrate_directly(self, db: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in db.execute(
                """
                SELECT name FROM sqlite_master WHERE type = 'table'
                """
            ).fetchall()
        }
        version = ""
        if "image_store_meta" in existing:
            row = db.execute(
                """
                SELECT value FROM image_store_meta
                WHERE key = 'schema_version'
                """
            ).fetchone()
            version = str(row["value"]) if row else ""
        if not existing:
            return
        if (
            version == IMAGE_JOB_SCHEMA_VERSION
            and self._schema_layout_is_current(db, existing)
        ):
            return
        # Pre-RC direct retirement: generated image request contracts are
        # disposable runtime queue state and are not migrated across hard
        # storage boundaries.
        for table in (
            "image_deliveries",
            "image_prose_gates",
            "image_prose_receipts",
            "image_reviewed_identity_bindings",
            "image_reviewed_location_bindings",
            "image_reviewed_references",
            "image_identity_candidates",
            "image_identity_policies",
            "image_jobs",
            "image_director_runs",
            "image_transactions",
            "image_eligible_beats",
            "image_store_meta",
        ):
            db.execute(f"DROP TABLE IF EXISTS {table}")

    @staticmethod
    def _schema_layout_is_current(
        db: sqlite3.Connection,
        existing: set[str],
    ) -> bool:
        for table, required_columns in _CURRENT_SCHEMA_COLUMNS.items():
            if table not in existing:
                return False
            columns = {
                str(row["name"])
                for row in db.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()
            }
            if not required_columns.issubset(columns):
                return False
        return True

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db


def _recent_illustration_summary(
    request: ImageGenerationRequest,
) -> str:
    marker = "Visible scene:"
    scene = request.prompt
    if marker in scene:
        scene = scene.split(marker, 1)[1].split("\n\n", 1)[0]
    scene = " ".join(scene.split())[:500].rstrip()
    subjects = ", ".join(request.subject_character_ids) or "none"
    return (
        f"title={request.title}; kind={request.kind}; "
        f"subjects={subjects}; scene={scene}"
    )


def _job_from_row(row: sqlite3.Row) -> ImageGenerationJob:
    artifact_json = str(row["artifact_json"] or "")
    return ImageGenerationJob(
        job_id=str(row["job_id"]),
        request=ImageGenerationRequest.model_validate_json(
            row["request_json"]
        ),
        status=ImageGenerationStatus(str(row["status"])),
        artifact=(
            GeneratedImageArtifact.model_validate_json(artifact_json)
            if artifact_json
            else None
        ),
        error_code=str(row["error_code"] or ""),
        attempts=int(row["attempts"] or 0),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        started_at=(
            float(row["started_at"])
            if row["started_at"] is not None
            else None
        ),
        completed_at=(
            float(row["completed_at"])
            if row["completed_at"] is not None
            else None
        ),
    )


def _delivery_from_row(row: sqlite3.Row) -> ImageDelivery:
    return ImageDelivery(
        delivery_id=str(row["delivery_id"]),
        job_id=str(row["job_id"]),
        session_id=str(row["session_id"]),
        source_turn_index=int(row["source_turn_index"]),
        pov_character_id=str(row["pov_character_id"]),
        delivery_kind=ImageDeliveryKind(str(row["delivery_kind"])),
        delivery=json.loads(str(row["delivery_json"])),
        status=ImageDeliveryStatus(str(row["status"])),
        attempts=int(row["attempts"] or 0),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        delivered_at=(
            float(row["delivered_at"])
            if row["delivered_at"] is not None
            else None
        ),
    )


def _candidate_from_row(row: sqlite3.Row) -> IdentityReferenceCandidate:
    return IdentityReferenceCandidate(
        candidate_id=str(row["candidate_id"]),
        session_id=str(row["session_id"]),
        character_id=str(row["character_id"]),
        job_id=str(row["job_id"]),
        artifact=GeneratedImageArtifact.model_validate_json(
            row["artifact_json"]
        ),
        status=IdentityReferenceStatus(str(row["status"])),
        active=bool(row["active"]),
        reminder_required=bool(row["reminder_required"]),
        reroll_of_reference_id=str(row["reroll_of_reference_id"] or ""),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


def _director_run_from_row(row: sqlite3.Row) -> DurableDirectorRun:
    output_json = str(row["output_json"] or "")
    return DurableDirectorRun(
        run_id=str(row["run_id"]),
        projection=VisibleEventProjection.from_storage_dict(
            json.loads(str(row["projection_json"]))
        ),
        status=str(row["status"]),
        output=(
            ImageDirectorOutput.model_validate_json(output_json)
            if output_json
            else None
        ),
        error_code=str(row["error_code"] or ""),
        attempts=int(row["attempts"] or 0),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


def _clean_reason(value: object) -> str:
    return " ".join(str(value or "cancelled").split())[:200]


def _chmod_private(path: Path, *, directory: bool) -> None:
    try:
        os.chmod(path, 0o700 if directory else 0o600)
    except OSError:
        pass
