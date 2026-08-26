from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
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
    materialized_at     REAL,
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

CREATE TABLE IF NOT EXISTS image_director_run_jobs (
    run_id              TEXT NOT NULL,
    attempt             INTEGER NOT NULL,
    request_ordinal     INTEGER NOT NULL,
    job_id              TEXT NOT NULL,
    finalized           INTEGER NOT NULL DEFAULT 0,
    created_at          REAL NOT NULL,
    PRIMARY KEY(run_id, attempt, request_ordinal),
    UNIQUE(run_id, attempt, job_id),
    FOREIGN KEY(run_id) REFERENCES image_director_runs(run_id)
        ON DELETE CASCADE,
    FOREIGN KEY(job_id) REFERENCES image_jobs(job_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS image_director_run_jobs_job_idx
ON image_director_run_jobs(job_id);

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
ON image_deliveries(status, next_attempt_at, created_at);

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


@dataclass(frozen=True)
class VisualNovelStageResolution:
    action: str
    artifact: GeneratedImageArtifact | None
    source_run_id: str = ""
    fallback_reason: str = ""

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
        "materialized_at",
    },
    "image_director_run_jobs": {
        "run_id",
        "attempt",
        "request_ordinal",
        "job_id",
        "finalized",
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
                WHERE transaction_id = ?
                  AND status IN (
                    'queued', 'running', 'materializing', 'succeeded'
                  )
                """,
                (_clean_reason(reason), now, transaction_id),
            )
            db.execute(
                """
                DELETE FROM image_director_run_jobs
                WHERE finalized = 0 AND run_id IN (
                    SELECT run_id FROM image_director_runs
                    WHERE transaction_id = ? AND status = 'cancelled'
                )
                """,
                (transaction_id,),
            )
            if transaction is not None:
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

    def admit_director_request(
        self,
        request: ImageGenerationRequest,
        *,
        run_id: str,
        attempt: int,
        queue_limit: int,
        per_session_queue_limit: int,
    ) -> ImageGenerationJob | None:
        """Atomically enqueue and provisionally link one director request.

        The provisional link closes the crash window between job admission and
        run finalization. Recovery can therefore retire only work owned by the
        expired attempt, while a same-position sibling that shares the exact
        dedupe job keeps that work alive through its own link.
        """

        now = time.time()
        job_id = f"img_{request.dedupe_key[:32]}"
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            run_row = db.execute(
                """
                SELECT * FROM image_director_runs
                WHERE run_id = ? AND attempts = ? AND status = 'materializing'
                """,
                (run_id, attempt),
            ).fetchone()
            if run_row is None:
                raise RuntimeError("director admission attempt is stale")
            run = _director_run_from_row(run_row)
            if run.output is None:
                raise RuntimeError("director admission output is unavailable")

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

            row = db.execute(
                "SELECT * FROM image_jobs WHERE dedupe_key = ?",
                (request.dedupe_key,),
            ).fetchone()
            needs_capacity = row is None
            if row is not None:
                existing = _job_from_row(row)
                if existing.request != request:
                    raise RuntimeError("image job dedupe contract mismatch")
                retryable_failed = bool(
                    existing.status == ImageGenerationStatus.failed
                    and existing.attempts < 2
                )
                retryable_cancelled = bool(
                    existing.status == ImageGenerationStatus.cancelled
                    and (
                        existing.error_code == "director_attempt_expired"
                        or existing.error_code.startswith("materialization_")
                    )
                )
                needs_capacity = retryable_failed or retryable_cancelled

            if needs_capacity and not self._image_capacity_available(
                db,
                session_id=request.session_id,
                queue_limit=queue_limit,
                per_session_queue_limit=per_session_queue_limit,
            ):
                db.commit()
                return None

            if row is None:
                db.execute(
                    """
                    INSERT INTO image_jobs (
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
                    "SELECT * FROM image_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
            elif needs_capacity:
                db.execute(
                    """
                    UPDATE image_jobs
                    SET status = ?, error_code = '', updated_at = ?,
                        started_at = NULL, completed_at = NULL
                    WHERE job_id = ? AND status IN (?, ?)
                    """,
                    (
                        ImageGenerationStatus.queued.value,
                        now,
                        job_id,
                        ImageGenerationStatus.failed.value,
                        ImageGenerationStatus.cancelled.value,
                    ),
                )
                row = db.execute(
                    "SELECT * FROM image_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
            if row is None:
                raise RuntimeError(
                    "director admission did not produce an image job"
                )

            job = _job_from_row(row)
            ordinal = request.request_ordinal
            link_error = _director_job_link_error(run, ordinal, job)
            if link_error:
                raise ValueError(f"director admission image-job {link_error}")
            db.execute(
                """
                INSERT INTO image_director_run_jobs (
                    run_id, attempt, request_ordinal, job_id,
                    finalized, created_at
                ) VALUES (?, ?, ?, ?, 0, ?)
                ON CONFLICT(run_id, attempt, request_ordinal) DO NOTHING
                """,
                (run_id, attempt, ordinal, job.job_id, now),
            )
            association = db.execute(
                """
                SELECT job_id FROM image_director_run_jobs
                WHERE run_id = ? AND attempt = ? AND request_ordinal = ?
                """,
                (run_id, attempt, ordinal),
            ).fetchone()
            if association is None or association["job_id"] != job.job_id:
                raise RuntimeError("director admission ordinal changed")
            db.execute(
                """
                UPDATE image_transactions SET updated_at = ?
                WHERE transaction_id = ?
                """,
                (now, request.transaction_id),
            )
            db.commit()
        return job

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
                  AND NOT EXISTS (
                    SELECT 1 FROM image_director_runs AS predecessor
                    JOIN image_transactions AS predecessor_transaction
                      ON predecessor_transaction.transaction_id =
                         predecessor.transaction_id
                    WHERE predecessor.session_id = r.session_id
                      AND predecessor.status IN ('running', 'materializing')
                      AND predecessor_transaction.status != 'cancelled'
                      AND (
                        predecessor.source_turn_index < r.source_turn_index
                        OR (
                          predecessor.source_turn_index = r.source_turn_index
                          AND predecessor.source_event_sequence <
                              r.source_event_sequence
                        )
                      )
                  )
                ORDER BY r.source_turn_index, r.source_event_sequence,
                         r.created_at, r.run_id
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
        *,
        attempt: int,
    ) -> DurableDirectorRun | None:
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE image_director_runs
                SET status = 'materializing', output_json = ?,
                    materialized_at = NULL, error_code = '', updated_at = ?
                WHERE run_id = ? AND attempts = ? AND status = 'running'
                """,
                (output.model_dump_json(), time.time(), run_id, attempt),
            )
            if cursor.rowcount != 1:
                return None
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
        *,
        attempt: int,
        projection: VisibleEventProjection,
        admitted_job_ids: Sequence[str],
    ) -> DurableDirectorRun:
        """Atomically bind one director run to its admitted generation jobs.

        A materializing director run exists before request admission finishes.
        The nullable ``materialized_at`` boundary distinguishes that in-progress
        interval from a completed materialization which admitted zero jobs.
        """
        job_ids = [str(job_id).strip() for job_id in admitted_job_ids]
        if any(not job_id for job_id in job_ids):
            raise ValueError("materialized image job ids must not be empty")
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("director materialization contains duplicate jobs")

        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """
                SELECT * FROM image_director_runs WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    f"director materialization run is unavailable: {run_id}"
                )
            run = _director_run_from_row(row)
            if run.projection != projection:
                raise ValueError("director materialization projection mismatch")
            if run.attempts != attempt:
                raise RuntimeError("director materialization attempt is stale")
            if run.status == "cancelled":
                db.commit()
                return run
            existing_rows = db.execute(
                """
                SELECT request_ordinal, job_id, finalized
                FROM image_director_run_jobs
                WHERE run_id = ? AND attempt = ?
                ORDER BY request_ordinal
                """,
                (run_id, attempt),
            ).fetchall()
            if run.status == "succeeded" and run.materialized_at is not None:
                existing_job_ids = [
                    str(item["job_id"])
                    for item in existing_rows
                    if bool(item["finalized"])
                ]
                if existing_job_ids != job_ids:
                    raise RuntimeError(
                        "director materialization was already finalized differently"
                    )
                db.commit()
                return run
            if run.status != "materializing" or run.output is None:
                raise RuntimeError(
                    "director materialization requires a materializing run"
                )
            if any(bool(item["finalized"]) for item in existing_rows):
                raise RuntimeError(
                    "materializing director run has finalized image jobs"
                )
            provisional_job_ids = [
                str(item["job_id"]) for item in existing_rows
            ]
            if provisional_job_ids != job_ids:
                raise RuntimeError(
                    "director materialization admissions changed before finalization"
                )
            for item in existing_rows:
                job_row = db.execute(
                    "SELECT * FROM image_jobs WHERE job_id = ?",
                    (item["job_id"],),
                ).fetchone()
                if job_row is None:
                    raise RuntimeError(
                        "director materialization references an unavailable "
                        "image job"
                    )
                link_error = _director_job_link_error(
                    run,
                    int(item["request_ordinal"]),
                    _job_from_row(job_row),
                )
                if link_error:
                    raise ValueError(
                        "director materialization image-job " + link_error
                    )
            db.execute(
                """
                UPDATE image_director_run_jobs SET finalized = 1
                WHERE run_id = ? AND attempt = ? AND finalized = 0
                """,
                (run_id, attempt),
            )
            cursor = db.execute(
                """
                UPDATE image_director_runs
                SET status = 'succeeded', materialized_at = ?, updated_at = ?
                WHERE run_id = ? AND attempts = ?
                  AND status = 'materializing'
                  AND materialized_at IS NULL
                """,
                (now, now, run_id, attempt),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("director materialization finalization raced")
            finalized_row = db.execute(
                "SELECT * FROM image_director_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            db.commit()
        if finalized_row is None:
            raise RuntimeError("director materialization final row is unavailable")
        return _director_run_from_row(finalized_row)

    def heartbeat_director_run(self, run_id: str, *, attempt: int) -> bool:
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE image_director_runs SET updated_at = ?
                WHERE run_id = ? AND attempts = ?
                  AND status IN ('running', 'materializing')
                """,
                (time.time(), run_id, attempt),
            )
        return cursor.rowcount == 1

    def fail_director_run(
        self,
        run_id: str,
        error_code: str,
        *,
        attempt: int,
    ) -> DurableDirectorRun | None:
        run, _ = self.fail_director_run_with_cleanup(
            run_id,
            error_code,
            attempt=attempt,
        )
        return run

    def fail_director_run_with_cleanup(
        self,
        run_id: str,
        error_code: str,
        *,
        attempt: int,
    ) -> tuple[DurableDirectorRun | None, tuple[tuple[str, int], ...]]:
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                """
                SELECT * FROM image_director_runs
                WHERE run_id = ? AND attempts = ?
                  AND status IN ('running', 'materializing')
                """,
                (run_id, attempt),
            ).fetchone()
            if current is None:
                db.commit()
                return None, ()
            job_ids = [
                str(row["job_id"])
                for row in db.execute(
                    """
                    SELECT job_id FROM image_director_run_jobs
                    WHERE run_id = ? AND attempt = ? AND finalized = 0
                    """,
                    (run_id, attempt),
                ).fetchall()
            ]
            db.execute(
                """
                DELETE FROM image_director_run_jobs
                WHERE run_id = ? AND attempt = ? AND finalized = 0
                """,
                (run_id, attempt),
            )
            cancelled_attempts = self._cancel_unlinked_jobs(
                db,
                job_ids=job_ids,
                error_code=_clean_reason(error_code),
                now=now,
            )
            cursor = db.execute(
                """
                UPDATE image_director_runs
                SET status = 'failed', error_code = ?, updated_at = ?
                WHERE run_id = ? AND attempts = ?
                  AND status IN ('running', 'materializing')
                """,
                (_clean_reason(error_code), now, run_id, attempt),
            )
            if cursor.rowcount != 1:
                db.rollback()
                return None, ()
            self._retire_cancelled_candidates(
                db,
                session_id=str(current["session_id"]),
            )
            row = db.execute(
                """
                SELECT * FROM image_director_runs WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            db.commit()
        return (
            _director_run_from_row(row) if row is not None else None,
            cancelled_attempts,
        )

    def recover_expired_director_runs(
        self,
        *,
        lease_seconds: float = 300,
    ) -> int:
        recovered, _ = self.recover_expired_director_runs_with_cleanup(
            lease_seconds=lease_seconds,
        )
        return recovered

    def recover_expired_director_runs_with_cleanup(
        self,
        *,
        lease_seconds: float = 300,
    ) -> tuple[int, tuple[tuple[str, int], ...]]:
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cutoff = now - max(1.0, lease_seconds)
            expired = db.execute(
                """
                SELECT run_id, attempts, session_id
                FROM image_director_runs
                WHERE status IN ('running', 'materializing')
                  AND updated_at < ?
                """,
                (cutoff,),
            ).fetchall()
            recovered = 0
            affected_sessions: set[str] = set()
            cancelled_attempts: list[tuple[str, int]] = []
            for row in expired:
                run_id = str(row["run_id"])
                attempt = int(row["attempts"])
                job_ids = [
                    str(item["job_id"])
                    for item in db.execute(
                        """
                        SELECT job_id FROM image_director_run_jobs
                        WHERE run_id = ? AND attempt = ? AND finalized = 0
                        """,
                        (run_id, attempt),
                    ).fetchall()
                ]
                db.execute(
                    """
                    DELETE FROM image_director_run_jobs
                    WHERE run_id = ? AND attempt = ? AND finalized = 0
                    """,
                    (run_id, attempt),
                )
                cancelled_attempts.extend(
                    self._cancel_unlinked_jobs(
                        db,
                        job_ids=job_ids,
                        error_code="director_attempt_expired",
                        now=now,
                    )
                )
                cursor = db.execute(
                    """
                    UPDATE image_director_runs
                    SET status = 'queued', output_json = '',
                        materialized_at = NULL, error_code = '', updated_at = ?
                    WHERE run_id = ? AND attempts = ?
                      AND status IN ('running', 'materializing')
                      AND updated_at < ?
                    """,
                    (now, run_id, attempt, cutoff),
                )
                recovered += cursor.rowcount
                affected_sessions.add(str(row["session_id"]))
            for session_id in affected_sessions:
                self._retire_cancelled_candidates(db, session_id=session_id)
            db.commit()
        return recovered, tuple(cancelled_attempts)

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
            db.execute(
                """
                INSERT OR IGNORE INTO image_deliveries (
                    delivery_id, job_id, session_id, transaction_id,
                    source_event_id, source_turn_index, pov_character_id,
                    delivery_kind, delivery_json, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def fail_unserviceable_finalized_jobs(
        self,
        *,
        error_code: str = "worker_unavailable",
    ) -> int:
        """Fail finalized jobs while the caller holds the queue-owner lease."""

        now = time.time()
        reason = _clean_reason(error_code or "worker_unavailable")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                """
                SELECT DISTINCT job.job_id
                FROM image_jobs AS job
                JOIN image_director_run_jobs AS association
                  ON association.job_id = job.job_id
                JOIN image_director_runs AS run
                  ON run.run_id = association.run_id
                WHERE association.finalized = 1
                  AND association.attempt = run.attempts
                  AND run.status = 'succeeded'
                  AND run.materialized_at IS NOT NULL
                  AND job.status IN (?, ?)
                """,
                (
                    ImageGenerationStatus.queued.value,
                    ImageGenerationStatus.running.value,
                ),
            ).fetchall()
            failed = 0
            for row in rows:
                cursor = db.execute(
                    """
                    UPDATE image_jobs
                    SET status = ?, error_code = ?, completed_at = ?,
                        updated_at = ?
                    WHERE job_id = ? AND status IN (?, ?)
                    """,
                    (
                        ImageGenerationStatus.failed.value,
                        reason,
                        now,
                        now,
                        row["job_id"],
                        ImageGenerationStatus.queued.value,
                        ImageGenerationStatus.running.value,
                    ),
                )
                failed += cursor.rowcount
            db.commit()
        return failed

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
                  AND d.next_attempt_at <= ?
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
                  AND status IN (
                    'queued', 'running', 'materializing', 'succeeded'
                  )
                """,
                (now, session_id, turn_index),
            )
            db.execute(
                """
                DELETE FROM image_director_run_jobs
                WHERE finalized = 0 AND run_id IN (
                    SELECT run_id FROM image_director_runs
                    WHERE session_id = ? AND source_turn_index > ?
                      AND status = 'cancelled'
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
            db.execute("BEGIN IMMEDIATE")
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
                      AND status IN (
                        'queued', 'running', 'materializing', 'succeeded'
                      )
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
                      AND status IN (
                        'queued', 'running', 'materializing', 'succeeded'
                      )
                    """,
                    (time.time(), session_id),
                )
            provisional_job_ids = [
                str(row["job_id"])
                for row in db.execute(
                    """
                    SELECT association.job_id
                    FROM image_director_run_jobs AS association
                    JOIN image_director_runs AS run
                      ON run.run_id = association.run_id
                    WHERE run.session_id = ? AND run.status = 'cancelled'
                      AND association.finalized = 0
                    """,
                    (session_id,),
                ).fetchall()
            ]
            db.execute(
                """
                DELETE FROM image_director_run_jobs
                WHERE finalized = 0 AND run_id IN (
                    SELECT run_id FROM image_director_runs
                    WHERE session_id = ? AND status = 'cancelled'
                )
                """,
                (session_id,),
            )
            self._cancel_unlinked_jobs(
                db,
                job_ids=provisional_job_ids,
                error_code="event_not_in_lineage",
                now=time.time(),
            )
            self._retire_cancelled_candidates(db, session_id=session_id)
            db.commit()
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

    def visual_novel_stage_context_before_run(
        self,
        run_id: str,
    ) -> list[str]:
        """Describe the shared stage strictly preceding one director run."""

        with self._connect() as db:
            current_row = db.execute(
                "SELECT * FROM image_director_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if current_row is None:
                raise RuntimeError(
                    f"visual-novel director run is unavailable: {run_id}"
                )
            current = _director_run_from_row(current_row)
            viewers = {
                str(character_id).strip()
                for character_id in current.projection.viewer_character_ids
                if str(character_id).strip()
            }
            if not viewers:
                return []
            rows = db.execute(
                """
                SELECT r.* FROM image_director_runs AS r
                JOIN image_transactions AS t
                  ON t.transaction_id = r.transaction_id
                WHERE r.session_id = ? AND t.status = 'committed'
                  AND r.status != 'cancelled'
                  AND (
                    r.source_turn_index < ?
                    OR (
                      r.source_turn_index = ?
                      AND r.source_event_sequence < ?
                    )
                  )
                ORDER BY r.source_turn_index DESC,
                         r.source_event_sequence DESC,
                         r.created_at DESC,
                         r.run_id DESC
                """,
                (
                    current.projection.session_id,
                    current.projection.source_turn_index,
                    current.projection.source_turn_index,
                    current.projection.event_sequence,
                ),
            ).fetchall()

        runs = [
            run
            for row in rows
            if (
                (run := _director_run_from_row(row)).projection.presentation_mode
                == "visual_novel"
            )
        ]

        def _effective_stage(viewer: str):
            inherited = False
            for run in runs:
                if viewer not in run.projection.viewer_character_ids:
                    continue
                if (
                    run.status != "succeeded"
                    or run.output is None
                    or run.materialized_at is None
                ):
                    return "neutral", "", None, inherited
                action = run.output.stage_action
                if action == "reuse":
                    inherited = True
                    continue
                if action != "replace":
                    return "neutral", "", None, inherited
                job = self._visual_novel_job_for_run(run)
                if job is None:
                    return "neutral", "", None, inherited
                return "active", job.job_id, job, inherited
            return "missing", "", None, inherited

        effective = [_effective_stage(viewer) for viewer in sorted(viewers)]
        if all(state == "missing" for state, *_ in effective):
            return []
        if any(state != "active" for state, *_ in effective):
            return ["current_stage=neutral; reason=no shared compatible stage"]
        job_ids = {job_id for _, job_id, _, _ in effective}
        if len(job_ids) != 1:
            return ["current_stage=neutral; reason=no shared compatible stage"]
        job = effective[0][2]
        if job is None:
            return ["current_stage=neutral; reason=no shared compatible stage"]
        inherited = any(item[3] for item in effective)
        state = "reused" if inherited else "active"
        return [
            f"current_stage={state}; {_recent_illustration_summary(job.request)}"
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
                SELECT r.* FROM image_director_runs AS r
                JOIN image_transactions AS t
                  ON t.transaction_id = r.transaction_id
                WHERE r.session_id = ?
                  AND r.source_event_id IN ({placeholders})
                  AND r.status != 'cancelled'
                  AND t.status != 'cancelled'
                """,
                (session_id, *povs_by_event),
            ).fetchall()

        relevant_runs: list[DurableDirectorRun] = []
        for row in run_rows:
            run = _director_run_from_row(row)
            povs = povs_by_event.get(run.projection.event_id, set())
            if povs.intersection(run.projection.viewer_character_ids):
                relevant_runs.append(run)
        if not relevant_runs:
            return False, True

        run_ids = [run.run_id for run in relevant_runs]
        run_placeholders = ",".join("?" for _ in run_ids)
        with self._connect() as db:
            linked_rows = db.execute(
                f"""
                SELECT association.run_id AS linked_run_id,
                       association.request_ordinal AS linked_request_ordinal,
                       job.*
                FROM image_director_run_jobs AS association
                JOIN image_jobs AS job ON job.job_id = association.job_id
                WHERE association.run_id IN ({run_placeholders})
                  AND association.finalized = 1
                ORDER BY association.run_id, association.request_ordinal
                """,
                run_ids,
            ).fetchall()

        jobs_by_run: dict[str, list[tuple[int, ImageGenerationJob]]] = {}
        for row in linked_rows:
            job = _job_from_row(row)
            jobs_by_run.setdefault(str(row["linked_run_id"]), []).append(
                (int(row["linked_request_ordinal"]), job)
            )

        for run in relevant_runs:
            if run.status in {"queued", "running", "materializing"}:
                return True, False
            if run.status != "succeeded":
                continue
            if run.output is None:
                raise RuntimeError(
                    "succeeded director run has no durable output"
                )
            if run.materialized_at is None:
                return True, False
            jobs = jobs_by_run.get(run.run_id, [])
            for ordinal, job in jobs:
                link_error = _director_job_link_error(run, ordinal, job)
                if link_error:
                    raise RuntimeError(
                        "finalized director run has inconsistent image-job "
                        f"links: {link_error}"
                    )
            if any(
                job.status in {
                    ImageGenerationStatus.queued,
                    ImageGenerationStatus.running,
                }
                for _, job in jobs
            ):
                return True, False
        return True, True

    def resolve_visual_novel_stage(
        self,
        *,
        session_id: str,
        pov_character_id: str,
        rendered_event_ids: Sequence[str],
    ) -> VisualNovelStageResolution:
        """Resolve the explicit stage transition for one accepted POV render.

        A failed or absent current transition never silently inherits an older
        image. Only a current ``reuse`` transition may walk backward to the
        last successful ``replace`` not superseded by ``clear`` or a failed
        replacement.
        """

        event_ids = {
            str(event_id).strip()
            for event_id in rendered_event_ids
            if str(event_id).strip()
        }
        if not event_ids:
            return VisualNovelStageResolution(
                action="clear",
                artifact=None,
                fallback_reason="no_rendered_events",
            )
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT r.* FROM image_director_runs AS r
                JOIN image_transactions AS t
                  ON t.transaction_id = r.transaction_id
                WHERE r.session_id = ? AND t.status = 'committed'
                ORDER BY r.source_turn_index DESC,
                         r.source_event_sequence DESC,
                         r.created_at DESC,
                         r.run_id DESC
                """,
                (session_id,),
            ).fetchall()

        runs: list[DurableDirectorRun] = []
        for row in rows:
            run = _director_run_from_row(row)
            if (
                run.projection.presentation_mode == "visual_novel"
                and pov_character_id in run.projection.viewer_character_ids
            ):
                runs.append(run)
        current_index = next(
            (
                index
                for index, run in enumerate(runs)
                if run.projection.event_id in event_ids
            ),
            None,
        )
        if current_index is None:
            return VisualNovelStageResolution(
                action="clear",
                artifact=None,
                fallback_reason="missing_current_transition",
            )
        current = runs[current_index]
        if (
            current.status != "succeeded"
            or current.output is None
            or current.materialized_at is None
        ):
            return VisualNovelStageResolution(
                action="clear",
                artifact=None,
                source_run_id=current.run_id,
                fallback_reason="current_direction_failed",
            )
        action = current.output.stage_action
        if action == "clear":
            return VisualNovelStageResolution(
                action=action,
                artifact=None,
                source_run_id=current.run_id,
                fallback_reason="stage_cleared",
            )
        if action == "replace":
            artifact = self._visual_novel_artifact_for_run(current)
            return VisualNovelStageResolution(
                action=action,
                artifact=artifact,
                source_run_id=current.run_id,
                fallback_reason=("" if artifact is not None else "replacement_failed"),
            )
        if action != "reuse":
            return VisualNovelStageResolution(
                action="clear",
                artifact=None,
                source_run_id=current.run_id,
                fallback_reason="invalid_stage_transition",
            )

        for prior in runs[current_index + 1:]:
            if (
                prior.status != "succeeded"
                or prior.output is None
                or prior.materialized_at is None
            ):
                return VisualNovelStageResolution(
                    action="reuse",
                    artifact=None,
                    source_run_id=current.run_id,
                    fallback_reason="reused_stage_transition_failed",
                )
            prior_action = prior.output.stage_action
            if prior_action == "reuse":
                continue
            if prior_action == "clear":
                return VisualNovelStageResolution(
                    action="reuse",
                    artifact=None,
                    source_run_id=current.run_id,
                    fallback_reason="reused_stage_was_cleared",
                )
            if prior_action == "replace":
                artifact = self._visual_novel_artifact_for_run(prior)
                return VisualNovelStageResolution(
                    action="reuse",
                    artifact=artifact,
                    source_run_id=current.run_id,
                    fallback_reason=(
                        "" if artifact is not None else "reused_replacement_failed"
                    ),
                )
        return VisualNovelStageResolution(
            action="reuse",
            artifact=None,
            source_run_id=current.run_id,
            fallback_reason="no_prior_stage",
        )

    def _visual_novel_artifact_for_run(
        self,
        run: DurableDirectorRun,
    ) -> GeneratedImageArtifact | None:
        job = self._visual_novel_job_for_run(run)
        return job.artifact if job is not None else None

    def _visual_novel_job_for_run(
        self,
        run: DurableDirectorRun,
    ) -> ImageGenerationJob | None:
        if run.materialized_at is None:
            return None
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT association.request_ordinal AS linked_request_ordinal,
                       job.*
                FROM image_director_run_jobs AS association
                JOIN image_jobs AS job ON job.job_id = association.job_id
                WHERE association.run_id = ? AND association.attempt = ?
                  AND association.finalized = 1
                ORDER BY association.request_ordinal
                """,
                (run.run_id, run.attempts),
            ).fetchall()
        for row in rows:
            job = _job_from_row(row)
            link_error = _director_job_link_error(
                run,
                int(row["linked_request_ordinal"]),
                job,
            )
            if link_error:
                raise RuntimeError(
                    "visual-novel director run has inconsistent image-job "
                    f"links: {link_error}"
                )
            if (
                job.status == ImageGenerationStatus.succeeded
                and job.request.kind != "portrait"
                and job.artifact is not None
            ):
                return job
        return None

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

    @staticmethod
    def _image_capacity_available(
        db: sqlite3.Connection,
        *,
        session_id: str,
        queue_limit: int,
        per_session_queue_limit: int,
    ) -> bool:
        active = db.execute(
            """
            SELECT COUNT(*) FROM image_jobs WHERE status IN (?, ?)
            """,
            (
                ImageGenerationStatus.queued.value,
                ImageGenerationStatus.running.value,
            ),
        ).fetchone()
        if active is not None and int(active[0]) >= max(1, queue_limit):
            return False
        session_active = db.execute(
            """
            SELECT COUNT(*) FROM image_jobs
            WHERE session_id = ? AND status IN (?, ?)
            """,
            (
                session_id,
                ImageGenerationStatus.queued.value,
                ImageGenerationStatus.running.value,
            ),
        ).fetchone()
        return bool(
            session_active is None
            or int(session_active[0]) < max(1, per_session_queue_limit)
        )

    @staticmethod
    def _cancel_unlinked_jobs(
        db: sqlite3.Connection,
        *,
        job_ids: Sequence[str],
        error_code: str,
        now: float,
    ) -> tuple[tuple[str, int], ...]:
        cancelled_attempts: list[tuple[str, int]] = []
        for job_id in dict.fromkeys(job_ids):
            retained = db.execute(
                """
                SELECT 1 FROM image_director_run_jobs
                WHERE job_id = ? LIMIT 1
                """,
                (job_id,),
            ).fetchone()
            if retained is not None:
                continue
            state = db.execute(
                "SELECT attempts FROM image_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            cursor = db.execute(
                """
                UPDATE image_jobs
                SET status = ?, error_code = ?, completed_at = ?, updated_at = ?
                WHERE job_id = ? AND status IN (?, ?, ?)
                """,
                (
                    ImageGenerationStatus.cancelled.value,
                    error_code,
                    now,
                    now,
                    job_id,
                    ImageGenerationStatus.queued.value,
                    ImageGenerationStatus.running.value,
                    ImageGenerationStatus.succeeded.value,
                ),
            )
            if cursor.rowcount == 1 and state is not None:
                cancelled_attempts.append(
                    (job_id, int(state["attempts"] or 0))
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
        return tuple(cancelled_attempts)

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
            "image_director_run_jobs",
            "image_deliveries",
            "image_prose_receipts",
            "image_prose_gates",
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


def _director_job_link_error(
    run: DurableDirectorRun,
    request_ordinal: int,
    job: ImageGenerationJob,
) -> str:
    """Return the durable contract error for one run-to-job association."""

    if run.output is None:
        return "run output is unavailable"
    request = job.request
    if request.request_ordinal != request_ordinal:
        return "request ordinal mismatch"
    provenance = (
        request.session_id,
        request.transaction_id,
        request.source_event_id,
        request.source_event_fingerprint,
        request.source_event_sequence,
        request.source_turn_index,
    )
    projection = run.projection
    expected_provenance = (
        projection.session_id,
        projection.transaction_id,
        projection.event_id,
        projection.event_fingerprint,
        projection.event_sequence,
        projection.source_turn_index,
    )
    if provenance != expected_provenance:
        return "provenance mismatch"
    if request_ordinal >= len(run.output.requests):
        return "request ordinal is out of range"
    direction = run.output.requests[request_ordinal]
    direction_contract = (
        direction.kind,
        direction.title,
        tuple(direction.subject_character_ids),
        direction.generation_mode,
    )
    request_contract = (
        request.kind,
        request.title,
        tuple(request.subject_character_ids),
        request.generation_mode,
    )
    if direction_contract != request_contract:
        return "direction mismatch"
    if not (
        request.prompt == direction.scene_prompt
        or request.prompt.startswith(f"{direction.scene_prompt}\n\n")
    ):
        return "scene prompt mismatch"
    frozen_reference_ids = {
        reference.reference_id for reference in request.reference_inputs
    }
    if not set(direction.reference_ids).issubset(frozen_reference_ids):
        return "reference selection mismatch"
    return ""


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
        materialized_at=(
            float(row["materialized_at"])
            if row["materialized_at"] is not None
            else None
        ),
    )


def _clean_reason(value: object) -> str:
    return " ".join(str(value or "cancelled").split())[:200]


def _chmod_private(path: Path, *, directory: bool) -> None:
    try:
        os.chmod(path, 0o700 if directory else 0o600)
    except OSError:
        pass
