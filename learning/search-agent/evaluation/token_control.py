"""Stage-aware token partitions for TongAgent evaluation runs.

The global :class:`evaluation.budget.ExecutionBudget` remains authoritative.
This module adds a stricter, TongAgent-only partition beneath that ceiling so
one required subquestion cannot consume the allocation of another or the
reserved final-synthesis pool.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any, Final, Literal


ModelStage = Literal[
    "planner",
    "control_status",
    "research_step",
    "evidence_selection",
    "final_synthesis",
    "final_extractor",
]

DEFAULT_TONGAGENT_STAGE_OUTPUT_CAPS: Final[dict[ModelStage, int]] = {
    "planner": 1_000,
    "control_status": 500,
    "research_step": 1_800,
    "evidence_selection": 800,
    "final_synthesis": 2_200,
    "final_extractor": 500,
}


@dataclass(frozen=True)
class PartitionReservation:
    """One provisional debit against a stage partition."""

    reservation_id: str
    bucket: str
    stage: ModelStage
    subquestion_id: str | None
    reserved_tokens: int


@dataclass(frozen=True)
class PartitionDenial:
    """Auditable partition rejection before provider I/O."""

    bucket: str
    stage: ModelStage
    subquestion_id: str | None
    requested_tokens: int
    available_tokens: int
    snapshot: dict[str, Any]


class StageTokenController:
    """Partition a fixed global token ceiling across required research work."""

    schema_version: Final[int] = 1

    def __init__(
        self,
        *,
        total_token_limit: int,
        stage_output_caps: Mapping[ModelStage, int] | None = None,
        final_ratio: float = 0.25,
        safety_ratio: float = 0.10,
    ) -> None:
        if total_token_limit <= 0:
            raise ValueError("total_token_limit must be positive")
        if not (0.0 < final_ratio < 1.0):
            raise ValueError("final_ratio must be between zero and one")
        if not (0.0 < safety_ratio < 1.0):
            raise ValueError("safety_ratio must be between zero and one")
        if final_ratio + safety_ratio >= 1.0:
            raise ValueError("final and safety ratios must leave research capacity")
        caps = dict(DEFAULT_TONGAGENT_STAGE_OUTPUT_CAPS)
        if stage_output_caps is not None:
            caps.update(stage_output_caps)
        if any(not isinstance(value, int) or value <= 0 for value in caps.values()):
            raise ValueError("stage output caps must be positive integers")

        self._lock = RLock()
        self.total_token_limit = total_token_limit
        self.stage_output_caps = caps
        self.final_ratio = final_ratio
        self.safety_ratio = safety_ratio
        self._configured = False
        self._active_subquestion_id: str | None = None
        self._limits: dict[str, int] = {}
        self._actual: defaultdict[str, int] = defaultdict(int)
        self._reserved: defaultdict[str, int] = defaultdict(int)
        self._calls: defaultdict[str, int] = defaultdict(int)
        self._stage_actual: defaultdict[str, int] = defaultdict(int)
        self._stage_calls: defaultdict[str, int] = defaultdict(int)
        self._reservations: dict[str, PartitionReservation] = {}
        self._subquestion_ids: tuple[str, ...] = ()

    @property
    def active_subquestion_id(self) -> str | None:
        with self._lock:
            return self._active_subquestion_id

    def output_cap(self, stage: ModelStage) -> int:
        return int(self.stage_output_caps[stage])

    def configure(self, subquestion_ids: Sequence[str]) -> None:
        """Create deterministic SQ/final/buffer partitions after planning."""

        normalized = tuple(dict.fromkeys(str(item) for item in subquestion_ids if item))
        if not normalized:
            raise ValueError("token partitions require at least one subquestion")
        with self._lock:
            if self._configured:
                if normalized != self._subquestion_ids:
                    raise ValueError(
                        "token partitions cannot be reconfigured for different SQs"
                    )
                return
            final_limit = max(
                self.output_cap("final_synthesis") + self.output_cap("final_extractor"),
                int(self.total_token_limit * self.final_ratio),
            )
            safety_limit = max(1, int(self.total_token_limit * self.safety_ratio))
            research_limit = self.total_token_limit - final_limit - safety_limit
            if research_limit < len(normalized):
                raise ValueError(
                    "token ceiling is too small for required SQ partitions"
                )
            base, remainder = divmod(research_limit, len(normalized))
            self._limits = {
                f"sq:{subquestion_id}": base + (index < remainder)
                for index, subquestion_id in enumerate(normalized)
            }
            self._limits["final"] = final_limit
            self._limits["buffer"] = safety_limit
            self._subquestion_ids = normalized
            self._configured = True
            self._validate_existing_debits_locked()

    def activate(self, subquestion_id: str | None) -> None:
        with self._lock:
            if subquestion_id is not None and self._configured:
                bucket = f"sq:{subquestion_id}"
                if bucket not in self._limits:
                    raise ValueError(
                        f"unknown token-partition subquestion: {subquestion_id}"
                    )
            self._active_subquestion_id = subquestion_id

    def reserve(
        self,
        *,
        stage: ModelStage,
        token_reservation: int,
        subquestion_id: str | None = None,
    ) -> PartitionReservation | PartitionDenial:
        """Reserve against one local bucket without touching the global budget."""

        if token_reservation <= 0:
            raise ValueError("token_reservation must be positive")
        with self._lock:
            active = subquestion_id or self._active_subquestion_id
            bucket = self._bucket_for_locked(stage, active)
            if self._configured:
                limit = self._limits[bucket]
                used = self._actual[bucket] + self._reserved[bucket]
                available = max(0, limit - used)
                if token_reservation > available:
                    return PartitionDenial(
                        bucket=bucket,
                        stage=stage,
                        subquestion_id=active,
                        requested_tokens=token_reservation,
                        available_tokens=available,
                        snapshot=self._snapshot_locked(),
                    )
            reservation = PartitionReservation(
                reservation_id=f"partition-{uuid.uuid4().hex}",
                bucket=bucket,
                stage=stage,
                subquestion_id=active,
                reserved_tokens=token_reservation,
            )
            self._reservations[reservation.reservation_id] = reservation
            self._reserved[bucket] += token_reservation
            self._calls[bucket] += 1
            self._stage_calls[stage] += 1
            return reservation

    def cancel(self, reservation: PartitionReservation) -> None:
        with self._lock:
            current = self._reservations.pop(reservation.reservation_id, None)
            if current is None:
                return
            self._reserved[current.bucket] -= current.reserved_tokens

    def settle(
        self,
        reservation: PartitionReservation,
        *,
        actual_tokens: int | None,
        charge_reservation_if_unknown: bool,
    ) -> None:
        with self._lock:
            current = self._reservations.pop(reservation.reservation_id, None)
            if current is None:
                return
            self._reserved[current.bucket] -= current.reserved_tokens
            charged = (
                current.reserved_tokens
                if actual_tokens is None and charge_reservation_if_unknown
                else max(0, int(actual_tokens or 0))
            )
            self._actual[current.bucket] += charged
            self._stage_actual[current.stage] += charged

    def can_start_subquestion(self, subquestion_id: str) -> bool:
        """Require enough room for one compact research turn."""

        with self._lock:
            if not self._configured:
                return True
            bucket = f"sq:{subquestion_id}"
            if bucket not in self._limits:
                return False
            remaining = (
                self._limits[bucket] - self._actual[bucket] - self._reserved[bucket]
            )
            minimum_turn = self.output_cap("research_step") + 1_000
            return remaining >= minimum_turn

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def _bucket_for_locked(
        self,
        stage: ModelStage,
        subquestion_id: str | None,
    ) -> str:
        if stage in {"final_synthesis", "final_extractor"}:
            return "final"
        if stage == "planner":
            return "buffer"
        if subquestion_id:
            return f"sq:{subquestion_id}"
        return "buffer"

    def _validate_existing_debits_locked(self) -> None:
        for bucket, limit in self._limits.items():
            used = self._actual[bucket] + self._reserved[bucket]
            if used > limit:
                raise ValueError(
                    f"existing token debit exceeds {bucket} partition: {used}>{limit}"
                )

    def _snapshot_locked(self) -> dict[str, Any]:
        buckets = sorted(
            set(self._limits)
            | set(self._actual)
            | set(self._reserved)
            | {"final", "buffer"}
        )
        return {
            "schema_version": self.schema_version,
            "configured": self._configured,
            "total_token_limit": self.total_token_limit,
            "stage_output_caps": dict(self.stage_output_caps),
            "final_ratio": self.final_ratio,
            "safety_ratio": self.safety_ratio,
            "active_subquestion_id": self._active_subquestion_id,
            "subquestion_ids": list(self._subquestion_ids),
            "buckets": {
                bucket: {
                    "limit_tokens": self._limits.get(bucket),
                    "actual_tokens": self._actual[bucket],
                    "reserved_tokens": self._reserved[bucket],
                    "remaining_tokens": (
                        None
                        if bucket not in self._limits
                        else max(
                            0,
                            self._limits[bucket]
                            - self._actual[bucket]
                            - self._reserved[bucket],
                        )
                    ),
                    "model_calls": self._calls[bucket],
                }
                for bucket in buckets
            },
            "stage_usage": {
                stage: {
                    "output_cap": self.stage_output_caps[stage],
                    "actual_tokens": self._stage_actual[stage],
                    "model_calls": self._stage_calls[stage],
                }
                for stage in self.stage_output_caps
            },
            "outstanding_reservations": len(self._reservations),
        }


__all__ = [
    "DEFAULT_TONGAGENT_STAGE_OUTPUT_CAPS",
    "ModelStage",
    "PartitionDenial",
    "PartitionReservation",
    "StageTokenController",
]
