"""Thread-safe monotonic execution budgets shared by every baseline."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import TypeVar

from pydantic import Field

from .config import BudgetLimits, FrozenStrictModel


Clock = Callable[[], float]
T = TypeVar("T")


class BudgetSnapshot(FrozenStrictModel):
    """Serializable point-in-time view of an execution budget."""

    search_calls: int = Field(ge=0)
    fetch_calls: int = Field(ge=0)
    total_tool_calls: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    # Provider-reported actual usage only.
    total_tokens: int = Field(ge=0)
    # Conservative charge for successful calls whose usage is unavailable.
    estimated_token_charges: int = Field(ge=0)
    accounted_tokens: int = Field(ge=0)
    reserved_tokens: int = Field(ge=0)
    outstanding_model_reservations: int = Field(ge=0)
    token_budget_exhausted: bool
    elapsed_seconds: float = Field(ge=0)
    remaining_wall_time_seconds: float = Field(ge=0)
    deadline_exceeded: bool
    remaining_search_calls: int = Field(ge=0)
    remaining_fetch_calls: int = Field(ge=0)
    remaining_total_tool_calls: int = Field(ge=0)
    remaining_model_calls: int = Field(ge=0)
    remaining_total_tokens: int = Field(ge=0)


class BudgetExceeded(RuntimeError):
    """Raised by ``require_*`` helpers when a shared limit denies work."""

    def __init__(self, resource: str) -> None:
        self.resource = resource
        super().__init__(f"evaluation budget exhausted: {resource}")


@dataclass(frozen=True)
class ModelCallReservation:
    """Opaque handle for one pre-provider model/token reservation."""

    reservation_id: int
    reserved_tokens: int


@dataclass(frozen=True)
class ModelCallSettlement:
    """Outcome of replacing a reservation with observed provider usage."""

    reserved_tokens: int
    actual_tokens: int | None
    estimated_tokens_charged: int
    refunded_tokens: int
    reservation_overrun_tokens: int
    token_budget_exceeded: bool


class ExecutionBudget:
    """Atomically enforce global model, tool, token, and deadline limits.

    Tool and call counters are monotonic. Token capacity is reserved before a
    provider call, then atomically settled against actual usage. Settlement
    refunds unused capacity but actual usage is never dropped, even when it
    exceeds the reservation or hard ceiling.
    """

    def __init__(
        self,
        limits: BudgetLimits,
        *,
        clock: Clock = time.monotonic,
    ) -> None:
        self._limits = limits
        self._clock = clock
        self._lock = RLock()
        started = float(clock())
        self._started_at = started
        self._last_now = started
        self._search_calls = 0
        self._fetch_calls = 0
        self._total_tool_calls = 0
        self._model_calls = 0
        self._total_tokens = 0
        self._estimated_token_charges = 0
        self._reserved_tokens = 0
        self._next_reservation_id = 1
        self._model_reservations: dict[int, int] = {}
        self._token_budget_exhausted = limits.max_total_tokens == 0

    @property
    def limits(self) -> BudgetLimits:
        """Return the immutable common limits."""
        return self._limits

    def try_reserve_tool(self, tool_name: str) -> bool:
        """Reserve one total tool call and its search/fetch sub-budget."""
        kind = _tool_kind(tool_name)
        with self._lock:
            if self._deadline_exceeded_locked():
                return False
            if self._total_tool_calls >= self._limits.max_total_tool_calls:
                return False
            if kind == "search" and (
                self._search_calls >= self._limits.max_search_calls
            ):
                return False
            if kind == "fetch" and self._fetch_calls >= self._limits.max_fetch_calls:
                return False

            self._total_tool_calls += 1
            if kind == "search":
                self._search_calls += 1
            elif kind == "fetch":
                self._fetch_calls += 1
            return True

    def require_tool(self, tool_name: str) -> None:
        """Reserve a tool call or raise a typed budget exception."""
        if not self.try_reserve_tool(tool_name):
            raise BudgetExceeded(_tool_kind(tool_name))

    def try_reserve_model_call(
        self,
        *,
        token_reservation: int,
    ) -> ModelCallReservation | None:
        """Reserve one model call and positive token allowance before provider I/O."""

        _validate_positive(token_reservation, name="token_reservation")
        with self._lock:
            if self._deadline_exceeded_locked():
                return None
            if self._model_calls >= self._limits.max_model_calls:
                return None
            available = (
                self._limits.max_total_tokens
                - self._total_tokens
                - self._estimated_token_charges
                - self._reserved_tokens
            )
            if available <= 0 or token_reservation > available:
                self._token_budget_exhausted = True
                return None
            reservation = ModelCallReservation(
                reservation_id=self._next_reservation_id,
                reserved_tokens=token_reservation,
            )
            self._next_reservation_id += 1
            self._model_calls += 1
            self._reserved_tokens += token_reservation
            self._model_reservations[reservation.reservation_id] = token_reservation
            return reservation

    def require_model_call(
        self,
        *,
        token_reservation: int,
    ) -> ModelCallReservation:
        """Reserve a model call or raise a typed budget exception."""

        reservation = self.try_reserve_model_call(token_reservation=token_reservation)
        if reservation is None:
            raise BudgetExceeded("model_or_token")
        return reservation

    def settle_model_call(
        self,
        reservation: ModelCallReservation,
        *,
        actual_tokens: int | None,
        charge_reservation_if_unknown: bool = False,
    ) -> ModelCallSettlement:
        """Replace one outstanding reservation with actual provider usage."""

        if actual_tokens is not None:
            _validate_nonnegative(actual_tokens, name="actual_tokens")
        with self._lock:
            reserved = self._model_reservations.pop(
                reservation.reservation_id,
                None,
            )
            if reserved is None or reserved != reservation.reserved_tokens:
                raise ValueError("unknown or already settled model reservation")
            self._reserved_tokens -= reserved
            if actual_tokens is None:
                estimated_charge = reserved if charge_reservation_if_unknown else 0
                self._estimated_token_charges += estimated_charge
                accounted = self._total_tokens + self._estimated_token_charges
                if accounted >= self._limits.max_total_tokens:
                    self._token_budget_exhausted = True
                return ModelCallSettlement(
                    reserved_tokens=reserved,
                    actual_tokens=None,
                    estimated_tokens_charged=estimated_charge,
                    refunded_tokens=reserved - estimated_charge,
                    reservation_overrun_tokens=0,
                    token_budget_exceeded=False,
                )

            self._total_tokens += actual_tokens
            refunded = max(0, reserved - actual_tokens)
            overrun = max(0, actual_tokens - reserved)
            exceeded = (
                self._total_tokens
                + self._estimated_token_charges
                + self._reserved_tokens
                > self._limits.max_total_tokens
            )
            if (
                self._total_tokens
                + self._estimated_token_charges
                + self._reserved_tokens
                >= self._limits.max_total_tokens
            ):
                self._token_budget_exhausted = True
            return ModelCallSettlement(
                reserved_tokens=reserved,
                actual_tokens=actual_tokens,
                estimated_tokens_charged=0,
                refunded_tokens=refunded,
                reservation_overrun_tokens=overrun,
                token_budget_exceeded=exceeded,
            )

    def cancel_model_call(
        self,
        reservation: ModelCallReservation,
    ) -> ModelCallSettlement:
        """Release capacity after a provider exception with no known usage."""

        return self.settle_model_call(
            reservation,
            actual_tokens=None,
            charge_reservation_if_unknown=False,
        )

    def try_reserve_tokens(self, tokens: int) -> bool:
        """Reserve known token consumption without reserving a model call."""
        _validate_nonnegative(tokens, name="tokens")
        with self._lock:
            if self._deadline_exceeded_locked():
                return False
            if (
                self._total_tokens
                + self._estimated_token_charges
                + self._reserved_tokens
                + tokens
                > self._limits.max_total_tokens
            ):
                self._token_budget_exhausted = True
                return False
            self._total_tokens += tokens
            return True

    def require_tokens(self, tokens: int) -> None:
        """Reserve token consumption or raise a typed budget exception."""
        if not self.try_reserve_tokens(tokens):
            raise BudgetExceeded("tokens")

    def deadline_exceeded(self) -> bool:
        """Return whether the monotonic wall-time deadline has been reached."""
        with self._lock:
            return self._deadline_exceeded_locked()

    def remaining_wall_time_seconds(self) -> float:
        """Return clamped time remaining before the shared deadline."""
        with self._lock:
            elapsed = self._elapsed_locked()
            return max(0.0, self._limits.wall_time_seconds - elapsed)

    def limit_search_results(self, results: Sequence[T]) -> list[T]:
        """Apply the shared per-search result cap without mutating input."""
        return list(results[: self._limits.max_results_per_search])

    def limit_page_content(self, content: str) -> tuple[str, bool]:
        """Apply the shared page-character cap and report truncation."""
        limit = self._limits.max_page_chars
        if len(content) <= limit:
            return content, False
        return content[:limit], True

    def snapshot(self) -> BudgetSnapshot:
        """Return an atomic snapshot suitable for trace and result artifacts."""
        with self._lock:
            elapsed = self._elapsed_locked()
            return BudgetSnapshot(
                search_calls=self._search_calls,
                fetch_calls=self._fetch_calls,
                total_tool_calls=self._total_tool_calls,
                model_calls=self._model_calls,
                total_tokens=self._total_tokens,
                estimated_token_charges=self._estimated_token_charges,
                accounted_tokens=(self._total_tokens + self._estimated_token_charges),
                reserved_tokens=self._reserved_tokens,
                outstanding_model_reservations=len(self._model_reservations),
                token_budget_exhausted=self._token_budget_exhausted,
                elapsed_seconds=elapsed,
                remaining_wall_time_seconds=max(
                    0.0,
                    self._limits.wall_time_seconds - elapsed,
                ),
                deadline_exceeded=elapsed >= self._limits.wall_time_seconds,
                remaining_search_calls=max(
                    0,
                    self._limits.max_search_calls - self._search_calls,
                ),
                remaining_fetch_calls=max(
                    0,
                    self._limits.max_fetch_calls - self._fetch_calls,
                ),
                remaining_total_tool_calls=max(
                    0,
                    self._limits.max_total_tool_calls - self._total_tool_calls,
                ),
                remaining_model_calls=max(
                    0,
                    self._limits.max_model_calls - self._model_calls,
                ),
                remaining_total_tokens=max(
                    0,
                    self._limits.max_total_tokens
                    - self._total_tokens
                    - self._estimated_token_charges
                    - self._reserved_tokens,
                ),
            )

    def _deadline_exceeded_locked(self) -> bool:
        return self._elapsed_locked() >= self._limits.wall_time_seconds

    def _elapsed_locked(self) -> float:
        observed = float(self._clock())
        self._last_now = max(self._last_now, observed)
        return max(0.0, self._last_now - self._started_at)


def _tool_kind(tool_name: str) -> str:
    """Normalize known tool aliases for common search/fetch accounting."""
    normalized = tool_name.strip().casefold()
    if normalized in {"search", "web_search"}:
        return "search"
    if normalized in {"fetch", "fetch_url", "open_page", "open_url"}:
        return "fetch"
    return "other_tool"


def _validate_nonnegative(value: int, *, name: str) -> None:
    """Reject bools and negative/non-integer reservations."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{name} must be a non-negative integer"
        raise ValueError(msg)


def _validate_positive(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
