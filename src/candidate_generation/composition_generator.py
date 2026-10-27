"""Dynamic composition proposal generation for the active-learning loop."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class CompositionCandidate:
    elem_a: str
    elem_b: str
    elem_x: str
    mixed_a: str | None = None
    mixed_b: str | None = None
    fraction: float = 0.5

    @property
    def formula(self) -> str:
        if self.mixed_a:
            return f"{self.elem_a}{self.fraction:g}{self.mixed_a}{1 - self.fraction:g}{self.elem_b}{self.elem_x}3"
        if self.mixed_b:
            return f"{self.elem_a}{self.elem_b}{self.fraction:g}{self.mixed_b}{1 - self.fraction:g}{self.elem_x}3"
        return f"{self.elem_a}{self.elem_b}{self.elem_x}3"


class CompositionGenerator:
    """Generate novel candidates from supported chemistry and memory."""

    def __init__(
        self,
        element_properties: dict[str, dict[str, float]],
        tolerance_fn: Callable[[str, str, str], tuple[float, float]],
        known_formulas: Iterable[str] = (),
    ) -> None:
        self.element_properties = element_properties
        self.tolerance_fn = tolerance_fn
        self.known_formulas = set(known_formulas)

    def propose(
        self,
        preferred_a: Sequence[str],
        preferred_b: Sequence[str],
        preferred_x: Sequence[str],
    ) -> list[CompositionCandidate]:
        candidates: list[CompositionCandidate] = []
        for elem_a, elem_b, elem_x in product(preferred_a, preferred_b, preferred_x):
            if elem_a not in self.element_properties or elem_b not in self.element_properties:
                continue
            tolerance, _ = self.tolerance_fn(elem_a, elem_b, elem_x)
            formula = f"{elem_a}{elem_b}{elem_x}3"
            if 0.81 <= tolerance <= 1.05 and formula not in self.known_formulas:
                candidates.append(CompositionCandidate(elem_a, elem_b, elem_x))
            for mixed_a in preferred_a:
                if mixed_a == elem_a:
                    continue
                mixed_tolerance, _ = self.tolerance_fn(mixed_a, elem_b, elem_x)
                mixed_formula = f"{elem_a}0.5{mixed_a}0.5{elem_b}{elem_x}3"
                if 0.81 <= mixed_tolerance <= 1.05 and mixed_formula not in self.known_formulas:
                    candidates.append(CompositionCandidate(
                        elem_a, elem_b, elem_x, mixed_a=mixed_a
                    ))
            for mixed_b in preferred_b:
                if mixed_b == elem_b:
                    continue
                mixed_tolerance, _ = self.tolerance_fn(elem_a, mixed_b, elem_x)
                mixed_formula = f"{elem_a}{elem_b}0.5{mixed_b}0.5{elem_x}3"
                if 0.81 <= mixed_tolerance <= 1.05 and mixed_formula not in self.known_formulas:
                    candidates.append(CompositionCandidate(
                        elem_a, elem_b, elem_x, mixed_b=mixed_b
                    ))
        return candidates
