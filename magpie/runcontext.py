"""Per-run mutable state for the research loop, extracted from :class:`ResearchService`.

Holds the budget and the accumulated lists/draft the loop mutates across
rounds: warnings, limitations, seen URLs, remaining questions, the last
synthesis draft, and the gathered evidence. Lifting these out of bare locals
in :meth:`ResearchService.research` into one named object makes the run's
mutable state explicit and shrinks the argument lists the loop's helpers
take (they receive the context rather than four or five loose lists).

The loop body itself stays in ``research``; this is a state holder, not a
loop driver. ``PlanningContext`` (built from the run context each round) is
a separate immutable snapshot handed to the resolver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .models import EvidenceItem, RunBudget, SynthesisDraft

if TYPE_CHECKING:
    pass


@dataclass(slots=True)
class RunContext:
    """Mutable state carried across rounds of a single research run.

    Fields are mutated in place by :meth:`ResearchService.research` and its
    helpers; callers should treat an instance as owned by one run.
    """

    budget: RunBudget
    evidence: list[EvidenceItem] = field(default_factory=list)
    seen_urls: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    remaining_questions: list[str] = field(default_factory=list)
    last_draft: SynthesisDraft | None = None
