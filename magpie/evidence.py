"""Evidence selection and scoring, extracted from :class:`ResearchService`.

Encapsulates the regex-driven chunk scoring and procedural-answer quality
checks that decide which passage of a fetched source becomes an
:class:`~magpie.models.EvidenceItem`. Keeping this logic out of the service
makes it independently testable and keeps the service focused on run
orchestration.

The tokenization pattern is shared with :mod:`magpie.providers.fake`, which
duplicates it deliberately to avoid a cross-module import from a provider
back into the service package.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .models import EvidenceItem, RunBudget

if TYPE_CHECKING:
    from .config import Settings
    from .storage import SQLiteStorage

# Tokenize for evidence overlap scoring: word runs for space-delimited scripts
# (Latin, Cyrillic, …) plus individual characters for CJK scripts that don't
# use word boundaries, so Japanese/Chinese/Korean queries score correctly.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]", re.IGNORECASE)
PROCEDURAL_SIGNALS = ("how do i ", "how to ", "steps to ", "guide to ")
ACTIONABLE_SECTION_SIGNALS = (
    "instruction", "method", "step", "directions", "preparation", "procedure",
    "how to", "process",
)
IMPERATIVE_SIGNALS = (
    "add ", "combine ", "connect ", "cover ", "create ", "enter ", "install ", "mix ",
    "place ", "press ", "remove ", "run ", "set ", "turn ", "type ", "wait ",
)
# Matches any measurement a procedural answer might cite, not only cooking ones.
_MEASUREMENT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:"
    r"g|kg|mg|ml|l|cup|cups|tsp|tbsp|oz|lb|"
    r"minutes?|mins?|seconds?|secs?|hours?|hrs?|days?|"
    r"°f|°c|"
    r"px|pt|em|rem|"
    r"mb|gb|tb|kb"
    r")\b"
)


class EvidenceSelector:
    """Selects and scores evidence extracts from fetched source text.

    The selector owns the regex scoring heuristics (term overlap, procedural
    signals, link-density penalties) and the chunking strategy that turns a
    raw source document into a bounded excerpt. It writes selected items via
    the storage layer's ``add_evidence_item`` so it stays a pure decision
    component over text rather than a persistence owner.
    """

    __slots__ = ("_storage", "_settings")

    def __init__(self, storage: "SQLiteStorage", settings: "Settings") -> None:
        self._storage = storage
        self._settings = settings

    def select_evidence(
        self,
        run_id: str,
        source_id: str,
        text: str,
        question: str,
        remaining_questions: list[str],
        budget: RunBudget,
        current_evidence: list[EvidenceItem],
        note: str = "Selected relevant extract",
    ) -> EvidenceItem | None:
        if budget.evidence_remaining <= 0:
            return None
        max_chars = self._settings.max_evidence_characters_per_item
        chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n|(?<=[.!?])\s+", text) if chunk.strip()]
        terms = set(_TOKEN_PATTERN.findall(" ".join([question, *remaining_questions]).lower()))
        procedural = self.is_procedural(question)
        scored: list[tuple[int, int, str]] = []
        for index, chunk in enumerate(chunks):
            lowered = chunk.lower()
            tokens = set(_TOKEN_PATTERN.findall(lowered))
            score = len(terms & tokens) * 3
            if procedural:
                score += sum(3 for signal in ACTIONABLE_SECTION_SIGNALS if signal in lowered)
                score += sum(2 for signal in IMPERATIVE_SIGNALS if signal in lowered)
                score += min(4, len(_MEASUREMENT_PATTERN.findall(lowered)))
            link_count = chunk.count("](")
            if link_count >= 4:
                score -= 20
            if len(chunk) < 40:
                score -= 2
            scored.append((score, index, chunk))
        useful = [item for item in scored if item[0] > 0]
        candidates = useful or scored
        selected = sorted(sorted(candidates, key=lambda item: item[0], reverse=True)[:12], key=lambda item: item[1])
        excerpt = "\n\n".join(chunk for _score, _index, chunk in selected)[:max_chars].strip()
        if not excerpt:
            return None
        source_limit = min(max_chars, self._settings.max_synthesis_input_characters)
        budget.evidence_remaining -= 1
        return self._storage.add_evidence_item(run_id, source_id, excerpt[:source_limit], note)

    def is_procedural(self, question: str) -> bool:
        lowered = question.lower().strip()
        return any(signal in lowered for signal in PROCEDURAL_SIGNALS)

    def answer_quality_issue(self, question: str, answer: str) -> str | None:
        lowered = answer.lower().strip()
        if not lowered:
            return f"Find a source that directly answers: {question}"
        if not self.is_procedural(question):
            return None
        step_count = len(re.findall(r"(?m)^\s*(?:\d+[.)]|[-*]\s+)", answer))
        imperative_count = sum(1 for signal in IMPERATIVE_SIGNALS if signal in lowered)
        if step_count < 3 and imperative_count < 3:
            return "Find enough evidence to provide a concrete, ordered set of instructions."
        return None
