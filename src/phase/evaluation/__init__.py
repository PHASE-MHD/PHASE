"""Held-out-test evaluation for tFNO, DINO, scOT, and PHASE."""

from .metrics import EvaluationRecord, evaluate_records
from .physics import derive_fields
from .reporting import write_evaluation_report

__all__ = [
    "EvaluationRecord",
    "derive_fields",
    "evaluate_records",
    "write_evaluation_report",
]
