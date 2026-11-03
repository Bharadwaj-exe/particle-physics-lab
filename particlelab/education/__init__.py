"""Learner-facing logic: guided reconstruction exercises and the process gate.

Pure Python/NumPy, no GUI: the desktop app shows it, the tests check it.
"""
from .reconstruction import Exercise, Feedback, MeasuredTrack, from_display_event, from_momenta
from .process_gate import GateResult, gate

__all__ = ["Exercise", "Feedback", "MeasuredTrack", "from_display_event", "from_momenta", "GateResult", "gate"]
