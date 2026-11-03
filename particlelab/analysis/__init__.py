"""Simulation pipeline, peak fitting and performance studies."""
from .fitting import FitResult, fit_peak, fit_counts, WINDOWS, DEFAULT_SIGNAL
from .pipeline import SimResult, simulate, CUTFLOW_STEPS
from .studies import pt_resolution, resolution_curve, toy_study, robust_sigma
