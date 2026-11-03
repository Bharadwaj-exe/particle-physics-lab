"""Detector model: geometry, hit simulation and track reconstruction."""
from .geometry import DetectorConfig, BarrelLayer, EndcapDisk, cms_inspired, uniform_barrel
from .simulation import Hits, simulate_hits, helix_at
from .tracking import TrackFit, fit_tracks, measure, gluckstern, ms_resolution
