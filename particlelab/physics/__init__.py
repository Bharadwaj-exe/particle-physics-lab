"""Event generation: production, decay and relativistic kinematics."""
from .kinematics import boost, inv_mass, minkowski_mass, pt_eta_phi, cos_theta_cs, M_MU
from .production import ProductionModel, default_model, sample_mass, sample_parents
from .decay import DecayBatch, generate, decay_rest, sample_cos_theta, frame_axes
