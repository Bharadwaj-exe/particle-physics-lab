"""Unit conventions used everywhere in particlelab.

Internal units: energy, momentum and mass in GeV (c = 1); length in metres;
magnetic field in tesla. Convert at the boundary (input files, GUI fields)
by multiplying by the constants below, e.g. ``20 * um`` or ``92.6 * keV``.
"""
from typing import Final

GeV: Final = 1.0
MeV: Final = 1e-3
keV: Final = 1e-6

m: Final = 1.0
cm: Final = 1e-2
mm: Final = 1e-3
um: Final = 1e-6

T: Final = 1.0

# pT [GeV] = C_TRACK * B [T] * R [m] for a unit-charge particle (c / 1e9).
C_TRACK: Final = 0.299792458

# Highland multiple-scattering constant, PDG "Passage of particles through matter".
HIGHLAND_E0: Final = 13.6 * MeV
