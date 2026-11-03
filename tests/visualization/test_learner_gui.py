"""Smoke tests of the learner-facing widgets, in a fresh offscreen Qt interpreter."""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
ROOT = Path(__file__).resolve().parents[2]


def test_gate_reconstruction_tab_and_diagram_note():
    script = textwrap.dedent(f"""
        import sys, json
        sys.path.insert(0, {str(ROOT)!r})
        import numpy as np
        from PySide6 import QtWidgets
        app = QtWidgets.QApplication(["t"])
        from particlelab.app import desktop as d
        from particlelab.detector import cms_inspired
        from particlelab.visualization.event_display import make_event

        class Main:
            pass
        main = Main()
        main.setup = d.SetupPanel()
        main.tabs = QtWidgets.QTabWidget(); main.mass = QtWidgets.QWidget(); main.tabs.addTab(main.mass, "m")
        main.status = ""
        main.set_status = lambda text, busy: setattr(main, "status", text)
        main.new_event = lambda seed=None: None
        out = {{}}
        out["gate_label"] = main.setup.gate_label.text()

        ex = d.ExplorerTab(main)
        main.setup.select_process("J/psi")
        ex.edit.setText("Z -> mu+ e-"); ex.run_check()
        out["forbidden_enabled"] = ex.use.isEnabled()
        out["forbidden_text"] = ex.out.toPlainText()
        ex.edit.setText("Z -> mu+ mu-"); ex.run_check()
        out["allowed_enabled"] = ex.use.isEnabled()
        ex.use.click()
        out["selected"] = main.setup.proc.currentData()

        rt = d.ReconstructTab(main)
        ev = make_event("Z", cms_inspired(), np.random.default_rng(3))
        rt.show_event(ev)
        e = rt.ex.expected()
        for i in (0, 1):
            rt.inputs[("R", i)].setText(f"{{e['R'][i]:.4f}}")
            rt.inputs[("pT", i)].setText(f"{{e['pT'][i]:.4f}}")
            rt.inputs[("E", i)].setText(f"{{e['E'][i]:.4f}}")
        rt.inputs[("mass", None)].setText(f"{{e['mass']:.4f}}")
        rt.check()
        out["fb_right"] = rt.feedback.toPlainText()
        out["add_enabled"] = rt.add_btn.isEnabled()
        rt.add_mass()
        out["n_masses"] = len(rt.my_masses)
        out["mass_matches_event"] = abs(e["mass"] - ev.reco_mass) < 1e-9 * ev.reco_mass
        rt.inputs[("mass", None)].setText(f"{{e['E'][0] + e['E'][1]:.4f}}")
        rt.check()
        out["fb_wrong"] = rt.feedback.toPlainText()
        rt.show_solution()
        out["solution"] = rt.feedback.toPlainText()

        et = d.EventTab(type("M", (), {{"new_event": lambda self, seed=None: None}})())
        out["contrast"] = et.contrast.text()
        print("RESULT" + json.dumps(out))
    """)
    p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300,
                       env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONIOENCODING": "utf-8"},
                       encoding="utf-8")
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT")]
    assert line, p.stdout + p.stderr
    r = json.loads(line[0][len("RESULT"):])
    assert r["gate_label"].startswith("✓")
    assert not r["forbidden_enabled"] and "muon number" in r["forbidden_text"]
    assert r["allowed_enabled"] and r["selected"] == "Z"
    assert "✗" not in r["fb_right"] and r["fb_right"].count("✓") == 7
    assert r["add_enabled"] and r["n_masses"] == 1 and r["mass_matches_event"]
    assert "energies only" in r["fb_wrong"] and "Worked solution" in r["solution"]
    assert "not for paths" in r["contrast"] and "inferred" in r["contrast"]
