"""The upgraded event displays in the real main window (fresh offscreen Qt interpreter):
animation clock, 2D animation states, CMS data events, comparison, theme and image export."""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
ROOT = Path(__file__).resolve().parents[2]


def test_animation_clock_runs_and_stops_at_the_end():
    script = textwrap.dedent(f"""
        import sys, time, json
        sys.path.insert(0, {str(ROOT)!r})
        from PySide6 import QtWidgets
        app = QtWidgets.QApplication(["t"])
        from particlelab.app.animation import AnimationClock, AnimationBar
        import particlelab.app.animation as A
        A.PLAY_SECONDS = 0.4
        c = AnimationClock(); bar = AnimationBar(c); bar2 = AnimationBar(c)
        frames = []
        c.frame.connect(frames.append)
        c.set_span(8.0)
        c.restart()
        t0 = time.perf_counter()
        while c.playing and time.perf_counter() - t0 < 5:
            app.processEvents(); time.sleep(0.005)
        out = dict(n=len(frames), first=frames[0], last=frames[-1], playing=c.playing,
                   monotonic=all(b >= a for a, b in zip(frames, frames[1:])), label=bar.time_label.text(),
                   slider=bar2.slider.value())
        bar.slider.setValue(250)
        out["seek"] = c.fraction; out["t"] = c.time_ns()
        bar.auto.setChecked(True); out["auto_synced"] = bar2.auto.isChecked() and c.autoplay
        c.new_event(4.0); out["autoplay_restarts"] = c.playing and c.fraction < 0.5
        c.set_autoplay(False); c.new_event(4.0); out["manual_complete"] = (not c.playing) and c.fraction == 1.0
        print("RESULT" + json.dumps(out))
    """)
    r = _run(script)
    assert r["n"] > 5 and r["first"] == 0.0 and r["last"] == 1.0 and not r["playing"] and r["monotonic"]
    assert "8.00 ns" in r["label"] and r["slider"] == 1000
    assert r["seek"] == 0.25 and abs(r["t"] - 2.0) < 1e-9
    assert r["auto_synced"] and r["autoplay_restarts"] and r["manual_complete"]


def test_main_window_display_upgrades(tmp_path):
    script = textwrap.dedent(f"""
        import sys, time, json, os
        sys.path.insert(0, {str(ROOT)!r})
        import numpy as np
        from PySide6 import QtWidgets, QtCore, QtGui
        app = QtWidgets.QApplication(["t"])
        from particlelab.app import desktop as d, theme
        from particlelab.data import cms
        settings = QtCore.QSettings({str(tmp_path / "s.ini")!r}, QtCore.QSettings.IniFormat)
        settings.setValue("animate_new_events", False)
        w = d.MainWindow(settings); w.resize(1300, 850); w.show()
        def wait(cond, t=90):
            t0 = time.perf_counter()
            while not cond() and time.perf_counter() - t0 < t:
                app.processEvents(); time.sleep(0.01)
            return bool(cond())
        out = {{}}
        out["first"] = wait(lambda: w.current_event is not None)
        w.tabs.setCurrentWidget(w.event2d); app.processEvents()
        ev = w.current_event
        out["complete_at_start"] = w.clock.fraction == 1.0 and abs(w.clock.t_max - ev.t_max) < 1e-9
        v = w.event2d.view
        it = v._tracks[0]["items"][0][1]
        out["final_state"] = [it["reco"].isVisible(), it["flight"].isVisible(), it["head"].isVisible()]
        w.clock.seek(0.4); app.processEvents()
        out["mid_state"] = [it["reco"].isVisible(), it["flight"].isVisible(), it["head"].isVisible(),
                            it["arrow"].isVisible()]
        t0 = ev.tracks[0]
        n_expected = int(np.sum(t0.hit_t <= 0.4 * w.clock.t_max))
        out["hits_mid"] = [len(it["hits"].getData()[0]), n_expected]
        hx, hy = it["head"].getData()
        path_pt = d.path_until(t0.true_path, t0.true_t, 0.4 * w.clock.t_max)[-1]
        out["head_at_path"] = float(np.hypot(hx[0] - path_pt[0], hy[0] - path_pt[1]))
        w.clock.seek(1.0); app.processEvents()

        # comparison with a CMS data event
        out["data_available"] = cms.available()
        if cms.available():
            old = w.current_event
            w.event2d.compare.setChecked(True)
            out["compare"] = wait(lambda: w.current_event is not old and w.companion is not None)
            out["companion_is_data"] = w.companion.is_data and not w.current_event.is_data
            out["other_shown"] = not w.event2d.other.isHidden()
            out["span_covers_both"] = w.clock.t_max >= max(w.current_event.t_max, w.companion.t_max) - 1e-9
            w.event2d.compare.setChecked(False); app.processEvents()
            # data as the main source: every listener receives it, the exercise uses it
            old = w.current_event
            w.setup.source.setCurrentIndex(1)
            out["data_main"] = wait(lambda: w.current_event is not old and w.current_event.is_data)
            ex_mass = w.reconstruct.ex.expected()["mass"]
            out["exercise_mass_matches"] = abs(ex_mass - w.current_event.reco_mass) < 1e-9 * ex_mass
            out["caption"] = w.event2d.view.caption.text()
            out["geometry_is_cms"] = w.geometry == d.detector_geometry(d.cms_inspired())

        # theme and image export
        w.dark_action.setChecked(True); app.processEvents()
        out["dark_bg"] = w.event2d.xy.backgroundBrush().color().name()
        out["theme_saved"] = settings.value("theme")
        p2 = {str(tmp_path / "e2d.png")!r}
        w.tabs.setCurrentWidget(w.event2d); app.processEvents()
        w.save_image(p2)
        img = QtGui.QImage(p2)
        out["png"] = [img.width() > 300, img.height() > 200]
        c = img.pixelColor(img.width() // 4, img.height() // 2)
        out["png_dark"] = (c.red() + c.green() + c.blue()) / 3 < 80
        w.set_theme("light"); app.processEvents()
        out["light_bg"] = w.event2d.xy.backgroundBrush().color().name()
        out["menu_unchecked"] = not w.dark_action.isChecked()
        w.tabs.setCurrentWidget(w.performance); app.processEvents()
        p3 = {str(tmp_path / "perf.png")!r}
        w.save_image(p3); out["generic_tab_png"] = os.path.getsize(p3) > 1000
        w.close()
        print("RESULT" + json.dumps(out))
    """)
    r = _run(script)
    assert r["first"] and r["complete_at_start"]
    assert r["final_state"] == [True, False, False]
    assert r["mid_state"] == [False, True, True, False]
    assert r["hits_mid"][0] == r["hits_mid"][1]
    assert r["head_at_path"] < 1e-6
    if r["data_available"]:
        assert r["compare"] and r["companion_is_data"] and r["other_shown"] and r["span_covers_both"]
        assert r["data_main"] and r["exercise_mass_matches"] and r["geometry_is_cms"]
        assert "CMS 2011 data" in r["caption"] and "not CMS hits" in r["caption"]
    assert r["dark_bg"] == "#1e1f22" and r["theme_saved"] == "dark"
    assert r["png"] == [True, True] and r["png_dark"]
    assert r["light_bg"] == "#ffffff" and r["menu_unchecked"] and r["generic_tab_png"]


def _run(script):
    p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600,
                       env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONIOENCODING": "utf-8"},
                       encoding="utf-8")
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT")]
    assert line, p.stdout[-3000:] + p.stderr[-3000:]
    return json.loads(line[0][len("RESULT"):])
