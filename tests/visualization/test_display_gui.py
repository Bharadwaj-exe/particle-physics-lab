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
        f_mid = 0.4 * ev.t_tracker / w.clock.t_max      # inside the tracker phase (outer detector may be on)
        w.clock.seek(f_mid); app.processEvents()
        out["mid_state"] = [it["reco"].isVisible(), it["flight"].isVisible(), it["head"].isVisible(),
                            it["arrow"].isVisible()]
        t0 = ev.tracks[0]
        n_expected = int(np.sum(t0.hit_t <= f_mid * w.clock.t_max))
        out["hits_mid"] = [len(it["hits"].getData()[0]), n_expected]
        hx, hy = it["head"].getData()
        path_pt = d.path_until(t0.true_path, t0.true_t, f_mid * w.clock.t_max)[-1]
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
            out["geometry_is_cms"] = w.geometry == d.detector_geometry(d.cms_inspired(),
                                                                      outer=w.current_event.has_outer)

        # theme and image export
        w.dark_action.setChecked(True); app.processEvents()
        out["dark_bg"] = w.event2d.xy.backgroundBrush().color().name()
        out["theme_saved"] = settings.value("theme")
        p2 = {str(tmp_path / "e2d.png")!r}
        w.tabs.setCurrentWidget(w.event2d); app.processEvents()
        w.save_image(p2)
        img = QtGui.QImage(p2)
        out["png"] = [img.width() > 300, img.height() > 200]
        cols = [img.pixelColor(int(img.width() * (i + 0.5) / 20), int(img.height() * (j + 0.5) / 20))
                for i in range(20) for j in range(20)]
        out["png_dark"] = sorted((c.red() + c.green() + c.blue()) / 3 for c in cols)[200] < 80   # median
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


def _run(script, platform="offscreen", scale=None):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if scale:
        env["QT_SCALE_FACTOR"] = str(scale)
    if platform:
        env["QT_QPA_PLATFORM"] = platform
    else:
        env.pop("QT_QPA_PLATFORM", None)
    p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600, env=env,
                       encoding="utf-8")
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT")]
    assert line, p.stdout[-3000:] + p.stderr[-3000:]
    return json.loads(line[0][len("RESULT"):])


@pytest.mark.skipif(sys.platform != "win32", reason="measured with the native Windows fonts")
@pytest.mark.parametrize("scale", [1.0, 1.5])
def test_window_fits_a_small_laptop_screen(tmp_path, scale):
    """Every tab must fit a 1366x768 laptop (1366x720 usable) without cutting anything off.
    Uses the native platform (the offscreen one has much wider fallback fonts); the window is not shown."""
    script = textwrap.dedent(f"""
        import sys, time, json
        sys.path.insert(0, {str(ROOT)!r})
        from PySide6 import QtWidgets, QtCore
        app = QtWidgets.QApplication(["t"])
        from particlelab.app import desktop as d
        s = QtCore.QSettings({str(tmp_path / "s.ini")!r}, QtCore.QSettings.IniFormat)
        s.setValue("animate_new_events", False)
        w = d.MainWindow(s); w.ensurePolished()
        t0 = time.perf_counter()
        while w.current_event is None and time.perf_counter() - t0 < 90:
            app.processEvents(); time.sleep(0.01)
        for wd in w.findChildren(QtWidgets.QWidget):
            wd.ensurePolished()
        out = {{"window_min": w.minimumSizeHint().toTuple(), "tabs": {{}}}}
        for i in range(w.tabs.count()):
            out["tabs"][w.tabs.tabText(i)] = w.tabs.widget(i).minimumSizeHint().toTuple()
        out["api"] = [w.tabs.indexOf(w.event2d) == 1, w.tabs.widget(2) is w.event3d]
        w.tabs.setCurrentWidget(w.performance); app.processEvents()
        out["api"].append(w.tabs.currentWidget() is w.performance)
        w.close()
        print("RESULT" + json.dumps(out))
    """)
    r = _run(script, platform=None, scale=scale)
    w_min, h_min = r["window_min"]
    # a 1366x768 laptop: 1366 px wide at 100 % display scaling, 911 px at 150 %
    width = 1366 / scale
    assert w_min <= width - 40 and h_min <= 640 / scale, r
    for name, (tw, th) in r["tabs"].items():
        assert tw <= width - 380, (name, tw, th)       # beside the settings panel, no sideways scrolling
        if scale == 1.0:
            assert th <= 560, (name, tw, th)
    assert all(r["api"])


def test_plots_cannot_be_distorted_and_layout_is_remembered(tmp_path):
    """Histogram views zoom in x only, stay inside their limits and reset on double-click;
    window size and pane sizes survive a restart; minimise and restore work."""
    script = textwrap.dedent(f"""
        import sys, time, json
        sys.path.insert(0, {str(ROOT)!r})
        from PySide6 import QtWidgets, QtCore
        app = QtWidgets.QApplication(["t"])
        from particlelab.app import desktop as d
        ini = {str(tmp_path / "s.ini")!r}
        s = QtCore.QSettings(ini, QtCore.QSettings.IniFormat); s.setValue("animate_new_events", False)
        w = d.MainWindow(s); w.show()
        def wait(cond, t=120):
            t0 = time.perf_counter()
            while not cond() and time.perf_counter() - t0 < t:
                app.processEvents(); time.sleep(0.01)
        wait(lambda: w.current_event is not None)
        w.setup.n.setValue(20000)
        w.mass.run(); wait(lambda: w.mass.result is not None)
        vb = w.mass.plot.getViewBox()
        fit = w.mass.result.get("data_fit") or w.mass.result["sim_fit"]
        out = {{"mouse": list(vb.state["mouseEnabled"]), "window": list(fit.window)}}
        # try to squash the plot as in the bug report: zoom x far out and y far in
        vb.scaleBy((20.0, 0.05))
        app.processEvents()
        (x0, x1), (y0, y1) = vb.viewRange()
        out["after_zoom"] = [x0, x1, y0]
        w.mass.plot.reset_view(); app.processEvents()
        out["after_reset_x"] = vb.viewRange()[0]
        out["y_auto"] = bool(vb.state["autoRange"][1])
        # 2D view stays inside the detector
        xy = w.event2d.view.xy.getViewBox()
        xy.scaleBy((50.0, 50.0)); app.processEvents()
        out["xy_span"] = xy.viewRange()[0][1] - xy.viewRange()[0][0]
        # layout memory
        w.panel_action.setChecked(False)            # hide the settings panel (Ctrl+B)
        w.showNormal(); w.resize(700, 480); w.move(20, 20)
        ex = w.tabs.widget(w.tabs.count() - 1)        # physics explorer: its content does not change
        w.tabs.setCurrentWidget(ex)
        for _ in range(10): app.processEvents()
        tot = sum(ex.split.sizes())
        ex.split.setSizes([tot // 3, tot - tot // 3]); app.processEvents()
        out["min_restore"] = []
        w.showMinimized(); app.processEvents(); out["min_restore"].append(w.isMinimized())
        w.showNormal(); app.processEvents(); out["min_restore"].append(not w.isMinimized())
        sizes = ex.split.sizes(); size = w.size().toTuple()
        w.close(); app.processEvents()
        w2 = d.MainWindow(QtCore.QSettings(ini, QtCore.QSettings.IniFormat)); w2.show()
        for _ in range(20): app.processEvents(); time.sleep(0.01)
        ex2 = w2.tabs.widget(w2.tabs.count() - 1)
        w2.tabs.setCurrentWidget(ex2)
        for _ in range(10): app.processEvents()
        out["restored"] = [w2.size().toTuple() == size, abs(ex2.split.sizes()[0] - sizes[0]) <= 3,
                           not w2.panel_action.isChecked() and w2.setup_area.isHidden()]
        out["debug"] = [w2.size().toTuple(), size, ex2.split.sizes(), sizes]
        w2.reset_layout(); app.processEvents()
        out["reset_forgets"] = w2.settings.value("window/geometry") is None
        w2.close()
        print("RESULT" + json.dumps(out))
    """)
    r = _run(script)
    assert r["mouse"] == [True, False]                       # y is never mouse-controlled on a histogram
    lo, hi = r["window"]
    pad = 0.011 * (hi - lo)
    x0, x1, y0 = r["after_zoom"]
    assert lo - pad <= x0 and x1 <= hi + pad and y0 >= 0     # cannot leave the fit window or go below zero
    assert abs(r["after_reset_x"][0] - (lo - 0.01 * (hi - lo))) < 1e-6 and r["y_auto"]
    assert r["xy_span"] < 50.0                                # bounded (equal-scale view of a 15 m detector)
    assert r["min_restore"] == [True, True]
    assert r["restored"] == [True, True, True] and r["reset_forgets"], r["debug"]
