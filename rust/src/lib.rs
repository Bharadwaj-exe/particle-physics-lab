//! Compiled kernels for particlelab: hit simulation and track fitting.
//!
//! These are line-by-line ports of `particlelab/detector/simulation.py::simulate_hits`
//! and `particlelab/detector/tracking.py::fit_tracks`. Every random number is drawn by
//! NumPy in Python and passed in, so for the same seed both versions produce the same
//! hits and fits (to floating-point rounding). Each track is independent, so tracks
//! are processed in parallel with rayon. The NumPy code remains the reference.

use numpy::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Bumped whenever a signature or a result changes; checked by the Python side.
const API_VERSION: u32 = 1;
const MAXS: usize = 128; // maximum number of surfaces (sensitive + passive)

// ---------------------------------------------------------------------------------------
// hit simulation

#[derive(Clone, Copy)]
struct Surf {
    disk: bool,
    r: f64,
    hl: f64,
    zd: f64,
    rmin: f64,
    rmax: f64,
    s1: f64,
    s2: f64,
    x0: f64,
    zmin: f64,
    sens: bool,
}

fn read_surfaces(a: ArrayView2<f64>) -> PyResult<Vec<Surf>> {
    if a.ncols() != 11 {
        return Err(PyValueError::new_err("surface table must have 11 columns"));
    }
    if a.nrows() > MAXS {
        return Err(PyValueError::new_err("too many surfaces"));
    }
    Ok(a.rows()
        .into_iter()
        .map(|r| Surf {
            disk: r[0] != 0.0,
            r: r[1],
            hl: r[2],
            zd: r[3],
            rmin: r[4],
            rmax: r[5],
            s1: r[6],
            s2: r[7],
            x0: r[8],
            zmin: r[9],
            sens: r[10] != 0.0,
        })
        .collect())
}

#[derive(Clone, Copy)]
struct Trk {
    ptot: f64,
    r: f64,
    k: f64,
    phi0: f64,
    cot: f64,
    sin_t: f64,
    cos_t: f64,
    z0: f64,
}

fn sign(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn track(px: f64, py: f64, pz: f64, q: f64, z0: f64, cb: f64) -> Trk {
    let pt = px.hypot(py);
    let ptot = (px * px + py * py + pz * pz).sqrt();
    let cot = pz / pt;
    let sin_t = 1.0 / (1.0 + cot * cot).sqrt();
    Trk { ptot, r: pt / cb, k: sign(q), phi0: py.atan2(px), cot, sin_t, cos_t: cot * sin_t, z0 }
}

/// Transverse arc length at which the track crosses the surface, or +inf.
fn crossing(t: &Trk, s: &Surf) -> f64 {
    if !s.disk {
        let ok = s.r < 2.0 * t.r;
        let sj = 2.0 * t.r * (s.r / (2.0 * t.r)).clamp(0.0, 1.0).asin();
        let zj = (t.z0 + sj * t.cot).abs();
        if ok && zj <= s.hl && zj >= s.zmin {
            sj
        } else {
            f64::INFINITY
        }
    } else {
        let zt = if t.cot >= 0.0 { s.zd } else { -s.zd };
        let sj = (zt - t.z0) / t.cot;
        let ok = sj.is_finite() && sj > 0.0 && sj < PI * t.r;
        if !ok {
            return f64::INFINITY;
        }
        let rj = 2.0 * t.r * (sj / (2.0 * t.r)).sin().abs();
        if rj >= s.rmin && rj <= s.rmax {
            sj
        } else {
            f64::INFINITY
        }
    }
}

struct Consts {
    e0: f64,
    m_mu: f64,
    ms: bool,
    smear: bool,
    eff: f64,
}

struct Randoms<'a> {
    th1: ArrayView2<'a, f32>,
    th2: ArrayView2<'a, f32>,
    ga: ArrayView2<'a, f32>,
    gl: ArrayView2<'a, f32>,
    u: ArrayView2<'a, f32>,
}

#[allow(clippy::too_many_arguments)]
fn hits_row(
    i: usize,
    t: &Trk,
    surfs: &[Surf],
    c: &Consts,
    rnd: &Randoms,
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
    os: &mut [f64],
    ocr: &mut [bool],
    ode: &mut [bool],
    os1: &mut [f64],
    os2: &mut [f64],
    odk: &mut [bool],
    osf: &mut [i64],
    too_many: &mut bool,
) {
    let ns = surfs.len();
    let sm = ox.len();
    let mut sv = [f64::INFINITY; MAXS];
    let mut idx = [0usize; MAXS];
    for j in 0..ns {
        sv[j] = crossing(t, &surfs[j]);
        idx[j] = j;
    }
    idx[..ns].sort_unstable_by(|&a, &b| sv[a].partial_cmp(&sv[b]).unwrap_or(std::cmp::Ordering::Equal));

    let n_crossed = (0..ns).filter(|&j| sv[j].is_finite()).count();
    if n_crossed > sm {
        *too_many = true;
        return;
    }
    let (r, k) = (t.r, t.k);
    let mut crossed = [false; MAXS];
    let mut disk = [false; MAXS];
    let mut sf = [0.0f64; MAXS];
    // sine and cosine of the track direction at each hit, reused below
    let (mut sp, mut cp) = ([0.0f64; MAXS], [0.0f64; MAXS]);
    let (mut x, mut y, mut z) = ([0.0f64; MAXS], [0.0f64; MAXS], [0.0f64; MAXS]);
    let xc = k * r * t.phi0.sin();
    let yc = -k * r * t.phi0.cos();
    for col in 0..sm {
        let j = idx[col];
        let s = sv[j];
        crossed[col] = s.is_finite();
        disk[col] = surfs[j].disk;
        sf[col] = if crossed[col] { s } else { 0.0 };
        let ph = t.phi0 - k * sf[col] / r;
        let (sph, cph) = ph.sin_cos();
        sp[col] = sph;
        cp[col] = cph;
        x[col] = xc - k * r * sph;
        y[col] = yc + k * r * cph;
        z[col] = t.z0 + sf[col] * t.cot;
        os[col] = s;
        os1[col] = surfs[j].s1;
        os2[col] = surfs[j].s2;
        odk[col] = disk[col];
        osf[col] = j as i64;
    }

    if c.ms {
        let mut xeff = [0.0f64; MAXS];
        let mut xtot = 0.0;
        for col in 0..sm {
            // cos(phi_t - atan2(y, x)) without trigonometric calls
            let rho = (x[col] * x[col] + y[col] * y[col]).sqrt();
            let cosd = if rho > 0.0 { (cp[col] * x[col] + sp[col] * y[col]) / rho } else { cp[col] };
            let rhat_dot_t = cosd * t.sin_t;
            let inc = if disk[col] { t.cos_t.abs() } else { rhat_dot_t.abs() };
            xeff[col] = if crossed[col] { surfs[idx[col]].x0 / inc.max(1e-3) } else { 0.0 };
            xtot += xeff[col];
        }
        let beta = t.ptot / (t.ptot * t.ptot + c.m_mu * c.m_mu).sqrt();
        let logf = if xtot > 0.0 { 1.0 + 0.038 * (xtot / (beta * beta)).ln() } else { 0.0 };
        let pref = c.e0 / (beta * t.ptot);
        let (mut cs1, mut cs1l, mut cs2, mut cs2l) = (0.0, 0.0, 0.0, 0.0);
        for col in 0..sm {
            let theta0 = pref * xeff[col].sqrt() * logf.max(0.0);
            let l = sf[col] / t.sin_t;
            let th1 = rnd.th1[[i, col]] as f64 * theta0;
            let th2 = rnd.th2[[i, col]] as f64 * theta0;
            cs1 += th1;
            cs1l += th1 * l;
            cs2 += th2;
            cs2l += th2 * l;
            let u = l * (cs1 - th1) - (cs1l - th1 * l);
            let v = l * (cs2 - th2) - (cs2l - th2 * l);
            let (sp, cp) = (sp[col], cp[col]);
            x[col] -= u * sp;
            y[col] += u * cp;
            if !disk[col] {
                z[col] += v / t.sin_t;
            }
            let dr = if disk[col] { -v / (if t.cos_t.abs() > 1e-6 { t.cos_t } else { 1e-6 }) } else { 0.0 };
            x[col] += dr * cp;
            y[col] += dr * sp;
            let r_nom = if disk[col] { 1.0 } else { 2.0 * r * (sf[col] / (2.0 * r)).clamp(0.0, PI / 2.0).sin() };
            let rr = (x[col] * x[col] + y[col] * y[col]).sqrt();
            let scale = if disk[col] || rr == 0.0 { 1.0 } else { r_nom / rr };
            x[col] *= scale;
            y[col] *= scale;
        }
    }

    if c.smear {
        for col in 0..sm {
            let rho = (x[col] * x[col] + y[col] * y[col]).sqrt();
            let s1 = os1[col];
            let s2 = os2[col];
            let w1 = if s1.is_finite() { s1 } else { 0.0 };
            let w2 = if s2.is_finite() { s2 } else { 0.0 };
            // rotate the hit by the azimuthal smear delta: same as atan2(y, x) + delta
            let delta = (rnd.ga[[i, col]] as f64 * w1) / rho.max(1e-9);
            let (sd, cd) = if delta.abs() < 1e-3 {
                let d2 = delta * delta;
                (delta * (1.0 - d2 / 6.0 * (1.0 - d2 / 20.0)), 1.0 - d2 / 2.0 * (1.0 - d2 / 12.0))
            } else {
                delta.sin_cos()
            };
            let (ca, sa) = if rho > 0.0 {
                ((x[col] * cd - y[col] * sd) / rho, (y[col] * cd + x[col] * sd) / rho)
            } else {
                (cd, sd)
            };
            let long = rnd.gl[[i, col]] as f64 * w2;
            let r_new = if disk[col] { rho + long } else { rho };
            x[col] = r_new * ca;
            y[col] = r_new * sa;
            if !disk[col] {
                z[col] += long;
            }
        }
    }

    let eff32 = c.eff as f32;
    for col in 0..sm {
        let cr = crossed[col] && surfs[idx[col]].sens;
        ocr[col] = cr;
        ode[col] = if c.eff < 1.0 { cr && rnd.u[[i, col]] < eff32 } else { cr };
        ox[col] = x[col];
        oy[col] = y[col];
        oz[col] = z[col];
    }
}

fn tracks_of(p: &ArrayView2<f64>, q: &ArrayView1<f64>, z0: &ArrayView1<f64>, cb: f64) -> Vec<Trk> {
    (0..p.nrows()).map(|i| track(p[[i, 0]], p[[i, 1]], p[[i, 2]], q[i], z0[i], cb)).collect()
}

/// Largest number of surfaces (sensitive or passive) crossed by any track.
#[pyfunction]
fn count_crossings(
    py: Python<'_>,
    p: PyReadonlyArray2<f64>,
    q: PyReadonlyArray1<f64>,
    z0: PyReadonlyArray1<f64>,
    table: PyReadonlyArray2<f64>,
    cb: f64,
) -> PyResult<usize> {
    let (p, q, z0) = (p.as_array(), q.as_array(), z0.as_array());
    let surfs = read_surfaces(table.as_array())?;
    let trks = tracks_of(&p, &q, &z0, cb);
    Ok(py.allow_threads(|| {
        trks.par_iter()
            .map(|t| surfs.iter().filter(|s| crossing(t, s).is_finite()).count())
            .max()
            .unwrap_or(0)
    }))
}

type HitsOut<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<bool>>,
    Bound<'py, PyArray2<bool>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<bool>>,
    Bound<'py, PyArray2<i64>>,
);

/// Hits of n tracks on smax sorted surfaces; see simulate_hits in simulation.py.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn simulate_hits<'py>(
    py: Python<'py>,
    p: PyReadonlyArray2<f64>,
    q: PyReadonlyArray1<f64>,
    z0: PyReadonlyArray1<f64>,
    table: PyReadonlyArray2<f64>,
    cb: f64,
    highland_e0: f64,
    m_mu: f64,
    smax: usize,
    multiple_scattering: bool,
    smear: bool,
    hit_efficiency: f64,
    th1: PyReadonlyArray2<f32>,
    th2: PyReadonlyArray2<f32>,
    ga: PyReadonlyArray2<f32>,
    gl: PyReadonlyArray2<f32>,
    u: PyReadonlyArray2<f32>,
) -> PyResult<HitsOut<'py>> {
    let (pa, qa, za) = (p.as_array(), q.as_array(), z0.as_array());
    let n = pa.nrows();
    let surfs = read_surfaces(table.as_array())?;
    let sm = smax.max(1);
    let need = |a: &ArrayView2<f32>, on: bool, what: &str| -> PyResult<()> {
        if on && (a.nrows() != n || a.ncols() != sm) {
            return Err(PyValueError::new_err(format!("{what}: expected shape ({n}, {sm})")));
        }
        Ok(())
    };
    let rnd = Randoms { th1: th1.as_array(), th2: th2.as_array(), ga: ga.as_array(), gl: gl.as_array(), u: u.as_array() };
    need(&rnd.th1, multiple_scattering, "th1")?;
    need(&rnd.th2, multiple_scattering, "th2")?;
    need(&rnd.ga, smear, "ga")?;
    need(&rnd.gl, smear, "gl")?;
    need(&rnd.u, hit_efficiency < 1.0, "u")?;
    let c = Consts { e0: highland_e0, m_mu, ms: multiple_scattering, smear, eff: hit_efficiency };
    let trks = tracks_of(&pa, &qa, &za, cb);
    if sm > surfs.len().max(1) {
        return Err(PyValueError::new_err("smax is larger than the number of surfaces"));
    }
    let overflow = std::sync::atomic::AtomicBool::new(false);

    let len = n * sm;
    let (mut x, mut y, mut z, mut s) = (vec![0.0; len], vec![0.0; len], vec![0.0; len], vec![0.0; len]);
    let (mut s1, mut s2) = (vec![0.0; len], vec![0.0; len]);
    let (mut cr, mut de, mut dk) = (vec![false; len], vec![false; len], vec![false; len]);
    let mut sf = vec![0i64; len];
    if n > 0 && !surfs.is_empty() {
        py.allow_threads(|| {
            (
                x.par_chunks_mut(sm),
                y.par_chunks_mut(sm),
                z.par_chunks_mut(sm),
                s.par_chunks_mut(sm),
                cr.par_chunks_mut(sm),
                de.par_chunks_mut(sm),
                s1.par_chunks_mut(sm),
                s2.par_chunks_mut(sm),
                dk.par_chunks_mut(sm),
                sf.par_chunks_mut(sm),
            )
                .into_par_iter()
                .enumerate()
                .for_each(|(i, (x, y, z, s, cr, de, s1, s2, dk, sf))| {
                    let mut too_many = false;
                    hits_row(i, &trks[i], &surfs, &c, &rnd, x, y, z, s, cr, de, s1, s2, dk, sf, &mut too_many);
                    if too_many {
                        overflow.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                });
        });
    }
    if overflow.into_inner() {
        return Err(PyValueError::new_err("smax is smaller than the number of surfaces crossed by a track"));
    }
    let sh = (n, sm);
    let f = |v: Vec<f64>| Array2::from_shape_vec(sh, v).unwrap().into_pyarray_bound(py);
    let b = |v: Vec<bool>| Array2::from_shape_vec(sh, v).unwrap().into_pyarray_bound(py);
    Ok((
        f(x),
        f(y),
        f(z),
        f(s),
        b(cr),
        b(de),
        f(s1),
        f(s2),
        b(dk),
        Array2::from_shape_vec(sh, sf).unwrap().into_pyarray_bound(py),
    ))
}

// ---------------------------------------------------------------------------------------
// track fitting

/// Solve the symmetric K×K system M x = v (Gaussian elimination, partial pivoting).
/// Returns None if a pivot is exactly zero (the matrix is singular).
fn solve<const K: usize>(m: &[[f64; K]; K], v: &[f64; K]) -> Option<[f64; K]> {
    let mut a = *m;
    let mut b = *v;
    for col in 0..K {
        let mut piv = col;
        for row in col + 1..K {
            if a[row][col].abs() > a[piv][col].abs() {
                piv = row;
            }
        }
        if a[piv][col] == 0.0 || !a[piv][col].is_finite() {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for row in col + 1..K {
            let f = a[row][col] / a[col][col];
            for c in col..K {
                a[row][c] -= f * a[col][c];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; K];
    for row in (0..K).rev() {
        let mut acc = b[row];
        for c in row + 1..K {
            acc -= a[row][c] * x[c];
        }
        x[row] = acc / a[row][row];
    }
    Some(x)
}

/// Pseudo-inverse of a symmetric matrix by Jacobi eigen-decomposition (as numpy.linalg.pinv, rcond 1e-15).
fn pinv_sym<const K: usize>(m: &[[f64; K]; K]) -> [[f64; K]; K] {
    let mut a = *m;
    let mut v = [[0.0; K]; K];
    for i in 0..K {
        v[i][i] = 1.0;
    }
    for _ in 0..100 {
        let mut off = 0.0;
        for p in 0..K {
            for q in p + 1..K {
                off += a[p][q] * a[p][q];
            }
        }
        if off < 1e-300 {
            break;
        }
        for p in 0..K {
            for q in p + 1..K {
                if a[p][q] == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let sgn = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sgn / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..K {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..K {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for k in 0..K {
                    let (vkp, vkq) = (v[k][p], v[k][q]);
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let lmax = (0..K).map(|i| a[i][i].abs()).fold(0.0, f64::max);
    let mut out = [[0.0; K]; K];
    for e in 0..K {
        let l = a[e][e];
        if l.abs() > 1e-15 * lmax && l != 0.0 {
            for i in 0..K {
                for j in 0..K {
                    out[i][j] += v[i][e] * v[j][e] / l;
                }
            }
        }
    }
    out
}

fn solve_or_pinv<const K: usize>(m: &[[f64; K]; K], v: &[f64; K]) -> [f64; K] {
    solve(m, v).unwrap_or_else(|| {
        let pi = pinv_sym(m);
        let mut x = [0.0; K];
        for i in 0..K {
            for j in 0..K {
                x[i] += pi[i][j] * v[j];
            }
        }
        x
    })
}

fn inverse3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for c in 0..3 {
        let mut e = [0.0; 3];
        e[c] = 1.0;
        match solve(m, &e) {
            Some(col) => {
                for r in 0..3 {
                    out[r][c] = col[r];
                }
            }
            None => return pinv_sym(m),
        }
    }
    out
}

/// Weighted normal equations for columns `cols` (each of length L), right-hand side and weights.
fn normal<const K: usize>(cols: &[&[f64]; K], rhs: &[f64], w: &[f64]) -> ([[f64; K]; K], [f64; K]) {
    let mut m = [[0.0; K]; K];
    let mut v = [0.0; K];
    for i in 0..K {
        for j in i..K {
            let mut acc = 0.0;
            for t in 0..w.len() {
                acc += w[t] * cols[i][t] * cols[j][t];
            }
            m[i][j] = acc;
            m[j][i] = acc;
        }
        let mut acc = 0.0;
        for t in 0..w.len() {
            acc += w[t] * cols[i][t] * rhs[t];
        }
        v[i] = acc;
    }
    (m, v)
}

#[derive(Clone, Copy)]
struct FitOut {
    p: [f64; 3],
    q: f64,
    ok: bool,
    nh: i64,
    sigma_pt: f64,
    d0: f64,
    z0: f64,
    chi2: f64,
}

struct FitIn<'a> {
    x: ArrayView2<'a, f64>,
    y: ArrayView2<'a, f64>,
    z: ArrayView2<'a, f64>,
    det: ArrayView2<'a, bool>,
    sig1: ArrayView2<'a, f64>,
    sig2: ArrayView2<'a, f64>,
    disk: ArrayView2<'a, bool>,
}

fn fit_row(i: usize, a: &FitIn, cb: f64, beamspot: f64, min_hits: i64, iterations: usize) -> FitOut {
    const L: usize = MAXS + 1;
    let ns = a.x.ncols();
    let nan = f64::NAN;
    let nh = (0..ns).filter(|&j| a.det[[i, j]]).count() as i64;
    let mut out = FitOut { p: [nan; 3], q: 0.0, ok: false, nh, sigma_pt: nan, d0: nan, z0: nan, chi2: nan };
    if nh < min_hits.max(3) || ns > MAXS {
        return out;
    }
    let off = if beamspot > 0.0 { 1 } else { 0 };
    let l = ns + off;
    let (mut xs, mut ys, mut w) = ([0.0f64; L], [0.0f64; L], [0.0f64; L]);
    if off == 1 {
        w[0] = 1.0 / (beamspot * beamspot);
    }
    for j in 0..ns {
        let s = a.sig1[[i, j]];
        if a.det[[i, j]] {
            w[j + off] = 1.0 / (s * s);
            if w[j + off] > 0.0 {
                xs[j + off] = a.x[[i, j]];
                ys[j + off] = a.y[[i, j]];
            }
        }
    }
    // algebraic start: x² + y² = 2a x + 2b y + c
    let (mut c1, mut c2, mut rhs) = ([0.0f64; L], [0.0f64; L], [0.0f64; L]);
    let ones = [1.0f64; L];
    for t in 0..l {
        c1[t] = 2.0 * xs[t];
        c2[t] = 2.0 * ys[t];
        rhs[t] = xs[t] * xs[t] + ys[t] * ys[t];
    }
    let (m, v) = normal(&[&c1[..l], &c2[..l], &ones[..l]], &rhs[..l], &w[..l]);
    let [mut ca, mut cbb, cc] = solve_or_pinv(&m, &v);
    let mut r = (ca * ca + cbb * cbb + cc).max(1e-12).sqrt();
    let (mut gx, mut gy, mut res) = ([0.0f64; L], [0.0f64; L], [0.0f64; L]);
    let gneg = [-1.0f64; L];
    let build = |ca: f64, cbb: f64, r: f64, gx: &mut [f64; L], gy: &mut [f64; L], res: &mut [f64; L]| {
        for t in 0..l {
            let dx = xs[t] - ca;
            let dy = ys[t] - cbb;
            let d = (dx * dx + dy * dy).sqrt().max(1e-12);
            gx[t] = -dx / d;
            gy[t] = -dy / d;
            res[t] = d - r;
        }
    };
    for _ in 0..iterations {
        build(ca, cbb, r, &mut gx, &mut gy, &mut res);
        let (m, v) = normal(&[&gx[..l], &gy[..l], &gneg[..l]], &res[..l], &w[..l]);
        let step = solve_or_pinv(&m, &v);
        ca -= step[0];
        cbb -= step[1];
        r -= step[2];
        if (step[2].abs() / r) < 1e-11 {
            break;
        }
    }
    build(ca, cbb, r, &mut gx, &mut gy, &mut res);
    let (m, _) = normal(&[&gx[..l], &gy[..l], &gneg[..l]], &res[..l], &w[..l]);
    let cov = inverse3(&m);
    let nw = w[..l].iter().filter(|&&v| v > 0.0).count() as i64;
    let ndf = (nw - 3).max(1) as f64;
    let chi2: f64 = (0..l).map(|t| w[t] * res[t] * res[t]).sum::<f64>() / ndf;

    let pt = cb * r;
    let first = (0..ns).find(|&j| a.det[[i, j]]).unwrap();
    let last = (0..ns).rev().find(|&j| a.det[[i, j]]).unwrap();
    let (x1, y1, x2, y2) = (a.x[[i, first]], a.y[[i, first]], a.x[[i, last]], a.y[[i, last]]);
    let mut k = -sign(x1 * y2 - y1 * x2);
    if k == 0.0 {
        k = 1.0;
    }
    let phi0 = cbb.atan2(ca) + k * PI / 2.0;
    let d0 = (ca * ca + cbb * cbb).sqrt() - r;

    // longitudinal fit: z = z0 + cot θ · s
    let (mut s, mut zr, mut wz) = ([0.0f64; L], [0.0f64; L], [0.0f64; L]);
    for j in 0..ns {
        let (xj, yj) = (a.x[[i, j]], a.y[[i, j]]);
        let rho = (xj * xj + yj * yj).sqrt();
        s[j] = 2.0 * r * (rho / (2.0 * r)).clamp(-1.0, 1.0).asin();
        zr[j] = if a.det[[i, j]] { a.z[[i, j]] } else { 0.0 };
    }
    let (mut z0, mut cot) = (0.0f64, 0.0f64);
    for it in 0..2 {
        for j in 0..ns {
            let sl = a.sig2[[i, j]];
            let se = if a.disk[[i, j]] && it > 0 { sl * cot.abs().max(0.1) } else { sl };
            wz[j] = if a.det[[i, j]] && se.is_finite() && se > 0.0 { 1.0 / (se * se) } else { 0.0 };
        }
        if wz[..ns].iter().filter(|&&v| v > 0.0).count() < 2 {
            for j in 0..ns {
                wz[j] = if a.det[[i, j]] { 1.0 } else { 0.0 };
            }
        }
        let (m, v) = normal(&[&ones[..ns], &s[..ns]], &zr[..ns], &wz[..ns]);
        let sol = solve_or_pinv(&m, &v);
        z0 = sol[0];
        cot = sol[1];
    }
    let (sphi, cphi) = phi0.sin_cos();
    let p = [pt * cphi, pt * sphi, pt * cot];
    out.p = p;
    out.q = k;
    out.ok = p.iter().all(|v| v.is_finite()) && r > 0.0;
    out.sigma_pt = cb * cov[2][2].abs().sqrt();
    out.d0 = d0;
    out.z0 = z0;
    out.chi2 = chi2;
    out
}

type FitResult<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

/// Circle + line fits of n tracks; see fit_tracks in tracking.py.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_tracks<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    z: PyReadonlyArray2<f64>,
    detected: PyReadonlyArray2<bool>,
    sigma_rphi: PyReadonlyArray2<f64>,
    sigma_long: PyReadonlyArray2<f64>,
    is_disk: PyReadonlyArray2<bool>,
    cb: f64,
    beamspot: f64,
    min_hits: i64,
    iterations: usize,
) -> PyResult<FitResult<'py>> {
    let a = FitIn {
        x: x.as_array(),
        y: y.as_array(),
        z: z.as_array(),
        det: detected.as_array(),
        sig1: sigma_rphi.as_array(),
        sig2: sigma_long.as_array(),
        disk: is_disk.as_array(),
    };
    let sh = a.x.dim();
    for (d, what) in [(a.y.dim(), "y"), (a.z.dim(), "z"), (a.det.dim(), "detected"), (a.sig1.dim(), "sigma_rphi"),
                      (a.sig2.dim(), "sigma_long"), (a.disk.dim(), "is_disk")] {
        if d != sh {
            return Err(PyValueError::new_err(format!("{what} has a different shape from x")));
        }
    }
    let n = sh.0;
    let fits: Vec<FitOut> =
        py.allow_threads(|| (0..n).into_par_iter().map(|i| fit_row(i, &a, cb, beamspot, min_hits, iterations)).collect());
    let mut p = Array2::<f64>::zeros((n, 3));
    for (i, f) in fits.iter().enumerate() {
        for c in 0..3 {
            p[[i, c]] = f.p[c];
        }
    }
    let col = |g: fn(&FitOut) -> f64| Array1::from_iter(fits.iter().map(g)).into_pyarray_bound(py);
    Ok((
        p.into_pyarray_bound(py),
        col(|f| f.q),
        Array1::from_iter(fits.iter().map(|f| f.ok)).into_pyarray_bound(py),
        Array1::from_iter(fits.iter().map(|f| f.nh)).into_pyarray_bound(py),
        col(|f| f.sigma_pt),
        col(|f| f.d0),
        col(|f| f.z0),
        col(|f| f.chi2),
    ))
}

#[pymodule]
fn particlelab_fast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("API_VERSION", API_VERSION)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(count_crossings, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_hits, m)?)?;
    m.add_function(wrap_pyfunction!(fit_tracks, m)?)?;
    Ok(())
}
