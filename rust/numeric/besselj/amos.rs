//! A transliteration of the AMOS routines scipy 1.18.1 reaches for a Bessel function of the
//! first kind of a real argument (`xsf/amos/amos.h`, the SciPy developers' C++ translation of
//! D. E. Amos's Fortran, ACM TOMS Algorithm 644): `besj`, `besy`, `besh`, `binu`, `seri`,
//! `asyi`, `mlri`, `uoik`, `buni`, `uni2`, `unhj`, `bknu`, `kscl`, `rati`, `wrsk`, `bunk`,
//! `unk2`, `airy`, `acai`, `s1s2`, `uchk`, `gamln`. Statement for statement, with every complex
//! operation computed as the C++ computes it (`cx`), the C++'s own quirks kept (they decide
//! bits): `seri`'s shadowed loop index, `buni`'s and `unk2`'s scaling bounds, `uoik`'s real
//! `exp`, and the C++'s left-to-right sums: where it writes `fnu + n - 1` (`binu`, `rati`,
//! `uni2`, `uoik`) the sum is `(fnu + n) - 1`, which drops `fnu`'s last bit when `fnu + n`
//! crosses a power of two, and where it writes `fnu + (n - 1)` it is exact.
//!
//! A real argument reaches these routines rotated onto the imaginary axis (`z = (0, -x)` or
//! `(0, x)`), never in the right or left half plane, so the branches the C++ takes only off
//! that axis (`uni1`, `unik`, `unk1`, `acon`, and `unk2`'s continuation to the left half plane)
//! are not ported: each such branch is marked, and returns the routine's overflow code, so the
//! value becomes NaN (and `cyl_bessel_j` falls back to Cephes) rather than anything unported.
//!
//! Copyright (C) 2024 SciPy developers (BSD-3-Clause); the Fortran is Sandia's, approved for
//! unlimited release. Notices in `LICENSES_bundled.txt`.

#![allow(unused_assignments)] // the C++'s assignments are kept, read or not

use super::cx::{c_int, C64};
use super::tables as t;
use crate::numeric::elementary as e;

const D1MACH: [f64; 5] = [
    2.2250738585072014e-308,
    1.7976931348623157e+308,
    1.1102230246251565e-16,
    2.220446049250313e-16,
    0.3010299956639812,
];

const I1MACH: [f64; 16] = [
    5.0,
    6.0,
    7.0,
    0.0,
    32.0,
    4.0,
    2.0,
    31.0,
    2147483647.0,
    2.0,
    24.0,
    -126.0,
    128.0,
    53.0,
    -1021.0,
    1024.0,
];

#[inline]
fn fmax(a: f64, b: f64) -> f64 {
    e::fmax(a, b)
}

#[inline]
fn fmin(a: f64, b: f64) -> f64 {
    e::fmin(a, b)
}

/// The machine parameters every entry routine derives the same way:
/// `(tol, elim, alim, dig, fnul, rl)`.
fn machine() -> (f64, f64, f64, f64, f64, f64) {
    let tol = fmax(D1MACH[3], 1e-18);
    let k1 = c_int(I1MACH[14]);
    let k2 = c_int(I1MACH[15]);
    let r1m5 = D1MACH[4];
    let k = if k1.abs() > k2.abs() {
        k2.abs()
    } else {
        k1.abs()
    };
    let elim = 2.303 * (k as f64 * r1m5 - 3.0);
    let k1 = c_int(I1MACH[13] - 1.0);
    let mut aa = r1m5 * k1 as f64;
    let dig = fmin(aa, 18.0);
    aa *= 2.303;
    let alim = elim + fmax(-aa, -41.45);
    let fnul = 10.0 + 6.0 * (dig - 3.0);
    let rl = 1.2 * dig + 3.0;
    (tol, elim, alim, dig, fnul, rl)
}

fn uchk(y: C64, ascle: f64, tol: f64) -> bool {
    let yr = y.re.abs();
    let yi = y.im.abs();
    let ss = fmax(yr, yi);
    let mut st = fmin(yr, yi);
    if st > ascle {
        false
    } else {
        st /= tol;
        ss < st
    }
}

pub(super) fn gamln(z: f64) -> f64 {
    const CON: f64 = 1.83787706640934548;
    let mut nz: i32 = 0;
    if z > 0.0 {
        if z <= 101.0 {
            nz = c_int(z);
            let fz = z - nz as f64;
            if fz <= 0.0 && nz <= 100 {
                return t::DGAMLN_GLN[(nz - 1) as usize];
            }
        }
        let wdtol = fmax(D1MACH[3], 1e-18);
        let i1m = c_int(I1MACH[13]);
        let rln = D1MACH[4] * i1m as f64;
        let fln = fmax(fmin(rln, 20.0), 3.0) - 3.0;
        let zm = 1.8 + 0.3875 * fln;
        let mz = c_int(zm) + 1;
        let zmin = mz as f64;
        let mut zdmy = z;
        let mut zinc = 0.0;
        if z < zmin {
            zinc = zmin - nz as f64;
            zdmy = z + zinc;
        }
        let mut zp = 1.0 / zdmy;
        let t1 = t::DGAMLN_CF[0] * zp;
        let mut s = t1;
        if zp >= wdtol {
            let zsq = zp * zp;
            let tst = t1 * wdtol;
            for i in 2..23 {
                zp *= zsq;
                let trm = t::DGAMLN_CF[i - 1] * zp;
                if trm.abs() < tst {
                    break;
                }
                s += trm;
            }
        }
        if zinc == 0.0 {
            let tlg = e::log(z);
            return z * (tlg - 1.0) + 0.5 * (CON - tlg) + s;
        }
        zp = 1.0;
        let nzi = c_int(zinc);
        for i in 0..nzi {
            zp *= z + i as f64;
        }
        let tlg = e::log(zdmy);
        return zdmy * (tlg - 1.0) - e::log(zp) + 0.5 * (CON - tlg) + s;
    }
    f64::NAN
}

fn s1s2(zr: C64, s1: &mut C64, s2: &mut C64, ascle: f64, alim: f64, iuf: &mut i32) -> i32 {
    let mut as1 = s1.abs();
    let as2 = s2.abs();
    let aa = s1.re;
    let aln = s1.im;
    if (aa != 0.0 || aln != 0.0) && as1 != 0.0 {
        let xx = zr.re;
        let aln = -xx - xx + e::log(as1);
        let s1d = *s1;
        *s1 = C64::ZERO;
        as1 = 0.0;
        if aln >= -alim {
            let c1 = s1d.ln() - zr - zr;
            *s1 = c1.exp();
            as1 = s1.abs();
            *iuf += 1;
        }
    }
    let aa = fmax(as1, as2);
    if aa > ascle {
        return 0;
    }
    *s1 = C64::ZERO;
    *s2 = C64::ZERO;
    *iuf = 0;
    1
}

/// `seri`: I(fnu, z) by the power series, for |z| <= 2 sqrt(fnu + 1).
#[allow(clippy::too_many_arguments)]
fn seri(z: C64, fnu: f64, kode: i32, n: i32, y: &mut [C64], tol: f64, elim: f64, alim: f64) -> i32 {
    let mut nz: i32 = 0;
    let az = z.abs();
    if az == 0.0 {
        y[0] = C64::ZERO;
        if fnu == 0.0 {
            y[0] = C64::real(1.0);
        }
        if n == 1 {
            return nz;
        }
        for v in y.iter_mut().take(n as usize).skip(1) {
            *v = C64::ZERO;
        }
        return nz;
    }
    let x = z.re;
    let arm = 1e3 * D1MACH[0];
    let rtr1 = arm.sqrt();
    let mut crsc = C64::real(1.0);
    let mut iflag = 0;
    if az >= arm {
        let half_z = 0.5 * z;
        let mut cz = C64::ZERO;
        if az > rtr1 {
            cz = half_z * half_z;
        }
        let acz = cz.abs();
        let mut nn = n;
        let ck = half_z.ln();
        let mut ss = 0.0;
        let mut ascle = 0.0;
        let mut w = [C64::ZERO; 2];
        loop {
            // L10
            let mut dfnu = fnu + (nn - 1) as f64;
            let mut fnup = dfnu + 1.0;
            let mut ak1 = ck * dfnu;
            let mut ak = gamln(fnup);
            ak1 -= ak;
            if kode == 2 {
                ak1 -= x;
            }
            let rak1 = ak1.re;
            let mut to_l20 = !(rak1 > -elim);
            if !to_l20 {
                // L30
                if rak1 <= -alim {
                    iflag = 1;
                    ss = 1.0 / tol;
                    crsc = C64::real(tol);
                    ascle = arm * ss;
                }
                ak = ak1.im;
                let mut aa = e::exp(rak1);
                if iflag == 1 {
                    aa *= ss;
                }
                let mut coef = aa * C64::new(e::cos(ak), e::sin(ak));
                let atol = tol * acz / fnup;
                let il = if nn > 2 { 2 } else { nn };
                for i in 1..(il + 1) {
                    dfnu = fnu + (nn - i) as f64;
                    fnup = dfnu + 1.0;
                    let mut s1 = C64::real(1.0);
                    if acz >= tol * fnup {
                        let mut ak1 = C64::real(1.0);
                        let mut ak = fnup + 2.0;
                        let mut s = fnup;
                        let mut aa = 2.0;
                        loop {
                            let rs = 1.0 / s;
                            ak1 *= cz;
                            ak1 *= rs;
                            s1 += ak1;
                            s += ak;
                            ak += 2.0;
                            aa *= acz;
                            aa *= rs;
                            if aa <= atol {
                                break;
                            }
                        }
                    }
                    let s2 = s1 * coef;
                    w[(i - 1) as usize] = s2;
                    if iflag != 0 && uchk(s2, ascle, tol) {
                        to_l20 = true;
                        break;
                    }
                    let m = nn - i + 1;
                    y[(m - 1) as usize] = s2 * crsc;
                    if i != il {
                        coef *= dfnu / half_z;
                    }
                }
                if !to_l20 {
                    if nn <= 2 {
                        return nz;
                    }
                    let mut k = nn - 2;
                    let mut ak = k as f64;
                    let rz = 2.0 / z;
                    let mut ib = 3;
                    if iflag == 1 {
                        // L80: recur backward with scaled values
                        let mut s1 = w[0];
                        let mut s2 = w[1];
                        let l = 3;
                        let mut reached = false;
                        for _l in 3..(nn + 1) {
                            let ck = s2;
                            s2 = s1 + (ak + fnu) * rz * s2;
                            s1 = ck;
                            let ck = s2 * crsc;
                            y[(k - 1) as usize] = ck;
                            ak -= 1.0;
                            k -= 1;
                            if ck.abs() > ascle {
                                reached = true;
                                break;
                            }
                        }
                        if !reached {
                            return nz;
                        }
                        // L100: the C++ reads the OUTER `l`, which its loop shadows: still 3
                        ib = l + 1;
                        if ib > nn {
                            return nz;
                        }
                    }
                    // L60
                    for _i in ib..(nn + 1) {
                        y[(k - 1) as usize] = (ak + fnu) * rz * y[k as usize] + y[(k + 1) as usize];
                        ak -= 1.0;
                        k -= 1;
                    }
                    return nz;
                }
            }
            // L20
            nz += 1;
            y[(nn - 1) as usize] = C64::ZERO;
            if acz > dfnu {
                return -nz;
            }
            nn -= 1;
            if nn == 0 {
                return nz;
            }
        }
    }
    nz = n;
    if fnu == 0.0 {
        nz -= 1;
    }
    y[0] = C64::ZERO;
    if fnu == 0.0 {
        y[0] = C64::real(1.0);
    }
    if n == 1 {
        return nz;
    }
    for v in y.iter_mut().take(n as usize).skip(1) {
        *v = C64::ZERO;
    }
    nz
}

/// `asyi`: I(fnu, z) by the asymptotic expansion for large |z|.
#[allow(clippy::too_many_arguments)]
fn asyi(
    z: C64,
    fnu: f64,
    kode: i32,
    n: i32,
    y: &mut [C64],
    rl: f64,
    tol: f64,
    elim: f64,
    alim: f64,
) -> i32 {
    let pi = 3.14159265358979324;
    let rpi = 0.159154943091895336;
    let nz = 0;
    let az = z.abs();
    let x = z.re;
    let arm = 1e3 * D1MACH[0];
    let rtr1 = arm.sqrt();
    let il = if n > 2 { 2 } else { n };
    let dfnu = fnu + (n - il) as f64;
    let mut ak1 = (rpi / z).sqrt();
    let mut cz = z;
    if kode == 2 {
        cz = C64::new(0.0, z.im);
    }
    let acz = cz.re;
    if acz.abs() <= elim {
        let dnu2 = dfnu + dfnu;
        let mut koded = 1;
        if !((acz.abs() > alim) && (n > 2)) {
            koded = 0;
            ak1 *= cz.exp();
        }
        let mut fdn = 0.0;
        if dnu2 > rtr1 {
            fdn = dnu2 * dnu2;
        }
        let ez = z * 8.0;
        let aez = 8.0 * az;
        let s = tol / aez;
        let jl = c_int(rl + rl).wrapping_add(2);
        let yy = z.im;
        let mut p1 = C64::ZERO;
        if yy != 0.0 {
            let mut inu = c_int(fnu);
            let arg = (fnu - inu as f64) * pi;
            inu = inu.wrapping_add(n - il);
            let ak = -e::sin(arg);
            let mut bk = e::cos(arg);
            if yy < 0.0 {
                bk = -bk;
            }
            p1 = C64::new(ak, bk);
            if inu % 2 == 1 {
                p1 = -p1;
            }
        }
        for k in 1..(il + 1) {
            let mut sqk = fdn - 1.0;
            let atol = s * sqk.abs();
            let mut sgn = 1.0;
            let mut cs1 = C64::real(1.0);
            let mut cs2 = C64::real(1.0);
            let mut ck = C64::real(1.0);
            let mut ak = 0.0;
            let mut aa = 1.0;
            let mut bb = aez;
            let mut dk = ez;
            let mut j = 1;
            while j < jl + 1 {
                ck *= sqk / dk;
                cs2 += ck;
                sgn = -sgn;
                cs1 += ck * sgn;
                dk += ez;
                aa *= sqk.abs() / bb;
                bb += aez;
                ak += 8.0;
                sqk -= ak;
                if aa <= atol {
                    break;
                }
                j += 1;
            }
            if (j == jl + 1) && (aa > atol) {
                return -2;
            }
            let mut s2 = cs1;
            if x + x < elim {
                s2 += p1 * cs2 * (-z - z).exp();
            }
            fdn += 8.0 * dfnu + 4.0;
            p1 = -p1;
            let m = n - il + k;
            y[(m - 1) as usize] = s2 * ak1;
        }
        if n <= 2 {
            return nz;
        }
        let nn = n;
        let mut k = nn - 2;
        let mut ak = k as f64;
        let rz = 2.0 / z;
        for _i in 3..(nn + 1) {
            y[(k - 1) as usize] = (ak + fnu) * rz * y[k as usize] + y[(k + 1) as usize];
            ak -= 1.0;
            k -= 1;
        }
        if koded == 0 {
            return nz;
        }
        // The C++ scales y[0..=nn], one element past the sequence; only n > 2 gets here, which
        // no real-argument path requests.
        let ck = cz.exp();
        for v in y.iter_mut().take(nn as usize) {
            *v *= ck;
        }
        return nz;
    }
    -1
}

/// `mlri`: I(fnu, z) by the Miller algorithm normalized by a Neumann series.
fn mlri(z: C64, fnu: f64, kode: i32, n: i32, y: &mut [C64], tol: f64) -> i32 {
    let scle = D1MACH[0] / tol;
    let nz = 0;
    let az = z.abs();
    let x = z.re;
    let iaz = c_int(az);
    let ifnu = c_int(fnu);
    let inu = ifnu.wrapping_add(n - 1);
    let mut at = iaz.wrapping_add(1) as f64;
    let mut ck = at / z;
    let rz = 2.0 / z;
    let mut p1 = C64::ZERO;
    let mut p2 = C64::real(1.0);
    let mut ack = (at + 1.0) / az;
    let mut rho = ack + (ack * ack - 1.0).sqrt();
    let rho2 = rho * rho;
    let mut tst = (rho2 + rho2) / ((rho2 - 1.0) * (rho - 1.0));
    tst /= tol;
    let mut ak = at;
    let mut i: i32 = 1;
    while i < 81 {
        let pt = p2;
        p2 = p1 - ck * p2;
        p1 = pt;
        ck += rz;
        let ap = p2.abs();
        if ap > tst * ak * ak {
            break;
        }
        ak += 1.0;
        if i == 80 {
            return -2;
        }
        i += 1;
    }
    i += 1;
    let mut k: i32 = 0;
    if inu >= iaz {
        p1 = C64::ZERO;
        p2 = C64::real(1.0);
        at = inu.wrapping_add(1) as f64;
        ck = at / z;
        ack = at / az;
        tst = (ack / tol).sqrt();
        let mut itime = 1;
        k = 1;
        while k < 81 {
            let pt = p2;
            p2 = p1 - ck * p2;
            p1 = pt;
            ck += rz;
            let ap = p2.abs();
            if ap >= tst {
                if itime == 2 {
                    break;
                }
                ack = ck.abs();
                let flam = ack + (ack * ack - 1.0).sqrt();
                let fkap = ap / p1.abs();
                rho = fmin(flam, fkap);
                tst *= (rho / (rho * rho - 1.0)).sqrt();
                itime = 2;
            }
            if k == 80 {
                return -2;
            }
            k += 1;
        }
    }
    k += 1;
    let kk = c_int(fmax(i.wrapping_add(iaz) as f64, k.wrapping_add(inu) as f64));
    let mut fkk = kk as f64;
    p1 = C64::ZERO;
    p2 = C64::real(scle);
    let fnf = fnu - ifnu as f64;
    let tfnf = fnf + fnf;
    let mut bk = gamln(fkk + tfnf + 1.0) - gamln(fkk + 1.0) - gamln(tfnf + 1.0);
    bk = e::exp(bk);
    let mut sum = C64::ZERO;
    let km = kk.wrapping_sub(inu);
    let step = |p1: &mut C64, p2: &mut C64, sum: &mut C64, bk: &mut f64, fkk: &mut f64| {
        let pt = *p2;
        *p2 = *p1 + (*fkk + fnf) * rz * *p2;
        *p1 = pt;
        let ak = 1.0 - tfnf / (*fkk + tfnf);
        let ack = *bk * ak;
        *sum += (ack + *bk) * *p1;
        *bk = ack;
        *fkk -= 1.0;
    };
    for _ in 1..(km.wrapping_add(1)) {
        step(&mut p1, &mut p2, &mut sum, &mut bk, &mut fkk);
    }
    y[(n - 1) as usize] = p2;
    if n != 1 {
        for i in 2..(n + 1) {
            step(&mut p1, &mut p2, &mut sum, &mut bk, &mut fkk);
            let m = n - i + 1;
            y[(m - 1) as usize] = p2;
        }
    }
    if ifnu > 0 {
        for _ in 1..(ifnu + 1) {
            step(&mut p1, &mut p2, &mut sum, &mut bk, &mut fkk);
        }
    }
    let mut pt = z;
    if kode == 2 {
        pt -= x;
    }
    p1 = -fnf * rz.ln() + pt;
    let ap = gamln(1.0 + fnf);
    pt = p1 - ap;
    p2 += sum;
    let ap = p2.abs();
    p1 = C64::real(1.0 / ap);
    let ck = pt.exp() * p1;
    let pt = p2.conj() * p1;
    let cnorm = ck * pt;
    for v in y.iter_mut().take(n as usize) {
        *v *= cnorm;
    }
    nz
}

/// `kscl`: set K functions to zero on underflow, continue the recurrence on scaled values.
#[allow(clippy::too_many_arguments)]
fn kscl(
    zr: C64,
    fnu: f64,
    n: i32,
    y: &mut [C64],
    rz: C64,
    ascle: &mut f64,
    tol: f64,
    elim: f64,
) -> i32 {
    let mut cy = [C64::ZERO; 2];
    let mut nz = 0;
    let mut ic = 0;
    let nn = if n > 2 { 2 } else { n };
    let elm = e::exp(-elim);
    let mut xx = zr.re;
    for i in 1..(nn + 1) {
        let s1 = y[(i - 1) as usize];
        cy[(i - 1) as usize] = s1;
        let as_ = s1.abs();
        let acs = -zr.re + e::log(as_);
        nz += 1;
        y[(i - 1) as usize] = C64::ZERO;
        if acs < -elim {
            continue;
        }
        let mut cs = -zr + s1.ln();
        cs = (e::exp(cs.re) / tol) * C64::new(e::cos(cs.im), e::sin(cs.im));
        if !uchk(cs, *ascle, tol) {
            y[(i - 1) as usize] = cs;
            nz -= 1;
            ic = i;
        }
    }
    if n == 1 {
        return nz;
    }
    if ic <= 1 {
        y[0] = C64::ZERO;
        nz = 2;
    }
    if n == 2 {
        return nz;
    }
    if nz == 0 {
        return nz;
    }
    let fn_ = fnu + 1.0;
    let mut ck = fn_ * rz;
    let mut s1 = cy[0];
    let mut s2 = cy[1];
    let zri = zr.im;
    let mut zd = zr;
    for i in 3..(n + 1) {
        let kk = i;
        let cs = s2;
        s2 *= ck;
        s2 += s1;
        s1 = cs;
        ck += rz;
        let as_ = s2.abs();
        let alas = e::log(as_);
        let acs = alas - xx;
        nz += 1;
        y[(i - 1) as usize] = C64::ZERO;
        if acs >= -elim {
            let mut cs = s2.ln();
            cs -= zd;
            cs = (e::exp(cs.re) / tol) * C64::new(e::cos(cs.im), e::sin(cs.im));
            if !uchk(cs, *ascle, tol) {
                y[(i - 1) as usize] = cs;
                nz -= 1;
                if ic == kk - 1 {
                    nz = kk - 2;
                    for v in y.iter_mut().take(nz as usize) {
                        *v = C64::ZERO;
                    }
                    return nz;
                }
                ic = kk;
                continue;
            }
        }
        if alas >= 0.5 * elim {
            xx -= elim;
            zd = C64::new(xx, zri);
            s1 *= elm;
            s2 *= elm;
        }
    }
    nz = n;
    if ic == n {
        nz = n - 1;
    }
    for v in y.iter_mut().take(nz as usize) {
        *v = C64::ZERO;
    }
    nz
}

/// `bknu`: K(fnu, z) in the right half plane.
#[allow(clippy::too_many_arguments)]
fn bknu(z: C64, fnu: f64, kode: i32, n: i32, y: &mut [C64], tol: f64, elim: f64, alim: f64) -> i32 {
    #[derive(Clone, Copy)]
    enum L {
        L100,
        L110,
        L130,
        L140,
        L160,
        L190,
    }
    let kmax = 30;
    let r1 = 2.0;
    let pi = 3.14159265358979324;
    let rthpi = 1.25331413731550025;
    let spi = 1.90985931710274403;
    let fpi = 1.89769999331517738;
    let tth = 2.0 / 3.0;
    let cc = [
        5.77215664901532861e-01,
        -4.20026350340952355e-02,
        -4.21977345555443367e-02,
        7.21894324666309954e-03,
        -2.15241674114950973e-04,
        -2.01348547807882387e-05,
        1.13302723198169588e-06,
        6.11609510448141582e-09,
    ];
    let xx = z.re;
    let yy = z.im;
    let caz = z.abs();
    let cscl = 1.0 / tol;
    let crsc = tol;
    let css = [C64::real(cscl), C64::real(1.0), C64::real(crsc)];
    let csr = [C64::real(crsc), C64::real(1.0), C64::real(cscl)];
    let bry = [1e3 * D1MACH[0] / tol, tol / (1e3 * D1MACH[0]), D1MACH[1]];
    let mut nz = 0;
    let mut iflag = 0;
    let mut koded = kode;
    let rz = 2.0 / z;
    let mut inu = c_int(fnu + 0.5);
    let dnu = fnu - inu as f64;
    let mut s1 = C64::ZERO;
    let mut s2 = C64::ZERO;
    let mut ck;
    let mut dnu2 = 0.0;
    let mut kflag = 2;
    let mut zd = z;
    let mut inub = 1;
    let mut kk = 0;
    let mut p1;
    let mut p2;
    let mut ascle;
    let mut cy = [C64::ZERO; 2];
    let mut label: L;

    'start: {
        if dnu.abs() != 0.5 {
            if dnu.abs() > tol {
                dnu2 = dnu * dnu;
            }
            if caz <= r1 {
                // series for |z| <= r1
                let mut fc = 1.0;
                let mut smu = rz.ln();
                let fmu = smu * dnu;
                let csh = fmu.sinh();
                let cch = fmu.cosh();
                if dnu != 0.0 {
                    fc = dnu * pi;
                    fc *= 1.0 / e::sin(fc);
                    smu = csh / dnu;
                }
                let a2 = 1.0 + dnu;
                let t2 = e::exp(-gamln(a2));
                let mut t1 = 1.0 / (t2 * fc);
                let g1;
                if dnu.abs() <= 0.1 {
                    let mut ak = 1.0;
                    let mut s = cc[0];
                    for k in 2..9 {
                        ak *= dnu2;
                        let tm = cc[k - 1] * ak;
                        s += tm;
                        if tm.abs() < tol {
                            break;
                        }
                    }
                    g1 = -s;
                } else {
                    g1 = (t1 - t2) / (dnu + dnu);
                }
                let g2 = 0.5 * (t1 + t2);
                let mut f = fc * (g1 * cch + smu * g2);
                let pt = fmu.exp();
                let mut p = (0.5 / t2) * pt;
                let mut q = (0.5 / t1) / pt;
                s1 = f;
                s2 = p;
                let mut ak = 1.0;
                let mut a1 = 1.0;
                ck = C64::real(1.0);
                let mut bk = 1.0 - dnu2;
                if (inu <= 0) && (n <= 1) {
                    if caz >= tol {
                        let cz = z * z * 0.25;
                        t1 = 0.25 * caz * caz;
                        loop {
                            f = (f * ak + p + q) / bk;
                            p = p / (ak - dnu);
                            q = q / (ak + dnu);
                            let rk = 1.0 / ak;
                            ck *= cz * rk;
                            s1 += ck * f;
                            a1 *= t1 * rk;
                            bk += ak + ak + 1.0;
                            ak += 1.0;
                            if !(a1 > tol) {
                                break;
                            }
                        }
                    }
                    y[0] = s1;
                    if koded == 1 {
                        return nz;
                    }
                    y[0] = s1 * z.exp();
                    return nz;
                }
                if caz >= tol {
                    let cz = z * z * 0.25;
                    t1 = 0.25 * caz * caz;
                    loop {
                        f = (f * ak + p + q) / bk;
                        p *= 1.0 / (ak - dnu);
                        q *= 1.0 / (ak + dnu);
                        let rk = 1.0 / ak;
                        ck *= cz * rk;
                        s1 += ck * f;
                        s2 += ck * (p - f * ak);
                        a1 *= t1 * rk;
                        bk += ak + ak + 1.0;
                        ak += 1.0;
                        if !(a1 > tol) {
                            break;
                        }
                    }
                }
                kflag = 2;
                let bk = smu.re;
                let a1 = fnu + 1.0;
                let ak = a1 * bk.abs();
                if ak > alim {
                    kflag = 3;
                }
                let p2 = s2 * css[(kflag - 1) as usize];
                s2 = p2 * rz;
                s1 *= css[(kflag - 1) as usize];
                if koded != 1 {
                    let f = z.exp();
                    s1 *= f;
                    s2 *= f;
                }
                label = L::L100;
                break 'start;
            }
        }
        // IFLAG=0: no underflow; IFLAG=1: an underflow, the computation proceeds with KODED=2
        let mut coef = rthpi / z.sqrt();
        kflag = 2;
        if koded != 2 {
            if xx > alim {
                koded = 2;
                iflag = 1;
                kflag = 2;
            } else {
                let a1 = e::exp(-xx) * css[(kflag - 1) as usize].re;
                let pt = a1 * C64::new(e::cos(yy), -e::sin(yy));
                coef *= pt;
            }
        }
        if dnu.abs() == 0.5 {
            s1 = coef;
            s2 = coef;
            label = L::L100;
            break 'start;
        }
        // Miller algorithm for |z| > r1
        let mut ak = e::cos(pi * dnu).abs();
        if ak == 0.0 {
            s1 = coef;
            s2 = coef;
            label = L::L100;
            break 'start;
        }
        let mut fhs = (0.25 - dnu2).abs();
        if fhs == 0.0 {
            s1 = coef;
            s2 = coef;
            label = L::L100;
            break 'start;
        }
        // log(10)/log(2), a constant the compiler folds to the correctly rounded quotient of the
        // correctly rounded logarithms
        let mut t1 =
            (I1MACH[13] - 1.0) * D1MACH[4] * (std::f64::consts::LN_10 / std::f64::consts::LN_2);
        t1 = fmin(fmax(t1, 12.0), 60.0);
        let t2 = tth * t1 - 6.0;
        t1 = e::atan2(yy, xx).abs();
        let mut fk;
        let mut to_l80 = false;
        if t2 <= caz {
            // forward recurrence loop when |z| >= r2
            let etest = ak / (pi * caz * tol);
            fk = 1.0;
            if etest < 1.0 {
                to_l80 = true;
            } else {
                let mut fks = 2.0;
                let mut rk = caz + caz + 2.0;
                let mut a1 = 0.0;
                let mut a2 = 1.0;
                for i in 1..(kmax + 1) {
                    ak = fhs / fks;
                    let bk = rk / (fk + 1.0);
                    let tm = a2;
                    a2 = bk * a2 - ak * a1;
                    a1 = tm;
                    rk += 2.0;
                    fks += fk + fk + 2.0;
                    fhs += fk + fk;
                    fk += 1.0;
                    let tm = a2.abs() * fk;
                    if etest < tm {
                        break;
                    }
                    if i == kmax {
                        return -2;
                    }
                }
                fk += spi * t1 * (t2 / caz).sqrt();
                fhs = (0.25 - dnu2).abs();
            }
        } else {
            // backward index k for |z| < r2
            let a2 = caz.sqrt();
            ak *= fpi / (tol * a2.sqrt());
            let aa = 3.0 * t1 / (1.0 + caz);
            let bb = 14.7 * t1 / (28.0 + caz);
            ak = (e::log(ak) + caz * e::cos(aa) / (1.0 + 0.008 * caz)) / e::cos(bb);
            fk = 0.12125 * ak * ak / caz + 1.5;
        }
        let _ = to_l80;
        // L80: backward recurrence loop for the Miller algorithm
        let k = c_int(fk);
        fk = k as f64;
        let mut fks = fk * fk;
        p1 = C64::ZERO;
        p2 = C64::real(tol);
        let mut cs = p2;
        for _ in 1..(k.wrapping_add(1)) {
            let a1 = fks - fk;
            let a2 = (fks + fk) / (a1 + fhs);
            let rk = 2.0 / (fk + 1.0);
            let t1 = (fk + xx) * rk;
            let t2 = yy * rk;
            let pt = p2;
            p2 = (p2 * C64::new(t1, t2) - p1) * a2;
            p1 = pt;
            cs += p2;
            fks = a1 - fk + 1.0;
            fk -= 1.0;
        }
        // (P2/CS) = (P2/|CS|) (CONJG(CS)/|CS|) for better scaling
        let tm = cs.abs();
        let mut pt = C64::real(1.0 / tm);
        s1 = pt * p2;
        cs = cs.conj() * pt;
        s1 *= coef * cs;
        if (inu <= 0) && (n <= 1) {
            zd = z;
            label = if iflag == 1 { L::L190 } else { L::L130 };
            break 'start;
        }
        // P1/P2 = (P1/|P2|) (CONJG(P2)/|P2|) for scaling
        let tm = p2.abs();
        pt = C64::real(1.0 / tm);
        p1 = pt * p1;
        p2 = p2.conj() * pt;
        pt = p1 * p2;
        s2 = s1 * (1.0 + (dnu + 0.5 - pt) / z);
        label = L::L100;
    }

    // Forward recursion on the three-term recursion with scaling near the exponent extremes.
    ck = (dnu + 1.0) * rz;
    let mut i_l160_exit = 0;
    loop {
        match label {
            L::L100 => {
                ck = (dnu + 1.0) * rz;
                if n == 1 {
                    inu -= 1;
                }
                if inu <= 0 {
                    if n <= 1 {
                        s1 = s2;
                    }
                    zd = z;
                    label = if iflag == 1 { L::L190 } else { L::L130 };
                    continue;
                }
                inub = 1;
                label = if iflag == 1 { L::L160 } else { L::L110 };
            }
            L::L110 => {
                p1 = csr[(kflag - 1) as usize];
                ascle = bry[(kflag - 1) as usize];
                for _ in inub..(inu + 1) {
                    let st = s2;
                    s2 = ck * s2 + s1;
                    s1 = st;
                    ck += rz;
                    if kflag < 3 {
                        p2 = s2 * p1;
                        let p2m = fmax(p2.re.abs(), p2.im.abs());
                        if p2m > ascle {
                            kflag += 1;
                            ascle = bry[(kflag - 1) as usize];
                            s1 *= p1;
                            s2 = p2;
                            s1 *= css[(kflag - 1) as usize];
                            s2 *= css[(kflag - 1) as usize];
                            p1 = csr[(kflag - 1) as usize];
                        }
                    }
                }
                if n == 1 {
                    s1 = s2;
                }
                label = L::L130;
            }
            L::L130 => {
                y[0] = s1 * csr[(kflag - 1) as usize];
                if n == 1 {
                    return nz;
                }
                y[1] = s2 * csr[(kflag - 1) as usize];
                if n == 2 {
                    return nz;
                }
                kk = 2;
                label = L::L140;
            }
            L::L140 => {
                kk += 1;
                if kk > n {
                    return nz;
                }
                p1 = csr[(kflag - 1) as usize];
                ascle = bry[(kflag - 1) as usize];
                for i in kk..(n + 1) {
                    p2 = s2;
                    s2 = ck * s2 + s1;
                    s1 = p2;
                    ck += rz;
                    p2 = s2 * p1;
                    y[(i - 1) as usize] = p2;
                    if kflag < 3 {
                        let p2m = fmax(p2.re.abs(), p2.im.abs());
                        if p2m > ascle {
                            kflag += 1;
                            ascle = bry[(kflag - 1) as usize];
                            s1 *= p1;
                            s2 = p2;
                            s1 *= css[(kflag - 1) as usize];
                            s2 *= css[(kflag - 1) as usize];
                            p1 = csr[(kflag - 1) as usize];
                        }
                    }
                }
                return nz;
            }
            L::L160 => {
                // IFLAG=1: forward recurrence on scaled values on underflow
                let elm = e::exp(-elim);
                ascle = bry[0];
                zd = z;
                let mut xd = xx;
                let yd = yy;
                let mut ic = -1;
                let mut j = 2;
                let mut to_l180 = false;
                let mut i = 1;
                while i < inu + 1 {
                    let st = s2;
                    s2 = ck * s2 + s1;
                    s1 = st;
                    ck += rz;
                    let as_ = s2.abs();
                    let alas = e::log(as_);
                    let p2r = alas - xd;
                    if p2r >= -elim {
                        let p2 = -zd + s2.ln();
                        let p2r = p2.re;
                        let p2i = p2.im;
                        let p2m = e::exp(p2r) / tol;
                        let p1 = p2m * C64::new(e::cos(p2i), e::sin(p2i));
                        if !uchk(p1, ascle, tol) {
                            j = 3 - j;
                            cy[(j - 1) as usize] = p1;
                            if ic == i - 1 {
                                to_l180 = true;
                                break;
                            }
                            ic = i;
                            i += 1;
                            continue;
                        }
                    }
                    if alas >= 0.5 * elim {
                        xd -= elim;
                        s1 *= elm;
                        s2 *= elm;
                        zd = C64::new(xd, yd);
                    }
                    i += 1;
                }
                if to_l180 {
                    // L180
                    kflag = 1;
                    inub = i + 1;
                    s2 = cy[(j - 1) as usize];
                    j = 3 - j;
                    s1 = cy[(j - 1) as usize];
                    if inub <= inu {
                        label = L::L110;
                        continue;
                    }
                    if n == 1 {
                        s1 = s2;
                    }
                    label = L::L130;
                    continue;
                }
                if n == 1 {
                    s1 = s2;
                }
                i_l160_exit = i;
                label = L::L190;
            }
            L::L190 => {
                let _ = i_l160_exit;
                y[0] = s1;
                if n != 1 {
                    y[1] = s2;
                }
                ascle = bry[0];
                nz = kscl(zd, fnu, n, y, rz, &mut ascle, tol, elim);
                inu = n - nz;
                if inu <= 0 {
                    return nz;
                }
                kk = nz + 1;
                s1 = y[(kk - 1) as usize];
                y[(kk - 1) as usize] = s1 * csr[0];
                if inu == 1 {
                    return nz;
                }
                kk = nz + 2;
                s2 = y[(kk - 1) as usize];
                y[(kk - 1) as usize] = s2 * csr[0];
                if inu == 2 {
                    return nz;
                }
                let t2 = fnu + (kk - 1) as f64;
                ck = t2 * rz;
                kflag = 1;
                label = L::L140;
            }
        }
    }
}

/// `rati`: ratios of I functions by backward recurrence.
fn rati(z: C64, fnu: f64, n: i32, cy: &mut [C64], tol: f64) {
    let az = z.abs();
    let inu = c_int(fnu);
    let idnu = inu.wrapping_add(n - 1);
    let fdnu = idnu as f64;
    let magz = c_int(az);
    let amagz = magz.wrapping_add(1) as f64;
    let fnup = fmax(amagz, fdnu);
    let mut id = idnu.wrapping_sub(magz).wrapping_sub(1);
    let mut itime = 1;
    let mut k = 1;
    let rz = 2.0 / z;
    let mut t1 = fnup * rz;
    let mut p2 = -t1;
    let mut p1 = C64::real(1.0);
    t1 += rz;
    if id > 0 {
        id = 0;
    }
    let mut ap2 = p2.abs();
    let mut ap1 = p1.abs();
    let arg = (ap2 + ap2) / (ap1 * tol);
    let test1 = arg.sqrt();
    let mut test = test1;
    let rap1 = 1.0 / ap1;
    p1 *= rap1;
    p2 *= rap1;
    ap2 *= rap1;
    loop {
        k += 1;
        ap1 = ap2;
        let pt = p2;
        p2 = p1 - t1 * p2;
        p1 = pt;
        t1 += rz;
        ap2 = p2.abs();
        if ap1 > test {
            break;
        }
        if itime != 2 {
            let ak = t1.abs() * 0.5;
            let flam = ak + (ak * ak - 1.0).sqrt();
            let rho = fmin(ap2 / ap1, flam);
            test = test1 * (rho / (rho * rho - 1.0)).sqrt();
            itime = 2;
        }
    }
    let kk = k + 1 - id;
    let ak = kk as f64;
    let dfnu = fnu + n as f64 - 1.0;
    let mut cdfnu = C64::real(dfnu);
    t1 = C64::real(ak);
    p1 = C64::real(1.0 / ap2);
    p2 = C64::ZERO;
    for _ in 1..(kk + 1) {
        let pt = p1;
        p1 = rz * (cdfnu + t1) * p1 + p2;
        p2 = pt;
        t1 -= 1.0;
    }
    if p1.eq_real(0.0) {
        p1 = C64::new(tol, tol);
    }
    cy[(n - 1) as usize] = p2 / p1;
    if n == 1 {
        return;
    }
    let mut k = n - 1;
    let ak = k as f64;
    t1 = C64::real(ak);
    cdfnu = fnu * rz;
    for _ in 2..(n + 1) {
        let mut pt = cdfnu + t1 * rz + cy[k as usize];
        if pt.eq_real(0.0) {
            pt = C64::new(tol, tol);
        }
        cy[(k - 1) as usize] = 1.0 / pt;
        t1 -= 1.0;
        k -= 1;
    }
}

/// `wrsk`: I(fnu, z) by normalizing `rati`'s ratios with the Wronskian.
#[allow(clippy::too_many_arguments)]
fn wrsk(
    zr: C64,
    fnu: f64,
    kode: i32,
    n: i32,
    y: &mut [C64],
    cw: &mut [C64],
    tol: f64,
    elim: f64,
    alim: f64,
) -> i32 {
    let nz = 0;
    let nw = bknu(zr, fnu, kode, 2, cw, tol, elim, alim);
    if nw != 0 {
        return if nw == -2 { -2 } else { -1 };
    }
    rati(zr, fnu, n, y, tol);
    let mut cinu = C64::real(1.0);
    if kode != 1 {
        let yy = zr.im;
        cinu = C64::new(e::cos(yy), e::sin(yy));
    }
    let acw = cw[1].abs();
    let mut ascle = 1e3 * D1MACH[0] / tol;
    let mut cscl = C64::real(1.0);
    if acw <= ascle {
        cscl = C64::real(1.0 / tol);
    } else {
        ascle = 1.0 / ascle;
        if acw >= ascle {
            cscl = C64::real(tol);
        }
    }
    let c1 = cw[0] * cscl;
    let c2 = cw[1] * cscl;
    let mut st = y[0];
    let mut ct = zr * (c2 + st * c1);
    let act = ct.abs();
    let rct = C64::real(1.0 / act);
    ct = ct.conj() * rct;
    cinu *= ct * rct;
    y[0] = cinu * cscl;
    if n == 1 {
        return nz;
    }
    for i in 2..(n + 1) {
        cinu *= st;
        st = y[(i - 1) as usize];
        y[(i - 1) as usize] = cinu * cscl;
    }
    nz
}

/// `unhj`: the parameters of the uniform asymptotic expansion for large orders (Olver).
/// Returns `(phi, arg, zeta1, zeta2, asum, bsum)`; `ipmtr = 1` leaves `asum` and `bsum` zero.
fn unhj(z: C64, fnu: f64, ipmtr: i32, tol: f64) -> (C64, C64, C64, C64, C64, C64) {
    let ex1 = 1.0 / 3.0;
    let ex2 = 2.0 / 3.0;
    let hpi = 1.57079632679489662;
    let pi = 3.14159265358979324;
    let thpi = 4.71238898038468986;
    let mut cr = [C64::ZERO; 14];
    let mut dr = [C64::ZERO; 14];
    let mut up = [C64::ZERO; 14];
    let mut p = [C64::ZERO; 30];
    let mut ap = [0.0f64; 30];
    let (mut phi, mut arg, mut zeta1, mut zeta2) = (C64::ZERO, C64::ZERO, C64::ZERO, C64::ZERO);
    let (mut asum, mut bsum) = (C64::ZERO, C64::ZERO);

    let rfnu = 1.0 / fnu;
    // overflow test (z/fnu too small)
    let test = D1MACH[0] * 1e3;
    let ac = fnu * test;
    if (z.re.abs() <= ac) && (z.im.abs() <= ac) {
        zeta1 = C64::real(2.0 * e::log(test).abs() + fnu);
        zeta2 = C64::real(fnu);
        phi = C64::real(1.0);
        arg = C64::real(1.0);
        return (phi, arg, zeta1, zeta2, asum, bsum);
    }
    let zb = z * rfnu;
    let rfnu2 = rfnu * rfnu;
    // compute in the fourth quadrant
    let fn13 = e::pow(fnu, ex1);
    let fn23 = fn13 * fn13;
    let rfn13 = 1.0 / fn13;
    let mut w2 = 1.0 - zb * zb;
    if w2.im == -0.0 {
        w2 = C64::real(w2.re);
    }
    let aw2 = w2.abs();
    if aw2 <= 0.25 {
        // power series for |w2| <= 0.25
        let mut k = 1;
        p[0] = C64::real(1.0);
        let mut suma = C64::real(t::ZUNHJ_GAMA[0]);
        ap[0] = 1.0;
        if aw2 >= tol {
            k = 2;
            while k < 31 {
                p[k - 1] = p[k - 2] * w2;
                suma += p[k - 1] * t::ZUNHJ_GAMA[k - 1];
                ap[k - 1] = ap[k - 2] * aw2;
                if ap[k - 1] < tol {
                    break;
                }
                k += 1;
            }
        }
        if k == 31 {
            k = 30;
        }
        let kmax = k;
        let zeta = w2 * suma;
        arg = zeta * fn23;
        let mut za = suma.sqrt();
        zeta2 = w2.sqrt() * fnu;
        zeta1 = zeta2 * (1.0 + zeta * za * ex2);
        za = za + za;
        phi = za.sqrt() * rfn13;
        if ipmtr == 1 {
            return (phi, arg, zeta1, zeta2, asum, bsum);
        }
        // sum series for asum and bsum
        let mut sumb = C64::ZERO;
        for k in 1..(kmax + 1) {
            sumb += p[k - 1] * t::ZUNHJ_BETA[k - 1];
        }
        asum = C64::ZERO;
        bsum = sumb;
        let mut l1 = 0;
        let mut l2 = 30;
        let btol = tol * (bsum.re.abs() + bsum.im.abs());
        let mut atol = tol;
        let mut pp = 1.0;
        let mut ias = 0;
        let mut ibs = 0;
        if rfnu2 < tol {
            asum += 1.0;
            bsum *= rfnu * rfn13;
            return (phi, arg, zeta1, zeta2, asum, bsum);
        }
        for _is in 2..8 {
            atol /= rfnu2;
            pp *= rfnu2;
            if ias != 1 {
                suma = C64::ZERO;
                for k in 1..(kmax + 1) {
                    let m = l1 + k;
                    suma += p[k - 1] * t::ZUNHJ_ALFA[m - 1];
                    if ap[k - 1] < atol {
                        break;
                    }
                }
                asum += suma * pp;
                if pp < tol {
                    ias = 1;
                }
            }
            if ibs != 1 {
                sumb = C64::ZERO;
                for k in 1..(kmax + 1) {
                    let m = l2 + k;
                    sumb += p[k - 1] * t::ZUNHJ_BETA[m - 1];
                    if ap[k - 1] < atol {
                        break;
                    }
                }
                bsum += sumb * pp;
                if pp < btol {
                    ibs = 1;
                }
            }
            if (ias == 1) && (ibs == 1) {
                break;
            }
            l1 += 30;
            l2 += 30;
        }
        asum += 1.0;
        bsum *= rfnu * rfn13;
        (phi, arg, zeta1, zeta2, asum, bsum)
    } else {
        // |w2| > 0.25
        let w = w2.sqrt();
        let mut wr = w.re;
        let mut wi = w.im;
        if wr < 0.0 {
            wr = 0.0;
        }
        if wi < 0.0 {
            wi = 0.0;
        }
        let _ = (wr, wi);
        let mut za = (1.0 + w) / zb;
        let mut zc = za.ln();
        let mut zcr = zc.re;
        let mut zci = zc.im;
        if zci < 0.0 {
            zci = 0.0;
        }
        if zci > hpi {
            zci = hpi;
        }
        if zcr < 0.0 {
            zcr = 0.0;
        }
        zc = C64::new(zcr, zci);
        let zth = (zc - w) * 1.5;
        let cfnu = C64::real(fnu);
        zeta1 = zc * cfnu;
        zeta2 = w * cfnu;
        let azth = zth.abs();
        let zthr = zth.re;
        let zthi = zth.im;
        let mut ang = thpi;
        if (zthr < 0.0) || (zthi >= 0.0) {
            ang = hpi;
            if zthr != 0.0 {
                ang = e::atan(zthi / zthr);
                if zthr < 0.0 {
                    ang += pi;
                }
            }
        }
        let mut pp = e::pow(azth, ex2);
        ang *= ex2;
        let zetar = pp * e::cos(ang);
        let mut zetai = pp * e::sin(ang);
        if zetai < 0.0 {
            zetai = 0.0;
        }
        let zeta = C64::new(zetar, zetai);
        arg = zeta * fn23;
        let rtzta = zth / zeta;
        za = rtzta / w;
        phi = (za + za).sqrt() * rfn13;
        if ipmtr == 1 {
            return (phi, arg, zeta1, zeta2, asum, bsum);
        }
        let tfn = rfnu / w;
        let rzth = rfnu / zth;
        zc = rzth * t::ZUNHJ_AR[1];
        let t2 = 1.0 / w2;
        up[1] = (t2 * t::ZUNHJ_C[1] + t::ZUNHJ_C[2]) * tfn;
        bsum = up[1] + zc;
        asum = C64::ZERO;
        if rfnu < tol {
            asum += 1.0;
            bsum *= -rfn13 / rtzta;
            return (phi, arg, zeta1, zeta2, asum, bsum);
        }
        let mut przth = rzth;
        let mut ptfn = tfn;
        up[0] = C64::real(1.0);
        pp = 1.0;
        let btol = tol * (bsum.re.abs() + bsum.im.abs());
        let _ = btol;
        let mut ks = 0usize;
        let mut kp1 = 2usize;
        let mut l = 3usize;
        let mut ias = 0;
        let mut ibs = 0;
        let mut lr = 2usize;
        while lr < 13 {
            let lrp1 = lr + 1;
            // two more CR, DR and UP for two more terms in the next SUMA and SUMB
            for _k in lr..(lrp1 + 1) {
                ks += 1;
                kp1 += 1;
                l += 1;
                let mut za = C64::real(t::ZUNHJ_C[l - 1]);
                for _j in 2..(kp1 + 1) {
                    l += 1;
                    za = za * t2 + t::ZUNHJ_C[l - 1];
                }
                ptfn *= tfn;
                up[kp1 - 1] = ptfn * za;
                cr[ks - 1] = przth * t::ZUNHJ_BR[ks];
                przth *= rzth;
                dr[ks - 1] = przth * t::ZUNHJ_AR[ks + 1];
            }
            pp *= rfnu2;
            if ias != 1 {
                let mut suma = up[lr];
                let mut ju = lrp1;
                for jr in 1..lrp1 {
                    ju -= 1;
                    suma += cr[jr - 1] * up[ju - 1];
                }
                asum += suma;
                let test = asum.re.abs() + asum.im.abs();
                if (pp < tol) && (test < tol) {
                    ias = 1;
                }
            }
            if ibs != 1 {
                let mut sumb = up[lr + 1] + up[lr] * zc;
                let mut ju = lrp1;
                for jr in 1..lrp1 {
                    ju -= 1;
                    sumb += dr[jr - 1] * up[ju - 1];
                }
                bsum += sumb;
                let test = bsum.re.abs() + bsum.im.abs();
                if (pp < tol) && (test < tol) {
                    ibs = 1;
                }
            }
            if (ias == 1) && (ibs == 1) {
                break;
            }
            lr += 2;
        }
        asum += 1.0;
        bsum *= -rfn13 / rtzta;
        (phi, arg, zeta1, zeta2, asum, bsum)
    }
}

/// `uoik`: the leading terms of the uniform expansions, tested against `alim`/`elim` for
/// overflow and underflow.
#[allow(clippy::too_many_arguments)]
fn uoik(
    z: C64,
    fnu: f64,
    kode: i32,
    ikflg: i32,
    n: i32,
    y: &mut [C64],
    tol: f64,
    elim: f64,
    alim: f64,
) -> i32 {
    let aic = 1.265512123484645396;
    let mut nuf = 0;
    let mut nn = n;
    let x = z.re;
    let mut zr = z;
    if x < 0.0 {
        zr = -z;
    }
    let zb = zr;
    let yy = zr.im;
    let ax = x.abs() * 3.0f64.sqrt();
    let ay = yy.abs();
    let iform = if ay > ax { 2 } else { 1 };
    if iform != 2 {
        // `unik` (the expansion for I and K in |arg z| <= pi/3): off the imaginary axis only.
        return -1;
    }
    let mut gnu = fmax(fnu, 1.0);
    if ikflg != 1 {
        let fnn = nn as f64;
        let gnn = fnu + fnn - 1.0;
        gnu = fmax(gnn, fnn);
    }
    let mut zn = -zr * C64::new(0.0, 1.0);
    if yy <= 0.0 {
        zn.re = -zn.re;
    }
    let (mut phi, mut arg, mut zeta1, mut zeta2, _, _) = unhj(zn, gnu, 1, tol);
    let mut cz = zeta2 - zeta1;
    let mut aarg = arg.abs();
    if kode == 2 {
        cz -= zb;
    }
    if ikflg == 2 {
        cz = -cz;
    }
    let mut aphi = phi.abs();
    let mut rcz = cz.re;
    // overflow test
    if rcz > elim {
        return -1;
    }
    if rcz >= alim {
        rcz += e::log(aphi);
        rcz -= 0.25 * e::log(aarg) + aic;
        if rcz > elim {
            return -1;
        }
    } else {
        // underflow test
        if rcz >= -elim {
            if rcz > -alim {
                // pass
            } else {
                rcz += e::log(aphi);
                rcz -= 0.25 * e::log(aarg) + aic;
                if rcz > -elim {
                    let ascle = 1e3 * D1MACH[0] / tol;
                    cz += phi.ln();
                    cz -= 0.25 * arg.ln() + aic;
                    let ax = e::exp(rcz) / tol;
                    let ay = cz.im;
                    cz = C64::real(ax * e::exp(ay));
                    if uchk(cz, ascle, tol) {
                        for v in y.iter_mut().take(nn as usize) {
                            *v = C64::ZERO;
                        }
                        return nn;
                    }
                } else {
                    for v in y.iter_mut().take(nn as usize) {
                        *v = C64::ZERO;
                    }
                    return nn;
                }
            }
        } else {
            for v in y.iter_mut().take(nn as usize) {
                *v = C64::ZERO;
            }
            return nn;
        }
    }
    if (ikflg == 2) || (n == 1) {
        return nuf;
    }
    loop {
        gnu = fnu + (nn - 1) as f64;
        (phi, arg, zeta1, zeta2, _, _) = unhj(zn, gnu, 1, tol);
        cz = zeta2 - zeta1;
        aarg = phi.abs();
        if kode == 2 {
            cz -= zb;
        }
        aphi = phi.abs();
        rcz = cz.re;
        if rcz >= -elim {
            if rcz > -alim {
                return nuf;
            }
            rcz += e::log(aphi);
            rcz -= 0.25 * e::log(aarg) + aic;
            if rcz > -elim {
                let ascle = 1e3 * D1MACH[0] / tol;
                cz += phi.ln();
                cz -= 0.25 * arg.ln() + aic;
                let ax = e::exp(rcz) / tol;
                let ay = cz.im;
                cz = ax * (e::cos(ay) + e::sin(ay) * C64::new(0.0, 1.0));
                if !uchk(cz, ascle, tol) {
                    return nuf;
                }
            }
        }
        y[(nn - 1) as usize] = C64::ZERO;
        nn -= 1;
        nuf += 1;
        if nn == 0 {
            return nuf;
        }
    }
}

/// `acai`: the analytic continuation of K to the left half plane, for `airy`.
#[allow(clippy::too_many_arguments)]
fn acai(
    z: C64,
    fnu: f64,
    kode: i32,
    mr: i32,
    n: i32,
    y: &mut [C64],
    rl: f64,
    tol: f64,
    elim: f64,
    alim: f64,
) -> i32 {
    let mut cy = [C64::ZERO; 2];
    let pi = 3.14159265358979324;
    let mut nz = 0;
    let zn = -z;
    let az = z.abs();
    let nn = n;
    let dfnu = fnu + (n - 1) as f64;
    if (az > 2.0) && (az * az * 0.25 > dfnu + 1.0) {
        let nw = if az >= rl {
            asyi(zn, fnu, kode, nn, y, rl, tol, elim, alim)
        } else {
            mlri(zn, fnu, kode, nn, y, tol)
        };
        if nw < 0 {
            return if nw == -2 { -2 } else { -1 };
        }
    } else {
        seri(zn, fnu, kode, nn, y, tol, elim, alim);
    }
    // analytic continuation to the left half plane for the K function
    let nw = bknu(zn, fnu, kode, 1, &mut cy, tol, elim, alim);
    if nw != 0 {
        return if nw == -2 { -2 } else { -1 };
    }
    let fmr = mr as f64;
    let sgn = if fmr < 0.0 { pi } else { -pi };
    let mut csgn = C64::new(0.0, sgn);
    if kode != 1 {
        let yy = -zn.im;
        let cpn = e::cos(yy);
        let spn = e::sin(yy);
        csgn *= C64::new(cpn, spn);
    }
    // CSPN = EXP(FNU*PI*I), computed to minimize losses of significance when fnu is large
    let inu = c_int(fnu);
    let arg = (fnu - inu as f64) * sgn;
    let cpn = e::cos(arg);
    let spn = e::sin(arg);
    let mut cspn = C64::new(cpn, spn);
    if inu % 2 == 1 {
        cspn = -cspn;
    }
    let mut c1 = cy[0];
    let mut c2 = y[0];
    if kode != 1 {
        let mut iuf = 0;
        let ascle = 1e3 * D1MACH[0] / tol;
        let nw = s1s2(zn, &mut c1, &mut c2, ascle, alim, &mut iuf);
        nz += nw;
    }
    y[0] = cspn * c1 + csgn * c2;
    nz
}

/// `airy`: Ai(z) (`id = 0`) or Ai'(z) (`id = 1`) of a complex argument; returns
/// `(value, nz, ierr)`.
fn airy(z: C64, id: i32, kode: i32) -> (C64, i32, i32) {
    let tth = 2.0 / 3.0;
    let c1 = 0.35502805388781723926;
    let c2 = 0.25881940379280679841;
    let coef = 0.18377629847393068317;
    let mut ierr = 0;
    let mut nz = 0;
    let mut ai = C64::ZERO;
    if !(0..=1).contains(&id) {
        ierr = 1;
    }
    if !(1..=2).contains(&kode) {
        ierr = 1;
    }
    if ierr != 0 {
        return (C64::ZERO, nz, ierr);
    }
    let az = z.abs();
    let tol = D1MACH[3];
    let fid = id as f64;
    if az <= 1.0 {
        // power series for |z| <= 1
        let mut s1 = C64::real(1.0);
        let mut s2 = C64::real(1.0);
        if az < tol {
            let mut aa = 1e3 * D1MACH[0];
            s1 = C64::ZERO;
            if id != 1 {
                if az > aa {
                    s1 = c2 * z;
                }
                ai = c1 - s1;
                return (ai, nz, ierr);
            }
            ai = C64::real(-c2);
            aa = aa.sqrt();
            if az > aa {
                s1 = z * z * 0.5;
            }
            ai += s1 * c1;
            return (ai, nz, ierr);
        }
        let aa = az * az;
        if aa >= tol / az {
            let mut trm1 = C64::real(1.0);
            let mut trm2 = C64::real(1.0);
            let mut atrm = 1.0;
            let z3 = z * z * z;
            let az3 = az * aa;
            let mut ak = 2.0 + fid;
            let mut bk = 3.0 - fid - fid;
            let ck = 4.0 - fid;
            let dk = 3.0 + fid + fid;
            let mut d1 = ak * dk;
            let mut d2 = bk * ck;
            let mut ad = if d1 > d2 { d2 } else { d1 };
            ak = 24.0 + 9.0 * fid;
            bk = 30.0 - 9.0 * fid;
            for _k in 1..26 {
                trm1 *= z3 / d1;
                s1 += trm1;
                trm2 *= z3 / d2;
                s2 += trm2;
                atrm *= az3 / ad;
                d1 += ak;
                d2 += bk;
                ad = if d1 > d2 { d2 } else { d1 };
                if atrm < tol * ad {
                    break;
                }
                ak += 18.0;
                bk += 18.0;
            }
        }
        if id != 1 {
            ai = s1 * c1 - z * s2 * c2;
            if kode == 1 {
                return (ai, nz, ierr);
            }
            let zta = z * z.sqrt() * tth;
            ai *= zta.exp();
            return (ai, nz, ierr);
        }
        ai = -s2 * c2;
        if az > tol {
            ai += z * z * s1 * c1 / (1.0 + fid);
        }
        if kode == 1 {
            return (ai, nz, ierr);
        }
        let zta = z * z.sqrt() * tth;
        return (ai * zta.exp(), nz, ierr);
    }
    // |z| > 1
    let fnu = (1.0 + fid) / 3.0;
    let k1 = c_int(I1MACH[14]);
    let k2 = c_int(I1MACH[15]);
    let r1m5 = D1MACH[4];
    let k = if k1.abs() > k2.abs() {
        k2.abs()
    } else {
        k1.abs()
    };
    let elim = 2.303 * (k as f64 * r1m5 - 3.0);
    let k1 = c_int(I1MACH[13] - 1.0);
    let mut aa = r1m5 * k1 as f64;
    let dig = if aa > 18.0 { 18.0 } else { aa };
    aa *= 2.303;
    let alim = elim + if -aa > -41.45 { -aa } else { -41.45 };
    let rl = 1.2 * dig + 3.0;
    let alaz = e::log(az);
    // test for range
    aa = 0.5 / tol;
    let bb = I1MACH[8] * 0.5;
    aa = if aa > bb { bb } else { aa };
    aa = e::pow(aa, tth);
    if az > aa {
        return (ai, 0, 4);
    }
    aa = aa.sqrt();
    if az > aa {
        ierr = 3;
    }
    let csq = z.sqrt();
    let mut zta = z * csq * tth;
    // RE(ZTA) <= 0 when RE(Z) < 0, especially when IM(Z) is small
    let mut iflag = 0;
    let mut sfac = 1.0;
    let zi = z.im;
    let zr = z.re;
    let ak = zta.im;
    if zr < 0.0 {
        let bk = zta.re;
        let ck = -bk.abs();
        zta = C64::new(ck, ak);
    }
    if (zi == 0.0) && (zr <= 0.0) {
        zta = C64::new(0.0, ak);
    }
    let mut aa = zta.re;
    let mut cy = [C64::ZERO; 2];
    if (aa < 0.0) || (zr <= 0.0) {
        if kode != 2 && aa <= -alim {
            // overflow test
            aa = -aa + 0.25 * alaz;
            iflag = 1;
            sfac = tol;
            if aa > elim {
                return (ai, 0, 2);
            }
        }
        // CBKNU and CACAI return EXP(ZTA)*K(FNU,ZTA) on KODE=2
        let mut mr = 1;
        if zi < 0.0 {
            mr = -1;
        }
        let nn = acai(zta, fnu, kode, mr, 1, &mut cy, rl, tol, elim, alim);
        if nn < 0 {
            if nn == -1 {
                return (C64::ZERO, 1, ierr);
            }
            return (C64::ZERO, 0, 5);
        }
        nz += nn;
    } else {
        if kode != 2 && aa >= alim {
            // underflow test
            aa = -aa - 0.25 * alaz;
            iflag = 2;
            sfac = 1.0 / tol;
            if aa < -elim {
                return (C64::ZERO, 1, ierr);
            }
        }
        nz = bknu(zta, fnu, kode, 1, &mut cy, tol, elim, alim);
    }
    let mut s1 = cy[0] * coef;
    if iflag == 0 {
        if id != 1 {
            return (csq * s1, nz, ierr);
        }
        return (-z * s1, nz, ierr);
    }
    s1 *= sfac;
    if id != 1 {
        s1 *= csq;
        return (s1 / sfac, nz, ierr);
    }
    s1 *= -z;
    (s1 / sfac, nz, ierr)
}

/// `uni2`: I(fnu, z) by the uniform expansion for J(fnu, zn), zn = z i or -z i in the right
/// half plane. Returns `(nz, nlast)`.
#[allow(clippy::too_many_arguments)]
fn uni2(
    z: C64,
    fnu: f64,
    kode: i32,
    n: i32,
    y: &mut [C64],
    fnul: f64,
    tol: f64,
    elim: f64,
    alim: f64,
) -> (i32, i32) {
    let hpi = 1.57079632679489662;
    let aic = 1.265512123484645396;
    let cip = [
        C64::real(1.0),
        C64::new(0.0, 1.0),
        C64::real(-1.0),
        -C64::new(0.0, 1.0),
    ];
    let ci = C64::new(0.0, 1.0);
    let cscl = 1.0 / tol;
    let crsc = tol;
    let csr = [C64::real(crsc), C64::real(1.0), C64::real(cscl)];
    let css = [C64::real(cscl), C64::real(1.0), C64::real(crsc)];
    let mut bry = [1e3 * D1MACH[0] / tol, 0.0, 0.0];
    let mut cy = [C64::ZERO; 2];
    let yy = z.im;
    let mut nz = 0;
    let mut nd = n;
    let nlast = 0;
    // zn is in the right half plane after rotation by ci or -ci
    let mut zn = -z * ci;
    let mut zb = z;
    let mut cid = -ci;
    let inu = c_int(fnu);
    let ang = hpi * (fnu - inu as f64);
    let mut c2 = C64::new(e::cos(ang), e::sin(ang));
    let zar = c2;
    let mut in_ = inu.wrapping_add(n).wrapping_sub(1);
    in_ %= 4;
    c2 *= cip[in_ as usize];
    if yy <= 0.0 {
        zn = (-zn).conj();
        zb = zb.conj();
        cid = -cid;
        c2 = c2.conj();
    }
    // check for underflow and overflow on the first member
    let fn0 = fmax(fnu, 1.0);
    let (_phi, _arg, zeta1, zeta2, _asum, _bsum) = unhj(zn, fn0, 0, tol);
    let s1 = if kode != 1 {
        let cfn = C64::real(fnu);
        -zeta1 + cfn * (cfn / (zb + zeta2))
    } else {
        -zeta1 + zeta2
    };
    let rs1 = s1.re;
    if rs1.abs() > elim {
        if rs1 > 0.0 {
            return (-1, nlast);
        }
        for v in y.iter_mut().take(n as usize) {
            *v = C64::ZERO;
        }
        return (nz, nlast);
    }
    let mut iflag = 0;
    loop {
        // L10
        let nn = if nd > 2 { 2 } else { nd };
        let mut to_l50 = false;
        let mut rs1 = 0.0;
        for i in 1..(nn + 1) {
            let fn_ = fnu + (nd - i) as f64;
            let (phi, arg, zeta1, zeta2, asum, bsum) = unhj(zn, fn_, 0, tol);
            let mut s1 = if kode != 1 {
                let cfn = C64::real(fn_);
                let ay = yy.abs();
                -zeta1 + cfn * (cfn / (zb + zeta2)) + ay * C64::new(0.0, 1.0)
            } else {
                -zeta1 + zeta2
            };
            // test for underflow and overflow
            rs1 = s1.re;
            if rs1.abs() > elim {
                to_l50 = true;
                break;
            }
            if i == 1 {
                iflag = 2;
            }
            if rs1.abs() >= alim {
                // refine test and scale
                let aphi = phi.abs();
                let aarg = arg.abs();
                rs1 += e::log(aphi) - 0.25 * e::log(aarg) - aic;
                if rs1.abs() > elim {
                    to_l50 = true;
                    break;
                }
                if i == 1 {
                    iflag = 1;
                }
                if rs1 >= 0.0 && i == 1 {
                    iflag = 3;
                }
            }
            // scale s1 to keep intermediate arithmetic on scale near exponent extremes
            let (ai, _nai, _) = airy(arg, 0, 2);
            let (dai, _ndai, _) = airy(arg, 1, 2);
            let mut s2 = phi * (ai * asum + dai * bsum);
            let c2r = e::exp(s1.re) * css[(iflag - 1) as usize].re;
            let c2i = s1.im;
            s1 = c2r * C64::new(e::cos(c2i), e::sin(c2i));
            s2 *= s1;
            if iflag == 1 && uchk(s2, bry[0], tol) {
                to_l50 = true;
                break;
            }
            if yy <= 0.0 {
                s2 = s2.conj();
            }
            let j = nd - i + 1;
            s2 *= c2;
            cy[(i - 1) as usize] = s2;
            y[(j - 1) as usize] = s2 * csr[(iflag - 1) as usize];
            c2 *= cid;
        }
        if !to_l50 {
            if nd > 2 {
                let rz = 2.0 / z;
                bry[1] = 1.0 / bry[0];
                bry[2] = D1MACH[1];
                let mut s1 = cy[0];
                let mut s2 = cy[1];
                let mut c1 = csr[(iflag - 1) as usize];
                let mut ascle = bry[(iflag - 1) as usize];
                let mut k = nd - 2;
                let mut fn_ = k as f64;
                for _i in 3..(nd + 1) {
                    c2 = s2;
                    s2 = s1 + (fnu + fn_) * rz * s2;
                    s1 = c2;
                    c2 = s2 * c1;
                    y[(k - 1) as usize] = c2;
                    k -= 1;
                    fn_ -= 1.0;
                    if iflag < 3 {
                        let c2r = c2.re.abs();
                        let c2i = c2.im.abs();
                        let c2m = fmax(c2r, c2i);
                        if c2m > ascle {
                            iflag += 1;
                            ascle = bry[(iflag - 1) as usize];
                            s1 *= c1;
                            s2 = c2;
                            s1 *= css[(iflag - 1) as usize];
                            s2 *= css[(iflag - 1) as usize];
                            c1 = csr[(iflag - 1) as usize];
                        }
                    }
                }
            }
            return (nz, nlast);
        }
        // L50
        if rs1 <= 0.0 {
            // set underflow and update parameters
            y[(nd - 1) as usize] = C64::ZERO;
            nz += 1;
            nd -= 1;
            if nd == 0 {
                return (nz, nlast);
            }
            let nuf = uoik(z, fnu, kode, 1, nd, y, tol, elim, alim);
            if nuf >= 0 {
                nd -= nuf;
                nz += nuf;
                if nd == 0 {
                    return (nz, nlast);
                }
                let fn_ = fnu + nd as f64 - 1.0;
                if fn_ >= fnul {
                    let in_ = inu.wrapping_add(nd).wrapping_sub(1) % 4;
                    c2 = zar * cip[in_ as usize];
                    if yy <= 0.0 {
                        c2 = c2.conj();
                    }
                    continue;
                }
                return (nz, nd);
            }
        }
        return (-1, nlast);
    }
}

/// `buni`: I(fnu, z) for large |z| > fnul by the uniform expansion, recurring backward from an
/// order raised past fnul. Returns `(nz, nlast)`.
#[allow(clippy::too_many_arguments)]
fn buni(
    z: C64,
    fnu: f64,
    kode: i32,
    n: i32,
    y: &mut [C64],
    nui: i32,
    fnul: f64,
    tol: f64,
    elim: f64,
    alim: f64,
) -> (i32, i32) {
    let mut cy = [C64::ZERO; 2];
    let mut bry = [0.0f64; 3];
    let nz = 0;
    let xx = z.re;
    let yy = z.im;
    let mut ax = xx.abs() * 3.0f64.sqrt();
    let mut ay = yy.abs();
    let iform = if ay > ax { 2 } else { 1 };
    if iform != 2 {
        // `uni1` (the expansion for I in |arg z| <= pi/3): off the imaginary axis only.
        return (-1, 0);
    }
    if nui == 0 {
        let (nw, nlast) = uni2(z, fnu, kode, n, y, fnul, tol, elim, alim);
        if nw < 0 {
            return (if nw == -2 { -2 } else { -1 }, nlast);
        }
        return (nw, nlast);
    }
    let mut fnui = nui as f64;
    let dfnu = fnu + (n - 1) as f64;
    let gnu = dfnu + fnui;
    let (nw, nlast) = uni2(z, gnu, kode, 2, &mut cy, fnul, tol, elim, alim);
    if nw >= 0 {
        if nw != 0 {
            return (nz, n);
        }
        ay = cy[0].abs();
        // scale backward recurrence, bry[2] is defined but never used
        bry[0] = 1e3 * D1MACH[0] / tol;
        bry[1] = tol / 1e3 * D1MACH[0];
        bry[2] = bry[1];
        let mut iflag = 2;
        let mut ascle = bry[1];
        ax = 1.0;
        let mut cscl = C64::real(ax);
        if ay <= bry[0] {
            iflag = 1;
            ascle = bry[0];
            ax = 1.0 / tol;
            cscl = C64::real(ax);
        } else if ay >= bry[1] {
            iflag = 3;
            ascle = bry[2];
            ax = tol;
            cscl = C64::real(ax);
        }
        ay = 1.0 / ax;
        let mut cscr = C64::real(ay);
        let mut s1 = cy[1] * cscl;
        let mut s2 = cy[0] * cscl;
        let rz = 2.0 / z;
        let rescale = |iflag: &mut i32,
                       ascle: &mut f64,
                       s1: &mut C64,
                       s2: &mut C64,
                       st: C64,
                       ax: &mut f64,
                       cscl: &mut C64,
                       cscr: &mut C64| {
            *iflag += 1;
            *ascle = bry[(*iflag - 1) as usize];
            *s1 *= *cscr;
            *s2 = st;
            *ax *= tol;
            let ay = 1.0 / *ax;
            *cscl = C64::real(*ax);
            *cscr = C64::real(ay);
            *s1 *= *cscl;
            *s2 *= *cscl;
        };
        for _i in 1..(nui + 1) {
            let mut st = s2;
            s2 = (dfnu + fnui) * rz * st + s1;
            s1 = st;
            fnui -= 1.0;
            if iflag < 3 {
                st = s2 * cscr;
                let str_ = st.re.abs();
                let sti = st.im.abs();
                let stm = fmax(str_, sti);
                if stm > ascle {
                    rescale(
                        &mut iflag, &mut ascle, &mut s1, &mut s2, st, &mut ax, &mut cscl, &mut cscr,
                    );
                }
            }
        }
        y[(n - 1) as usize] = s2 * cscr;
        if n == 1 {
            return (nz, nlast);
        }
        let nl = n - 1;
        fnui = nl as f64;
        let mut k = nl;
        for _i in 0..(nl + 1) {
            let mut st = s2;
            s2 = (fnu + fnui) * rz * s2 + s1;
            s1 = st;
            st = s2 * cscr;
            y[(k - 1) as usize] = st;
            fnui -= 1.0;
            k -= 1;
            if iflag < 3 {
                st = s2 * cscr;
                let str_ = st.re.abs();
                let sti = st.im.abs();
                let stm = fmax(str_, sti);
                if stm > ascle {
                    rescale(
                        &mut iflag, &mut ascle, &mut s1, &mut s2, st, &mut ax, &mut cscl, &mut cscr,
                    );
                }
            }
        }
        return (nz, nlast);
    }
    (if nw == -2 { -2 } else { -1 }, nlast)
}

/// `unk2`: K(fnu, z) by the uniform expansion for H(2, fnu, zn), here only for `mr = 0` (the
/// right half plane; the continuation to the left one is off the imaginary axis only).
#[allow(clippy::too_many_arguments)]
fn unk2(
    z: C64,
    fnu: f64,
    kode: i32,
    mr: i32,
    n: i32,
    y: &mut [C64],
    tol: f64,
    elim: f64,
    alim: f64,
) -> i32 {
    if mr != 0 {
        return -1;
    }
    let cr1 = C64::new(1.0, 1.73205080756887729);
    let cr2 = C64::new(-0.5, -8.66025403784438647e-1);
    let hpi = 1.57079632679489662;
    let aic = 1.26551212348464539;
    let cip = [
        C64::real(1.0),
        -C64::new(0.0, 1.0),
        C64::real(-1.0),
        C64::new(0.0, 1.0),
    ];
    let cscl = 1.0 / tol;
    let crsc = tol;
    let css = [C64::real(cscl), C64::real(1.0), C64::real(crsc)];
    let csr = [C64::real(crsc), C64::real(1.0), C64::real(cscl)];
    let mut phi = [C64::ZERO; 2];
    let mut arg = [C64::ZERO; 2];
    let mut zeta1 = [C64::ZERO; 2];
    let mut zeta2 = [C64::ZERO; 2];
    let mut asum = [C64::ZERO; 2];
    let mut bsum = [C64::ZERO; 2];
    let mut cy = [C64::ZERO; 2];
    let bry = [
        1.0 + 1e3 * D1MACH[0] / tol,
        1.0 / (1.0 + 1e3 * D1MACH[0] / tol),
        D1MACH[1],
    ];
    let mut kdflg = 1;
    let mut kflag = 1;
    let mut fn_ = fnu;
    let mut nz = 0;
    // EXP(-ALIM)=EXP(-ELIM)/TOL: about one precision above the underflow limit
    let x = z.re;
    let mut zr = z;
    if x < 0.0 {
        zr = -z;
    }
    let yy = zr.im;
    let mut zn = -zr * C64::new(0.0, 1.0);
    let mut zb = zr;
    let inu = c_int(fnu);
    let fnf = fnu - inu as f64;
    let ang = -hpi * fnf;
    let car = e::cos(ang);
    let sar = e::sin(ang);
    let cpn = hpi * car;
    let spn = hpi * sar;
    let c2 = C64::new(spn, -cpn);
    let kk = (inu % 4) + 1;
    let mut cs = cr1 * c2 * cip[(kk - 1) as usize];
    if yy <= 0.0 {
        zn = (-zn).conj();
        zb = zb.conj();
    }
    // K(fnu, z) from H(2, fnu, -iz), z in the first quadrant; fourth quadrant values by
    // conjugation, since K is real on the positive real axis
    let mut j = 2;
    let mut i = 1;
    let minus_i = -C64::new(0.0, 1.0);
    let zero_member = |i: i32, y: &mut [C64], nz: &mut i32, cs: &mut C64| {
        y[(i - 1) as usize] = C64::ZERO;
        *cs *= minus_i;
        *nz += 1;
        if i != 1 && !y[(i - 2) as usize].eq_real(0.0) {
            y[(i - 2) as usize] = C64::ZERO;
            *nz += 1;
        }
    };
    while i < n + 1 {
        j = 3 - j;
        fn_ = fnu + (i - 1) as f64;
        let ju = (j - 1) as usize;
        (phi[ju], arg[ju], zeta1[ju], zeta2[ju], asum[ju], bsum[ju]) = unhj(zn, fn_, 0, tol);
        let mut s1 = if kode != 1 {
            let cfn = C64::real(fn_);
            zeta1[ju] - cfn * (cfn / (zb + zeta2[ju]))
        } else {
            zeta1[ju] - zeta2[ju]
        };
        // test for underflow and overflow
        let mut rs1 = s1.re;
        if rs1.abs() <= elim {
            if kdflg == 1 {
                kflag = 2;
            }
            let mut to_70 = false;
            if rs1.abs() >= alim {
                // refine test and scale
                let aphi = phi[ju].abs();
                let aarg = arg[ju].abs();
                rs1 += e::log(aphi) - 0.25 * e::log(aarg) - aic;
                if rs1.abs() > elim {
                    to_70 = true;
                } else {
                    if kdflg == 1 {
                        kflag = 1;
                    }
                    if rs1 >= 0.0 && kdflg == 1 {
                        kflag = 3;
                    }
                }
            }
            if !to_70 {
                // scale s1 to keep intermediate arithmetic on scale near exponent extremes
                let c2 = arg[ju] * cr2;
                let (ai, _nai, _) = airy(c2, 0, 2);
                let (dai, _ndai, _) = airy(c2, 1, 2);
                let mut s2 = cs * phi[ju] * (ai * asum[ju] + cr2 * dai * bsum[ju]);
                let c2r = s1.re;
                let c2i = s1.im;
                let c2m = e::exp(c2r) * css[(kflag - 1) as usize].re;
                s1 = c2m * C64::new(e::cos(c2i), e::sin(c2i));
                s2 *= s1;
                if kflag == 1 && uchk(s2, bry[0], tol) {
                    to_70 = true;
                } else {
                    if yy <= 0.0 {
                        s2 = s2.conj();
                    }
                    cy[(kdflg - 1) as usize] = s2;
                    y[(i - 1) as usize] = s2 * csr[(kflag - 1) as usize];
                    cs *= minus_i;
                    if kdflg == 2 {
                        break;
                    }
                    kdflg = 2;
                    i += 1;
                    continue;
                }
            }
            // GO TO 70 (from the refined test or the scaled underflow)
            if rs1 > 0.0 {
                return -1;
            }
            if x < 0.0 {
                return -1;
            }
            kdflg = 1;
            zero_member(i, y, &mut nz, &mut cs);
            i += 1;
            continue;
        }
        // GO TO 70
        if rs1 > 0.0 {
            return -1;
        }
        if x < 0.0 {
            return -1;
        }
        kdflg = 1;
        zero_member(i, y, &mut nz, &mut cs);
        i += 1;
    }
    if i == n + 1 {
        i = n;
    }
    let rz = 2.0 / zr;
    let mut ck = fn_ * rz;
    let ib = i + 1;
    if n >= ib {
        fn_ = fnu + (n - 1) as f64;
        let ipard = if mr != 0 { 0 } else { 1 };
        let (phid, argd, zeta1d, zeta2d, _asumd, _bsumd) = unhj(zn, fn_, ipard, tol);
        let s1 = if kode != 1 {
            let cfn = C64::real(fn_);
            zeta1d - cfn * (cfn / (zb + zeta2d))
        } else {
            zeta1d - zeta2d
        };
        let mut rs1 = s1.re;
        let mut to_l120 = false;
        if rs1.abs() <= elim {
            if rs1.abs() < alim {
                to_l120 = true;
            } else {
                // refine estimate and test
                let aphi = phid.abs();
                let aarg = argd.abs();
                rs1 += e::log(aphi) - 0.25 * e::log(aarg) - aic;
                if rs1.abs() < elim {
                    to_l120 = true;
                }
            }
        }
        if !to_l120 {
            if rs1 > 0.0 {
                return -1;
            }
            // for x < 0 the I function to be added would overflow
            if x < 0.0 {
                return -1;
            }
            nz = n;
            for v in y.iter_mut().take(n as usize) {
                *v = C64::ZERO;
            }
            return nz;
        }
        // L120: scaled forward recurrence for the remainder of the sequence
        let mut s1 = cy[0];
        let mut s2 = cy[1];
        let mut c1 = csr[(kflag - 1) as usize];
        let mut ascle = bry[(kflag - 1) as usize];
        for i in ib..(n + 1) {
            let c2 = s2;
            s2 = ck * s2 + s1;
            s1 = c2;
            ck += rz;
            let c2 = s2 * c1;
            y[(i - 1) as usize] = c2;
            if kflag < 3 {
                let c2m = fmax(c2.re.abs(), c2.im.abs());
                if c2m > ascle {
                    kflag += 1;
                    ascle = bry[(kflag - 1) as usize];
                    s1 *= c1;
                    s2 = c2;
                    s1 *= css[(kflag - 1) as usize];
                    s2 *= css[(kflag - 1) as usize];
                    c1 = csr[(kflag - 1) as usize];
                }
            }
        }
    }
    nz
}

/// `bunk`: K(fnu, z) for fnu > fnul by the uniform expansion.
#[allow(clippy::too_many_arguments)]
fn bunk(
    z: C64,
    fnu: f64,
    kode: i32,
    mr: i32,
    n: i32,
    y: &mut [C64],
    tol: f64,
    elim: f64,
    alim: f64,
) -> i32 {
    let ax = z.re.abs() * 1.7321;
    let ay = z.im.abs();
    if ay <= ax {
        // `unk1` (the expansion for K in |arg z| <= pi/3): off the imaginary axis only.
        return -1;
    }
    unk2(z, fnu, kode, mr, n, y, tol, elim, alim)
}

/// `binu`: I(fnu, z) in the right half plane.
#[allow(clippy::too_many_arguments)]
fn binu(
    z: C64,
    fnu: f64,
    kode: i32,
    n: i32,
    cy: &mut [C64],
    rl: f64,
    fnul: f64,
    tol: f64,
    elim: f64,
    alim: f64,
) -> i32 {
    let mut cw = [C64::ZERO; 2];
    let mut nz = 0;
    let az = z.abs();
    let mut nn = n;
    let mut dfnu = fnu + n as f64 - 1.0;
    if (az <= 2.0) || (az * az * 0.25 <= (dfnu + 1.0)) {
        let nw = seri(z, fnu, kode, nn, cy, tol, elim, alim);
        let inw = nw.abs();
        nz += inw;
        nn -= inw;
        if nn == 0 {
            return nz;
        }
        if nw >= 0 {
            return nz;
        }
        dfnu = fnu + nn as f64 - 1.0;
    }
    let miller_by_series = |nn: i32, cy: &mut [C64]| -> i32 {
        let nw = mlri(z, fnu, kode, nn, cy, tol);
        if nw < 0 {
            return if nw == -2 { -2 } else { -1 };
        }
        nz
    };
    if az < rl {
        if dfnu <= 1.0 {
            return miller_by_series(nn, cy);
        }
    } else if (dfnu <= 1.0) || (az + az >= dfnu * dfnu) {
        // asymptotic expansion for large z
        let nw = asyi(z, fnu, kode, nn, cy, rl, tol, elim, alim);
        if nw < 0 {
            return if nw == -2 { -2 } else { -1 };
        }
        return nz;
    }
    // L50: overflow and underflow test on the I sequence for the Miller algorithm
    let nw = uoik(z, fnu, kode, 1, nn, cy, tol, elim, alim);
    if nw < 0 {
        return if nw == -2 { -2 } else { -1 };
    }
    nz += nw;
    nn -= nw;
    if nn == 0 {
        return nz;
    }
    dfnu = fnu + (nn - 1) as f64;
    if (dfnu > fnul) || (az > fnul) {
        // increment fnu+nn-1 up to fnul, compute and recur backward
        let mut nui = c_int(fnul - dfnu).wrapping_add(1);
        nui = if nui > 0 { nui } else { 0 };
        let (nw, nlast) = buni(z, fnu, kode, nn, cy, nui, fnul, tol, elim, alim);
        if nw < 0 {
            return if nw == -2 { -2 } else { -1 };
        }
        nz += nw;
        if nlast == 0 {
            return nz;
        }
        nn = nlast;
    }
    if az <= rl {
        let nw = mlri(z, fnu, kode, nn, cy, tol);
        if nw < 0 {
            return if nw == -2 { -2 } else { -1 };
        }
        return nz;
    }
    // L80: Miller algorithm normalized by the Wronskian; overflow test on the K functions
    let nw = uoik(z, fnu, kode, 2, 2, &mut cw, tol, elim, alim);
    if nw < 0 {
        nz = nn;
        for v in cy.iter_mut().take(nn as usize) {
            *v = C64::ZERO;
        }
        return nz;
    }
    if nw > 0 {
        return -1;
    }
    let nw = wrsk(z, fnu, kode, nn, cy, &mut cw, tol, elim, alim);
    if nw < 0 {
        return if nw == -2 { -2 } else { -1 };
    }
    nz
}

/// `besj`: the J Bessel function of a complex argument and nonnegative order. Returns
/// `(nz, ierr)`.
pub(super) fn besj(z: C64, fnu: f64, kode: i32, n: i32, cy: &mut [C64]) -> (i32, i32) {
    let hpi = 1.570796326794896619;
    let mut ierr = 0;
    let mut nz = 0;
    if fnu < 0.0 || !(1..=2).contains(&kode) || n < 1 {
        return (nz, 1);
    }
    let (tol, elim, alim, _dig, fnul, rl) = machine();
    // test for proper range
    let yy = z.im;
    let az = z.abs();
    let fn_ = fnu + (n - 1) as f64;
    let mut aa = 0.5 / tol;
    let bb = D1MACH[1] * 0.5;
    aa = fmin(aa, bb);
    if (az > aa) || (fn_ > aa) {
        return (0, 4);
    }
    aa = aa.sqrt();
    if az > aa {
        ierr = 3;
    }
    if fn_ > aa {
        ierr = 3;
    }
    // CSGN = EXP(FNU*HPI*I), to minimize losses of significance when fnu is large
    let mut ci = C64::ZERO;
    ci.im = 1.0;
    let inu = c_int(fnu);
    let inuh = inu / 2;
    let ir = inu.wrapping_sub(inuh.wrapping_mul(2));
    let arg = (fnu - inu.wrapping_sub(ir) as f64) * hpi;
    let r1 = e::cos(arg);
    let r2 = e::sin(arg);
    let mut csgn = C64::new(r1, r2);
    if inuh % 2 == 1 {
        csgn = -csgn;
    }
    // zn is in the right half plane
    let mut zn = -z * ci;
    if yy < 0.0 {
        zn = -zn;
        csgn = csgn.conj();
        ci = ci.conj();
    }
    nz = binu(zn, fnu, kode, n, cy, rl, fnul, tol, elim, alim);
    if nz < 0 {
        if nz == -2 {
            return (0, 5);
        }
        return (0, 2);
    }
    let nl = n - nz;
    if nl == 0 {
        return (nz, ierr);
    }
    let rtol = 1.0 / tol;
    let ascle = D1MACH[0] * rtol * 1e3;
    for v in cy.iter_mut().take(nl as usize) {
        let mut zn = *v;
        let aa = zn.re;
        let bb = zn.im;
        let mut atol = 1.0;
        if fmax(aa.abs(), bb.abs()) <= ascle {
            zn *= rtol;
            atol = tol;
        }
        *v = atol * (zn * csgn);
        csgn = csgn * ci;
    }
    (nz, ierr)
}

/// `besh`: the Hankel function H(m, fnu, z). Returns `(nz, ierr)`.
fn besh(z: C64, fnu: f64, kode: i32, m: i32, n: i32, cy: &mut [C64]) -> (i32, i32) {
    let hpi = 1.57079632679489662;
    let mut nz = 0;
    let xx = z.re;
    let yy = z.im;
    let mut ierr = 0;
    if (xx == 0.0) && (yy == 0.0) {
        ierr = 1;
    }
    if fnu < 0.0 {
        ierr = 1;
    }
    if !(1..=2).contains(&m) {
        ierr = 1;
    }
    if !(1..=2).contains(&kode) {
        ierr = 1;
    }
    if n < 1 {
        ierr = 1;
    }
    if ierr != 0 {
        return (nz, ierr);
    }
    let mut nn = n;
    let (tol, elim, alim, _dig, fnul, _rl) = machine();
    let fn_ = fnu + (nn - 1) as f64;
    let mm = 3 - m - m;
    let fmm = mm as f64;
    let mut zn = z * C64::new(0.0, -fmm);
    let xn = zn.re;
    let yn = zn.im;
    // test for proper range
    let az = z.abs();
    let bb = D1MACH[1] * 0.5;
    let mut aa = fmin(0.5 / tol, bb);
    if (az > aa) || (fn_ > aa) {
        return (0, 4);
    }
    aa = aa.sqrt();
    if az > aa {
        ierr = 3;
    }
    if fn_ > aa {
        ierr = 3;
    }
    // overflow test on the last member of the sequence
    let ufl = D1MACH[0] * 1.0e3;
    if az < ufl {
        return (0, 2);
    }
    if fnu <= fnul {
        if (fn_ > 1.0) && (fn_ <= 2.0) && (az <= tol) {
            let arg = 0.5 * az;
            let aln = -fn_ * e::log(arg);
            if aln > elim {
                return (0, 2);
            }
        } else if fn_ > 2.0 {
            // the C++ tests `(fn > 1) && (fn > 2)`
            let nuf = uoik(zn, fnu, kode, 2, nn, cy, tol, elim, alim);
            if nuf < 0 {
                return (0, 2);
            }
            nz += nuf;
            nn -= nuf;
            // nn = n or nn = 0, since nuf = 0, nn or -1 on return from uoik
            if nn == 0 {
                if xn < 0.0 {
                    return (0, 2);
                }
                return (nz, ierr);
            }
        }
        if (xn < 0.0) || ((xn == 0.0) && (yn < 0.0) && (m == 2)) {
            // `acon` (the continuation to the left half plane): off the imaginary axis only.
            return (0, 2);
        }
        // right half plane computation
        nz = bknu(zn, fnu, kode, nn, cy, tol, elim, alim);
    } else {
        // uniform asymptotic expansions for fnu > fnul
        let mut mr = 0;
        if !((xn >= 0.0) && ((xn != 0.0) || (yn >= 0.0) || (m != 2))) {
            mr = -mm;
            if (xn == 0.0) && (yn < 0.0) {
                zn = -zn;
            }
        }
        let nw = bunk(zn, fnu, kode, mr, nn, cy, tol, elim, alim);
        if nw < 0 {
            if nw == -1 {
                return (0, 2);
            }
            return (0, 5);
        }
        nz += nw;
    }
    // H(M,FNU,Z) = -FMM*(I/HPI)*(ZT**FNU)*K(FNU,-Z*ZT), ZT = EXP(-FMM*HPI*I) = (0, -FMM)
    let sgn = if -fmm < 0.0 { -hpi } else { hpi };
    // EXP(FNU*HPI*I), computed to minimize losses of significance when fnu is large
    let inu = c_int(fnu);
    let inuh = inu / 2;
    let ir = inu.wrapping_sub(inuh.wrapping_mul(2));
    let arg = (fnu - inu.wrapping_sub(ir) as f64) * sgn;
    let rhpi = 1.0 / sgn;
    let cpn = rhpi * e::cos(arg);
    let spn = -rhpi * e::sin(arg);
    let mut csgn = C64::new(spn, cpn);
    if inuh % 2 == 1 {
        csgn = -csgn;
    }
    let zt = C64::new(0.0, -fmm);
    let rtol = 1.0 / tol;
    let ascle = ufl * rtol;
    for v in cy.iter_mut().take(nn as usize) {
        let mut zn = *v;
        let mut atol = 1.0;
        if fmax(zn.re.abs(), zn.im.abs()) <= ascle {
            zn *= rtol;
            atol = tol;
        }
        zn *= csgn;
        *v = zn * atol;
        csgn *= zt;
    }
    (nz, ierr)
}

/// `besy`: the Y Bessel function, `(H1 - H2) / 2i`. Returns `(nz, ierr)`.
pub(super) fn besy(z: C64, fnu: f64, kode: i32, n: i32, cy: &mut [C64]) -> (i32, i32) {
    let xx = z.re;
    let yy = z.im;
    let mut ierr = 0;
    let mut nz = 0;
    if (xx == 0.0) && (yy == 0.0) {
        ierr = 1;
    }
    if fnu < 0.0 {
        ierr = 1;
    }
    if !(1..=2).contains(&kode) {
        ierr = 1;
    }
    if n < 1 {
        ierr = 1;
    }
    if ierr != 0 {
        return (nz, ierr);
    }
    let hci = C64::new(0.0, 0.5);
    let (nz1, ierr1) = besh(z, fnu, kode, 1, n, cy);
    if (ierr1 != 0) && (ierr1 != 3) {
        return (0, ierr1);
    }
    let mut cwrk = vec![C64::ZERO; n as usize];
    let (nz2, ierr2) = besh(z, fnu, kode, 2, n, &mut cwrk);
    ierr = ierr2;
    if (ierr != 0) && (ierr != 3) {
        return (0, ierr);
    }
    nz = if nz1 > nz2 { nz2 } else { nz1 };
    if kode != 2 {
        for i in 0..(n as usize) {
            cy[i] = hci * (cwrk[i] - cy[i]);
        }
        return (nz, ierr);
    }
    let (tol, elim, _alim, _dig, _fnul, _rl) = machine();
    let exr = e::cos(xx);
    let exi = e::sin(xx);
    let mut ey = 0.0;
    let tay = (yy + yy).abs();
    if tay < elim {
        ey = e::exp(-tay);
    }
    let (c1, c2) = if yy < 0.0 {
        (C64::new(exr, exi), ey * C64::new(exr, -exi))
    } else {
        (ey * C64::new(exr, exi), C64::new(exr, -exi))
    };
    nz = 0;
    let rtol = 1.0 / tol;
    let ascle = 1e3 * D1MACH[0] * rtol;
    for i in 0..(n as usize) {
        let mut aa = cwrk[i].re;
        let mut bb = cwrk[i].im;
        let mut atol = 1.0;
        if fmax(aa.abs(), bb.abs()) <= ascle {
            aa *= rtol;
            bb *= rtol;
            atol = tol;
        }
        let mut st = C64::new(aa, bb) * c2 * atol;
        aa = cy[i].re;
        bb = cy[i].im;
        atol = 1.0;
        if fmax(aa.abs(), bb.abs()) <= ascle {
            aa *= rtol;
            bb *= rtol;
            atol = tol;
        }
        st -= C64::new(aa, bb) * c1 * atol;
        cy[i] = st * hci;
        if st.eq_real(0.0) && (ey == 0.0) {
            nz += 1;
        }
    }
    (nz, ierr)
}
