//! A transliteration of the Cephes routines scipy 1.18.1 falls back to for a real-order Bessel
//! function of the first kind (`xsf::cephes::jv` and what it calls: `j0`, `j1`, `Gamma`,
//! `rgamma`, `lgam_sgn`, `airy`, `cbrt`, `sinpi`, `cospi`, `polevl`, `chbevl`), statement for
//! statement, so that every operation rounds as the C++ does.
//!
//! Cephes Math Library Release 2.8, Copyright 1984, 1987, 1989, 1992, 2000 by Stephen L.
//! Moshier, as translated into C++ by the SciPy developers (xsf, BSD-3-Clause); notices in
//! `LICENSES_bundled.txt`.

use super::cx::c_int;
use super::tables as t;
use crate::numeric::elementary as e;

const MACHEP: f64 = 1.1102230246251565404236316680908203125E-16;
const MAXLOG: f64 = 7.097827128933839730962063185871E2;
const SQRT1OPI: f64 = 5.641895835477562869480794515607725858441E-1;
const SQRT2OPI: f64 = 7.978845608028653558798921198687637369517E-1;
const SQRT2PI: f64 = 2.506628274631000502415765284811045253007;
const THPIO4: f64 = 2.356194490192344928846982537459627163148;
const SQRT3: f64 = 1.732050807568877293527446341505872366943;
const LOGPI: f64 = 1.144729885849400174143427351353058711647;
const MAXGAM: f64 = 171.624376956302725;
const M_PI: f64 = 3.14159265358979323846;
const M_PI_4: f64 = 0.785398163397448309616;
const M_SQRT1_2: f64 = 0.707106781186547524401;

/// `y = c0 x^N + ... + cN`, Horner from the highest coefficient (`coef[0]`).
#[inline]
fn polevl(x: f64, coef: &[f64], n: usize) -> f64 {
    let mut ans = coef[0];
    for &c in &coef[1..=n] {
        ans = ans * x + c;
    }
    ans
}

/// `polevl` with a leading coefficient 1 omitted from `coef`.
#[inline]
fn p1evl(x: f64, coef: &[f64], n: usize) -> f64 {
    let mut ans = x + coef[0];
    for &c in &coef[1..n] {
        ans = ans * x + c;
    }
    ans
}

fn chbevl(x: f64, array: &[f64], n: usize) -> f64 {
    let mut b0 = array[0];
    let mut b1 = 0.0;
    let mut b2 = 0.0;
    for &c in &array[1..n] {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + c;
    }
    0.5 * (b0 - b2)
}

/// `sin(pi x)`.
pub(super) fn sinpi(mut x: f64) -> f64 {
    let mut s = 1.0;
    if x.is_sign_negative() {
        x = -x;
        s = -1.0;
    }
    let r = e::fmod(x, 2.0);
    if r < 0.5 {
        s * e::sin(M_PI * r)
    } else if r > 1.5 {
        s * e::sin(M_PI * (r - 2.0))
    } else {
        -s * e::sin(M_PI * (r - 1.0))
    }
}

/// `cos(pi x)`, `+0` (never -0) at the half-integers.
pub(super) fn cospi(mut x: f64) -> f64 {
    if x < 0.0 {
        x = -x;
    }
    let r = e::fmod(x, 2.0);
    if r == 0.5 {
        return 0.0;
    }
    if r < 1.0 {
        -e::sin(M_PI * (r - 0.5))
    } else {
        e::sin(M_PI * (r - 1.5))
    }
}

fn cbrt(mut x: f64) -> f64 {
    const CBRT2: f64 = 1.2599210498948731647672;
    const CBRT4: f64 = 1.5874010519681994747517;
    const CBRT2I: f64 = 0.79370052598409973737585;
    const CBRT4I: f64 = 0.62996052494743658238361;
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let sign = if x > 0.0 {
        1
    } else {
        x = -x;
        -1
    };
    let z = x;
    let (m, mut ex) = e::frexp(x);
    x = m;
    x = (((-1.3466110473359520655053e-1 * x + 5.4664601366395524503440e-1) * x
        - 9.5438224771509446525043e-1)
        * x
        + 1.1399983354717293273738e0)
        * x
        + 4.0238979564544752126924e-1;
    if ex >= 0 {
        let mut rem = ex;
        ex /= 3;
        rem -= 3 * ex;
        if rem == 1 {
            x *= CBRT2;
        } else if rem == 2 {
            x *= CBRT4;
        }
    } else {
        ex = -ex;
        let mut rem = ex;
        ex /= 3;
        rem -= 3 * ex;
        if rem == 1 {
            x *= CBRT2I;
        } else if rem == 2 {
            x *= CBRT4I;
        }
        ex = -ex;
    }
    x = e::ldexp(x, ex);
    x -= (x - (z / (x * x))) * 0.33333333333333333333;
    x -= (x - (z / (x * x))) * 0.33333333333333333333;
    if sign < 0 {
        x = -x;
    }
    x
}

fn stirf(x: f64) -> f64 {
    const MAXSTIR: f64 = 143.01608;
    if x >= MAXGAM {
        return f64::INFINITY;
    }
    let mut w = 1.0 / x;
    w = 1.0 + w * polevl(w, &t::GAMMA_STIR, 4);
    let mut y = e::exp(x);
    if x > MAXSTIR {
        let v = e::pow(x, 0.5 * x - 0.25);
        y = v * (v / y);
    } else {
        y = e::pow(x, x - 0.5) / y;
    }
    SQRT2PI * y * w
}

#[allow(non_snake_case)]
fn Gamma(mut x: f64) -> f64 {
    let mut sgngam: i32 = 1;
    if !x.is_finite() {
        if x > 0.0 {
            return x;
        }
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY.copysign(x);
    }
    let mut q = x.abs();
    if q > 33.0 {
        let z;
        if x < 0.0 {
            let mut p = q.floor();
            if p == q {
                return f64::NAN;
            }
            let i = c_int(p);
            if (i & 1) == 0 {
                sgngam = -1;
            }
            let mut zz = q - p;
            if zz > 0.5 {
                p += 1.0;
                zz = q - p;
            }
            zz = q * sinpi(zz);
            if zz == 0.0 {
                return sgngam as f64 * f64::INFINITY;
            }
            zz = zz.abs();
            z = M_PI / (zz * stirf(q));
        } else {
            z = stirf(x);
        }
        return sgngam as f64 * z;
    }
    let mut z = 1.0;
    while x >= 3.0 {
        x -= 1.0;
        z *= x;
    }
    let small = |x: f64, z: f64| -> f64 {
        if x == 0.0 {
            f64::NAN
        } else {
            z / ((1.0 + 0.5772156649015329 * x) * x)
        }
    };
    while x < 0.0 {
        if x > -1.0E-9 {
            return small(x, z);
        }
        z /= x;
        x += 1.0;
    }
    while x < 2.0 {
        if x < 1.0e-9 {
            return small(x, z);
        }
        z /= x;
        x += 1.0;
    }
    if x == 2.0 {
        return z;
    }
    x -= 2.0;
    let p = polevl(x, &t::GAMMA_P, 6);
    q = polevl(x, &t::GAMMA_Q, 7);
    z * p / q
}

fn lgam_large_x(x: f64) -> f64 {
    const LS2PI: f64 = 0.91893853320467274178;
    let q = (x - 0.5) * e::log(x) - x + LS2PI;
    if x > 1.0e8 {
        return q;
    }
    let mut p = 1.0 / (x * x);
    p = ((7.9365079365079365079365e-4 * p - 2.7777777777777777777778e-3) * p
        + 0.0833333333333333333333)
        / x;
    q + p
}

fn lgam_sgn(mut x: f64, sign: &mut i32) -> f64 {
    const LS2PI: f64 = 0.91893853320467274178;
    const MAXLGM: f64 = 2.556348e305;
    *sign = 1;
    if !x.is_finite() {
        return x;
    }
    if x < -34.0 {
        let q = -x;
        let w = lgam_sgn(q, sign);
        let mut p = q.floor();
        if p == q {
            return f64::INFINITY;
        }
        let i = c_int(p);
        *sign = if (i & 1) == 0 { -1 } else { 1 };
        let mut z = q - p;
        if z > 0.5 {
            p += 1.0;
            z = p - q;
        }
        z = q * sinpi(z);
        if z == 0.0 {
            return f64::INFINITY;
        }
        return LOGPI - e::log(z) - w;
    }
    if x < 13.0 {
        let mut z = 1.0;
        let mut p = 0.0;
        let mut u = x;
        while u >= 3.0 {
            p -= 1.0;
            u = x + p;
            z *= u;
        }
        while u < 2.0 {
            if u == 0.0 {
                return f64::INFINITY;
            }
            z /= u;
            p += 1.0;
            u = x + p;
        }
        if z < 0.0 {
            *sign = -1;
            z = -z;
        } else {
            *sign = 1;
        }
        if u == 2.0 {
            return e::log(z);
        }
        p -= 2.0;
        x += p;
        p = x * polevl(x, &t::GAMMA_B, 5) / p1evl(x, &t::GAMMA_C, 6);
        return e::log(z) + p;
    }
    if x > MAXLGM {
        return *sign as f64 * f64::INFINITY;
    }
    if x >= 1000.0 {
        return lgam_large_x(x);
    }
    let q = (x - 0.5) * e::log(x) - x + LS2PI;
    let p = 1.0 / (x * x);
    q + polevl(p, &t::GAMMA_A, 4) / x
}

/// `1/Gamma(x)`.
pub(super) fn rgamma(x: f64) -> f64 {
    if x == 0.0 {
        return x;
    }
    if x < 0.0 && x == x.floor() {
        return 0.0;
    }
    if x.abs() > 4.0 {
        return 1.0 / Gamma(x);
    }
    let mut z = 1.0;
    let mut w = x;
    while w > 1.0 {
        w -= 1.0;
        z *= w;
    }
    while w < 0.0 {
        z /= w;
        w += 1.0;
    }
    if w == 0.0 {
        return 0.0;
    }
    if w == 1.0 {
        return 1.0 / z;
    }
    w * (1.0 + chbevl(4.0 * w - 2.0, &t::RGAMMA_R, 16)) / z
}

/// The Airy functions `(Ai, Ai', Bi, Bi')` of a real argument.
fn airy(x: f64) -> (f64, f64, f64, f64) {
    const C1: f64 = 0.35502805388781723926;
    const C2: f64 = 0.258819403792806798405;
    const MAXAIRY: f64 = 103.892;
    let (mut ai, mut aip, mut bi, mut bip) = (0.0, 0.0, 0.0, 0.0);
    let mut domflg = 0;
    if x > MAXAIRY {
        return (0.0, 0.0, f64::INFINITY, f64::INFINITY);
    }
    if x < -2.09 {
        let mut tt = (-x).sqrt();
        let zeta = -2.0 * x * tt / 3.0;
        tt = tt.sqrt();
        let mut k = SQRT1OPI / tt;
        let z = 1.0 / zeta;
        let zz = z * z;
        let mut uf = 1.0 + zz * polevl(zz, &t::AIRY_AFN, 8) / p1evl(zz, &t::AIRY_AFD, 9);
        let mut ug = z * polevl(zz, &t::AIRY_AGN, 10) / p1evl(zz, &t::AIRY_AGD, 10);
        let theta = zeta + 0.25 * M_PI;
        let f = e::sin(theta);
        let g = e::cos(theta);
        ai = k * (f * uf - g * ug);
        bi = k * (g * uf + f * ug);
        uf = 1.0 + zz * polevl(zz, &t::AIRY_APFN, 8) / p1evl(zz, &t::AIRY_APFD, 9);
        ug = z * polevl(zz, &t::AIRY_APGN, 10) / p1evl(zz, &t::AIRY_APGD, 10);
        k = SQRT1OPI * tt;
        aip = -k * (g * uf + f * ug);
        bip = k * (f * uf - g * ug);
        return (ai, aip, bi, bip);
    }
    if x >= 2.09 {
        domflg = 5;
        let mut tt = x.sqrt();
        let zeta = 2.0 * x * tt / 3.0;
        let g = e::exp(zeta);
        tt = tt.sqrt();
        let mut k = 2.0 * tt * g;
        let z = 1.0 / zeta;
        let mut f = polevl(z, &t::AIRY_AN, 7) / polevl(z, &t::AIRY_AD, 7);
        ai = SQRT1OPI * f / k;
        k = -0.5 * SQRT1OPI * tt / g;
        f = polevl(z, &t::AIRY_APN, 7) / polevl(z, &t::AIRY_APD, 7);
        aip = f * k;
        if x > 8.3203353 {
            f = z * polevl(z, &t::AIRY_BN16, 4) / p1evl(z, &t::AIRY_BD16, 5);
            k = SQRT1OPI * g;
            bi = k * (1.0 + f) / tt;
            f = z * polevl(z, &t::AIRY_BPPN, 4) / p1evl(z, &t::AIRY_BPPD, 5);
            bip = k * tt * (1.0 + f);
            return (ai, aip, bi, bip);
        }
    }
    let mut f = 1.0;
    let mut g = x;
    let mut tt = 1.0;
    let mut uf = 1.0;
    let mut ug = x;
    let mut k = 1.0;
    let z = x * x * x;
    while tt > MACHEP {
        uf *= z;
        k += 1.0;
        uf /= k;
        ug *= z;
        k += 1.0;
        ug /= k;
        uf /= k;
        f += uf;
        k += 1.0;
        ug /= k;
        g += ug;
        tt = (uf / f).abs();
    }
    uf = C1 * f;
    ug = C2 * g;
    if (domflg & 1) == 0 {
        ai = uf - ug;
    }
    if (domflg & 2) == 0 {
        bi = SQRT3 * (uf + ug);
    }
    k = 4.0;
    uf = x * x / 2.0;
    ug = z / 3.0;
    f = uf;
    g = 1.0 + ug;
    uf /= 3.0;
    tt = 1.0;
    while tt > MACHEP {
        uf *= z;
        ug /= k;
        k += 1.0;
        ug *= z;
        uf /= k;
        f += uf;
        k += 1.0;
        ug /= k;
        uf /= k;
        g += ug;
        k += 1.0;
        tt = (ug / g).abs();
    }
    uf = C1 * f;
    ug = C2 * g;
    if (domflg & 4) == 0 {
        aip = uf - ug;
    }
    if (domflg & 8) == 0 {
        bip = SQRT3 * (uf + ug);
    }
    (ai, aip, bi, bip)
}

fn j0(mut x: f64) -> f64 {
    const DR1: f64 = 5.78318596294678452118E0;
    const DR2: f64 = 3.04712623436620863991E1;
    if x < 0.0 {
        x = -x;
    }
    if x <= 5.0 {
        let z = x * x;
        if x < 1.0e-5 {
            return 1.0 - z / 4.0;
        }
        let mut p = (z - DR1) * (z - DR2);
        p = p * polevl(z, &t::J0_RP, 3) / p1evl(z, &t::J0_RQ, 8);
        return p;
    }
    let w = 5.0 / x;
    let mut q = 25.0 / (x * x);
    let mut p = polevl(q, &t::J0_PP, 6) / polevl(q, &t::J0_PQ, 6);
    q = polevl(q, &t::J0_QP, 7) / p1evl(q, &t::J0_QQ, 7);
    if x < 10.0 {
        let xn = x - M_PI_4;
        p = p * e::cos(xn) - w * q * e::sin(xn);
        return p * SQRT2OPI / x.sqrt();
    }
    p = (p + w * q) * e::cos(x) + (p - w * q) * e::sin(x);
    p * SQRT1OPI / x.sqrt()
}

fn j1(x: f64) -> f64 {
    const Z1: f64 = 1.46819706421238932572E1;
    const Z2: f64 = 4.92184563216946036703E1;
    let mut w = x;
    if x < 0.0 {
        return -j1(-x);
    }
    if w <= 5.0 {
        let z = x * x;
        w = polevl(z, &t::J1_RP, 3) / p1evl(z, &t::J1_RQ, 8);
        w = w * x * (z - Z1) * (z - Z2);
        return w;
    }
    w = 5.0 / x;
    let z = w * w;
    let mut p = polevl(z, &t::J1_PP, 6) / polevl(z, &t::J1_PQ, 6);
    let q = polevl(z, &t::J1_QP, 7) / p1evl(z, &t::J1_QQ, 7);
    if x < 10.0 {
        let xn = x - THPIO4;
        p = p * e::cos(xn) - w * q * e::sin(xn);
        return p * SQRT2OPI / x.sqrt();
    }
    let a = M_SQRT1_2 * (w * q - p);
    let b = M_SQRT1_2 * (p + w * q);
    (a * e::cos(x) + b * e::sin(x)) * SQRT2OPI / x.sqrt()
}

/// Reduce the order by backward recurrence (AMS55 #9.1.27 and 9.1.73).
fn jv_recur(n: &mut f64, x: f64, newn: &mut f64, cancel: bool) -> f64 {
    const BIG: f64 = 1.44115188075855872E+17;
    let maxiter = 22000;
    let mut miniter = c_int(x.abs() - n.abs());
    if miniter < 1 {
        miniter = 1;
    }
    let mut nflag = if *n < 0.0 { 1 } else { 0 };
    let mut ans;
    let (mut pk, mut pkm1, mut pkm2);
    loop {
        pkm2 = 0.0;
        let mut qkm2 = 1.0;
        pkm1 = x;
        let mut qkm1 = *n + *n;
        let xk = -x * x;
        let mut yk = qkm1;
        ans = 0.0;
        let mut ctr = 0;
        loop {
            yk += 2.0;
            pk = pkm1 * yk + pkm2 * xk;
            let qk = qkm1 * yk + qkm2 * xk;
            pkm2 = pkm1;
            pkm1 = pk;
            qkm2 = qkm1;
            qkm1 = qk;
            let r = if qk != 0.0 && ctr > miniter {
                pk / qk
            } else {
                0.0
            };
            let t;
            if r != 0.0 {
                t = ((ans - r) / r).abs();
                ans = r;
            } else {
                t = 1.0;
            }
            ctr += 1;
            if ctr > maxiter || t < MACHEP {
                break;
            }
            if pk.abs() > BIG {
                pkm2 /= BIG;
                pkm1 /= BIG;
                qkm2 /= BIG;
                qkm1 /= BIG;
            }
            if !(t > MACHEP) {
                break;
            }
        }
        if ans == 0.0 {
            ans = 1.0;
        }
        if nflag > 0 && ans.abs() < 0.125 {
            nflag = -1;
            *n -= 1.0;
            continue;
        }
        break;
    }
    let kf = *newn;
    pk = 1.0;
    pkm1 = 1.0 / ans;
    let mut k = *n - 1.0;
    let mut r = 2.0 * k;
    loop {
        pkm2 = (pkm1 * r - pk * x) / x;
        pk = pkm1;
        pkm1 = pkm2;
        r -= 2.0;
        k -= 1.0;
        if !(k > (kf + 0.5)) {
            break;
        }
    }
    if cancel && (kf >= 0.0) && (pk.abs() > pkm1.abs()) {
        k += 1.0;
        pkm2 = pk;
    }
    *newn = k;
    pkm2
}

/// Ascending power series (AMS55 #9.1.10).
fn jv_jvs(n: f64, x: f64) -> f64 {
    let z = -x * x / 4.0;
    let mut u = 1.0;
    let mut y = u;
    let mut k = 1.0;
    let mut t = 1.0;
    while t > MACHEP {
        u *= z / (k * (n + k));
        y += u;
        k += 1.0;
        if y != 0.0 {
            t = (u / y).abs();
        }
    }
    let (_, ex) = e::frexp(0.5 * x);
    let ex = c_int(ex as f64 * n);
    if (ex > -1023) && (ex < 1023) && (n > 0.0) && (n < (MAXGAM - 1.0)) {
        t = e::pow(0.5 * x, n) * rgamma(n + 1.0);
        y *= t;
    } else {
        let mut sgngam = 0;
        t = n * e::log(0.5 * x) - lgam_sgn(n + 1.0, &mut sgngam);
        if y < 0.0 {
            sgngam = -sgngam;
            y = -y;
        }
        t += e::log(y);
        if t < -MAXLOG {
            return 0.0;
        }
        if t > MAXLOG {
            return f64::INFINITY;
        }
        y = sgngam as f64 * e::exp(t);
    }
    y
}

/// Hankel's asymptotic expansion for large x (AMS55 #9.2.5).
fn jv_hankel(n: f64, x: f64) -> f64 {
    let m = 4.0 * n * n;
    let mut j = 1.0;
    let z = 8.0 * x;
    let mut k = 1.0;
    let mut p = 1.0;
    let mut u = (m - 1.0) / z;
    let mut q = u;
    let mut sign = 1.0;
    let mut conv = 1.0;
    let mut flag = 0;
    let mut t = 1.0;
    let mut pp = 1.0e38;
    let mut qq = 1.0e38;
    while t > MACHEP {
        k += 2.0;
        j += 1.0;
        sign = -sign;
        u *= (m - k * k) / (j * z);
        p += sign * u;
        k += 2.0;
        j += 1.0;
        u *= (m - k * k) / (j * z);
        q += sign * u;
        t = (u / p).abs();
        if t < conv {
            conv = t;
            qq = q;
            pp = p;
            flag = 1;
        }
        if (flag != 0) && (t > conv) {
            break;
        }
    }
    let u = x - (0.5 * n + 0.25) * M_PI;
    (2.0 / (M_PI * x)).sqrt() * (pp * e::cos(u) - qq * e::sin(u))
}

/// Transition-region expansion, n large and x close to n (AMS55 #9.3.23).
fn jv_jnt(n: f64, x: f64) -> f64 {
    let cbn = cbrt(n);
    let z = (x - n) / cbn;
    let cbtwo = cbrt(2.0);
    let mut zz = -cbtwo * z;
    let (ai, aip, _, _) = airy(zz);
    zz = z * z;
    let z3 = zz * z;
    let f = [
        1.0,
        -z / 5.0,
        polevl(z3, &t::JV_PF2, 1) * zz,
        polevl(z3, &t::JV_PF3, 2),
        polevl(z3, &t::JV_PF4, 3) * z,
    ];
    let g = [
        0.3 * zz,
        polevl(z3, &t::JV_PG1, 1),
        polevl(z3, &t::JV_PG2, 2) * z,
        polevl(z3, &t::JV_PG3, 2) * zz,
    ];
    let mut pp = 0.0;
    let mut qq = 0.0;
    let mut nk = 1.0;
    let n23 = cbrt(n * n);
    for k in 0..=4 {
        pp += f[k] * nk;
        if k != 4 {
            qq += g[k] * nk;
        }
        nk /= n23;
    }
    cbtwo * ai * pp / cbn + cbrt(4.0) * aip * qq / n
}

/// Uniform asymptotic expansion for large n (AMS55 #9.3.35).
fn jv_jnx(n: f64, x: f64) -> f64 {
    let cbn = cbrt(n);
    let mut z = (x - n) / cbn;
    if z.abs() <= 0.7 {
        return jv_jnt(n, x);
    }
    z = x / n;
    let zz = 1.0 - z * z;
    if zz == 0.0 {
        return 0.0;
    }
    let (sz, mut t, zeta, nflg);
    if zz > 0.0 {
        sz = zz.sqrt();
        t = 1.5 * (e::log((1.0 + sz) / z) - sz);
        zeta = cbrt(t * t);
        nflg = 1;
    } else {
        sz = (-zz).sqrt();
        t = 1.5 * (sz - e::acos(1.0 / z));
        zeta = -cbrt(t * t);
        nflg = -1;
    }
    let z32i = (1.0 / t).abs();
    let sqz = cbrt(t);
    let n23 = cbrt(n * n);
    t = n23 * zeta;
    let (ai, aip, _, _) = airy(t);
    let mut u = [0.0f64; 8];
    u[0] = 1.0;
    let zzi = 1.0 / zz;
    u[1] = polevl(zzi, &t::JV_P1, 1) / sz;
    u[2] = polevl(zzi, &t::JV_P2, 2) / zz;
    u[3] = polevl(zzi, &t::JV_P3, 3) / (sz * zz);
    let mut pp = zz * zz;
    u[4] = polevl(zzi, &t::JV_P4, 4) / pp;
    u[5] = polevl(zzi, &t::JV_P5, 5) / (pp * sz);
    pp *= zz;
    u[6] = polevl(zzi, &t::JV_P6, 6) / pp;
    u[7] = polevl(zzi, &t::JV_P7, 7) / (pp * sz);
    pp = 0.0;
    let mut qq = 0.0;
    let mut np = 1.0;
    let mut doa = true;
    let mut dob = true;
    let mut akl = f64::INFINITY;
    let mut bkl = f64::INFINITY;
    for k in 0..=3usize {
        let tk = 2 * k;
        let tkp1 = tk + 1;
        let mut zp = 1.0;
        let mut ak = 0.0;
        let mut bk = 0.0;
        for s in 0..=tk {
            if doa {
                let sign = if (s & 3) > 1 { nflg } else { 1 };
                ak += sign as f64 * t::JV_MU[s] * zp * u[tk - s];
            }
            if dob {
                let m = tkp1 - s;
                let sign = if ((m + 1) & 3) > 1 { nflg } else { 1 };
                bk += sign as f64 * t::JV_LAMBDA[s] * zp * u[m];
            }
            zp *= z32i;
        }
        if doa {
            ak *= np;
            t = ak.abs();
            if t < akl {
                akl = t;
                pp += ak;
            } else {
                doa = false;
            }
        }
        if dob {
            bk += t::JV_LAMBDA[tkp1] * zp * u[0];
            bk *= -np / sqz;
            t = bk.abs();
            if t < bkl {
                bkl = t;
                qq += bk;
            } else {
                dob = false;
            }
        }
        if np < MACHEP {
            break;
        }
        np /= n * n;
    }
    t = 4.0 * zeta / zz;
    t = t.sqrt().sqrt();
    t *= ai * pp / cbrt(n) + aip * qq / (n23 * n);
    t
}

/// `xsf::cephes::jv(n, x)`: the Bessel function of the first kind of real order.
pub(super) fn jv(mut n: f64, mut x: f64) -> f64 {
    let mut nint = false;
    let mut sign: i32 = 1;
    let an = n.abs();
    let mut y = an.floor();
    if y == an {
        nint = true;
        let i = c_int(an - 16384.0 * (an / 16384.0).floor());
        if n < 0.0 {
            if (i & 1) != 0 {
                sign = -sign;
            }
            n = an;
        }
        if x < 0.0 {
            if (i & 1) != 0 {
                sign = -sign;
            }
            x = -x;
        }
        if n == 0.0 {
            return j0(x);
        }
        if n == 1.0 {
            return sign as f64 * j1(x);
        }
    }
    if (x < 0.0) && (y != an) {
        return sign as f64 * f64::NAN;
    }
    if x == 0.0 && n < 0.0 && !nint {
        return f64::INFINITY * rgamma(n + 1.0);
    }
    y = x.abs();
    if y * y < (n + 1.0).abs() * MACHEP {
        return e::pow(0.5 * x, n) * rgamma(n + 1.0);
    }
    let mut k = 3.6 * y.sqrt();
    let mut t = 3.6 * an.sqrt();
    if (y < t) && (an > 21.0) {
        return sign as f64 * jv_jvs(n, x);
    }
    if (an < k) && (y > 21.0) {
        return sign as f64 * jv_hankel(n, x);
    }
    let q;
    if an < 500.0 {
        if nint {
            k = 0.0;
            let q = jv_recur(&mut n, x, &mut k, true);
            if k == 0.0 {
                return sign as f64 * (j0(x) / q);
            }
            if k == 1.0 {
                return sign as f64 * (j1(x) / q);
            }
        }
        if (an > 2.0 * y) || ((n >= 0.0) && (n < 20.0) && (y > 6.0) && (y < 20.0)) {
            // rlarger: recur backwards from a larger order
            k = n;
            y = y + an + 1.0;
            if y < 30.0 {
                y = 30.0;
            }
            y = n + (y - n).floor();
            let q = jv_recur(&mut y, x, &mut k, false);
            return sign as f64 * (jv_jvs(y, x) * q);
        }
        if k <= 30.0 {
            k = 2.0;
        } else if k < 90.0 {
            k = (3.0 * k) / 4.0;
        }
        if an > (k + 3.0) {
            if n < 0.0 {
                k = -k;
            }
            let frac = n - n.floor();
            k = k.floor() + frac;
            if n > 0.0 {
                q = jv_recur(&mut n, x, &mut k, true);
            } else {
                let mut tt = k;
                k = n;
                q = jv_recur(&mut tt, x, &mut k, true);
                k = tt;
            }
            if q == 0.0 {
                return sign as f64 * 0.0;
            }
        } else {
            k = n;
            q = 1.0;
        }
        y = k.abs();
        if y < 26.0 {
            t = (0.0083 * y + 0.09) * y + 12.9;
        } else {
            t = 0.9 * y;
        }
        if x > t {
            y = jv_hankel(k, x);
        } else {
            y = jv_jvs(k, x);
        }
        if n > 0.0 {
            y /= q;
        } else {
            y *= q;
        }
    } else {
        if n < 0.0 {
            return sign as f64 * f64::NAN;
        }
        t = x / n;
        t /= n;
        if t > 0.3 {
            y = jv_hankel(n, x);
        } else {
            y = jv_jnx(n, x);
        }
    }
    sign as f64 * y
}
