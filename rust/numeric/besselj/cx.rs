//! Complex double arithmetic exactly as the C++ the besselj port transliterates computes it:
//! libstdc++'s `std::complex<double>` compiled by GCC without fast-math.
//!
//! * `z * w` is `(ac - bd) + (ad + bc)i`, with the C runtime's `__muldc3` recovering the
//!   infinities only when both parts are NaN; `z / w` and `x / w` are the runtime's `__divdc3`
//!   (both through `numeric::elementary`).
//! * Mixed operations with a real scalar act componentwise (`x * z`, `z / x`), `x + z` and
//!   `z - x` touch the real part only, and `x - z` is `(-z) + x`, as libstdc++ writes them.
//! * `abs`, `exp`, `ln` and `sqrt` are the C library's `cabs`, `cexp`, `clog` and `csqrt`.

use crate::numeric::elementary::{self, Complex};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct C64 {
    pub(super) re: f64,
    pub(super) im: f64,
}

impl C64 {
    pub(super) const ZERO: C64 = C64 { re: 0.0, im: 0.0 };

    #[inline]
    pub(super) const fn new(re: f64, im: f64) -> C64 {
        C64 { re, im }
    }

    /// `std::complex<double>(x)`: imaginary part `+0`.
    #[inline]
    pub(super) const fn real(re: f64) -> C64 {
        C64 { re, im: 0.0 }
    }

    #[inline]
    pub(super) fn conj(self) -> C64 {
        C64 {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline]
    fn c(self) -> Complex {
        Complex {
            re: self.re,
            im: self.im,
        }
    }

    #[inline]
    fn from_c(z: Complex) -> C64 {
        C64 { re: z.re, im: z.im }
    }

    #[inline]
    pub(super) fn abs(self) -> f64 {
        elementary::cabs(self.c())
    }

    #[inline]
    pub(super) fn exp(self) -> C64 {
        C64::from_c(elementary::cexp(self.c()))
    }

    #[inline]
    pub(super) fn ln(self) -> C64 {
        C64::from_c(elementary::clog(self.c()))
    }

    #[inline]
    pub(super) fn sqrt(self) -> C64 {
        C64::from_c(elementary::csqrt(self.c()))
    }

    #[inline]
    pub(super) fn sinh(self) -> C64 {
        C64::from_c(elementary::csinh(self.c()))
    }

    #[inline]
    pub(super) fn cosh(self) -> C64 {
        C64::from_c(elementary::ccosh(self.c()))
    }

    /// `z == x` for a real `x`: `re == x && im == 0`.
    #[inline]
    pub(super) fn eq_real(self, x: f64) -> bool {
        self.re == x && self.im == 0.0
    }
}

impl Neg for C64 {
    type Output = C64;
    #[inline]
    fn neg(self) -> C64 {
        C64 {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl Add for C64 {
    type Output = C64;
    #[inline]
    fn add(self, w: C64) -> C64 {
        C64 {
            re: self.re + w.re,
            im: self.im + w.im,
        }
    }
}

impl Sub for C64 {
    type Output = C64;
    #[inline]
    fn sub(self, w: C64) -> C64 {
        C64 {
            re: self.re - w.re,
            im: self.im - w.im,
        }
    }
}

impl Mul for C64 {
    type Output = C64;
    #[inline]
    fn mul(self, w: C64) -> C64 {
        C64::from_c(elementary::cmul(self.re, self.im, w.re, w.im))
    }
}

impl Div for C64 {
    type Output = C64;
    #[inline]
    fn div(self, w: C64) -> C64 {
        C64::from_c(elementary::cdiv(self.re, self.im, w.re, w.im))
    }
}

impl Add<f64> for C64 {
    type Output = C64;
    #[inline]
    fn add(self, x: f64) -> C64 {
        C64 {
            re: self.re + x,
            im: self.im,
        }
    }
}

impl Sub<f64> for C64 {
    type Output = C64;
    #[inline]
    fn sub(self, x: f64) -> C64 {
        C64 {
            re: self.re - x,
            im: self.im,
        }
    }
}

impl Mul<f64> for C64 {
    type Output = C64;
    #[inline]
    fn mul(self, x: f64) -> C64 {
        C64 {
            re: self.re * x,
            im: self.im * x,
        }
    }
}

impl Div<f64> for C64 {
    type Output = C64;
    #[inline]
    fn div(self, x: f64) -> C64 {
        C64 {
            re: self.re / x,
            im: self.im / x,
        }
    }
}

impl Add<C64> for f64 {
    type Output = C64;
    #[inline]
    fn add(self, z: C64) -> C64 {
        C64 {
            re: z.re + self,
            im: z.im,
        }
    }
}

impl Sub<C64> for f64 {
    type Output = C64;
    /// libstdc++: `complex r = -z; r += x;`.
    #[inline]
    fn sub(self, z: C64) -> C64 {
        C64 {
            re: -z.re + self,
            im: -z.im,
        }
    }
}

impl Mul<C64> for f64 {
    type Output = C64;
    #[inline]
    fn mul(self, z: C64) -> C64 {
        C64 {
            re: z.re * self,
            im: z.im * self,
        }
    }
}

impl Div<C64> for f64 {
    type Output = C64;
    /// libstdc++: `complex r = x; r /= z;`, a full complex division.
    #[inline]
    fn div(self, z: C64) -> C64 {
        C64::real(self) / z
    }
}

impl AddAssign for C64 {
    #[inline]
    fn add_assign(&mut self, w: C64) {
        *self = *self + w;
    }
}

impl SubAssign for C64 {
    #[inline]
    fn sub_assign(&mut self, w: C64) {
        *self = *self - w;
    }
}

impl MulAssign for C64 {
    #[inline]
    fn mul_assign(&mut self, w: C64) {
        *self = *self * w;
    }
}

impl DivAssign for C64 {
    #[inline]
    fn div_assign(&mut self, w: C64) {
        *self = *self / w;
    }
}

impl AddAssign<f64> for C64 {
    #[inline]
    fn add_assign(&mut self, x: f64) {
        *self = *self + x;
    }
}

impl SubAssign<f64> for C64 {
    #[inline]
    fn sub_assign(&mut self, x: f64) {
        *self = *self - x;
    }
}

impl MulAssign<f64> for C64 {
    #[inline]
    fn mul_assign(&mut self, x: f64) {
        *self = *self * x;
    }
}

impl DivAssign<f64> for C64 {
    #[inline]
    fn div_assign(&mut self, x: f64) {
        *self = *self / x;
    }
}

/// The C conversion `(int) x` as x86-64 compiles it (`cvttsd2si`): truncation toward zero, and
/// `INT_MIN` for NaN and for anything outside the `int` range.
#[inline]
pub(super) fn c_int(x: f64) -> i32 {
    if x.is_nan() || x >= 2_147_483_648.0 || x <= -2_147_483_649.0 {
        i32::MIN
    } else {
        x as i32
    }
}
