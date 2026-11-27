//! The AC expression type and its CANONICAL CONSTRUCTORS.
//!
//! Canonical invariants (established by [`add`], [`mul`], [`pow`], [`fun`]; every consumer may
//! rely on them):
//! * `Add(v)`: `v.len() >= 2`; no element is an `Add`; at most one `Num` (0 only in `f64`, the
//!   `+0` term kept where every other term may be -0 on a region, see `signed_zero_null`); at most
//!   one `Const`; like terms collected as far as the soundness gates allow; sorted by [`cmp_ex`].
//! * `Mul(v)`: `v.len() >= 2`; no element is a `Mul`; at most one `Num` (never 1; 0 only in the
//!   uncertified `0 * t` case documented at the zero-coefficient gate); at most one bare `Const`;
//!   like bases collected under the gates, and at most one power of e (`exp(a) exp(b)` is
//!   `exp(a + b)`); sorted by [`cmp_ex`].
//! * `Pow(b, e)`: `e` is never `Num(0)` or `Num(1)`; `b` is never a `Mul` when `e` is a
//!   non-negative or even integer (integer powers distribute) -- for ODD NEGATIVE `e` the
//!   distribution carries a zero-set licence (see `pow`), so a refused `Pow(Mul[..], e)`
//!   is a legal canonical form; rational^rational is folded when exact.
//! * No element anywhere is a nested bag of the same operator (Flat), and bags carry no order
//!   information beyond the canonical sort (Orderless).
//!
//! Soundness: every transform here is licensed either as TOTAL (equal as extended-real
//! evaluations everywhere), or as almost-everywhere equal with the null set certified by the
//! caller-provided certificates (the same two the shipped engine uses: `!` = finite a.e., `$` =
//! finite-and-nonzero a.e.), or as a `forall c_s exists c_t` fitted-constant absorption per the
//! contract. Each gate cites its counterexample. Refusing to transform is always sound; the
//! constructors prefer refusal over any uncertified step.

use crate::engine::RuleMode;
use std::cmp::Ordering;

use crate::tokens::{Tok, TokenView};

use super::laws::{Law, Powers};
use super::rat::{Mantissa, Rat};
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;

mod add;
mod canon;
mod classes;
mod coef;
mod confine;
mod cx;
mod ex;
mod flat;
mod fun;
mod lattice;
mod limit;
mod literals;
mod mul;
mod opposite;
mod order;
mod pow;
mod price;
mod sign_place;
mod terms;
mod written;
mod zero_sign;

pub use self::add::*;
pub use self::canon::*;
use self::classes::*;
pub(crate) use self::coef::*;
pub(crate) use self::confine::*;
pub use self::cx::*;
pub use self::ex::*;
pub(crate) use self::flat::*;
pub use self::fun::*;
pub(crate) use self::lattice::*;
use self::limit::*;
pub use self::literals::*;
pub use self::mul::*;
use self::opposite::*;
pub use self::order::*;
pub use self::pow::*;
pub use self::price::*;
pub(crate) use self::sign_place::*;
pub(crate) use self::terms::*;
pub use self::written::*;
pub use self::zero_sign::*;

#[cfg(test)]
mod tests;
