//! The power constructor, and integers beyond 128 bits.

use super::*;

#[test]
fn pow_folds_and_licences() {
    with_view(|view| {
        let cx = Cx::bare(view);
        assert_eq!(pow(x(view), Ex::int(1), &cx), x(view));
        assert_eq!(pow(x(view), Ex::int(0), &cx), Ex::int(1));
        assert_eq!(pow(Ex::int(1), x(view), &cx), Ex::int(1));
        assert_eq!(pow(Ex::int(2), Ex::int(10), &cx), Ex::int(1024));
        assert_eq!(
            pow(Ex::int(4), Ex::Num(Rat::new(1, 2).unwrap()), &cx),
            Ex::int(2)
        );
        assert_eq!(
            pow(Ex::int(2), Ex::int(-2), &cx),
            Ex::Num(Rat::new(1, 4).unwrap())
        );
        // Inexact EVEN root stays symbolic -- as a `rootn`, which is now the canonical
        // spelling for an even index.
        let e = pow(Ex::int(2), Ex::Num(Rat::new(1, 2).unwrap()), &cx);
        assert!(matches!(e, Ex::Fun(..)), "expected rootn(2,2), got {e:?}");
        // (-8)^(1/3) is NOT the real root in Pow semantics -> stays symbolic.
        let e = pow(Ex::int(-8), Ex::Num(Rat::new(1, 3).unwrap()), &cx);
        assert!(matches!(e, Ex::Pow(..)));
        // (x^2)^3 -> x^6 (both integers).
        let e = pow(pow(x(view), Ex::int(2), &cx), Ex::int(3), &cx);
        assert_eq!(e, Ex::Pow(Box::new(x(view)), Box::new(Ex::int(6))));
        // (x^2)^(1/2) must NOT become x (|x| != x).
        let e = pow(
            pow(x(view), Ex::int(2), &cx),
            Ex::Num(Rat::new(1, 2).unwrap()),
            &cx,
        );
        // Refusal is the PROPERTY: (x^2)^(1/2) must not become x. It is now spelled
        // `rootn(x^2, 2)` (= |x|), which is exactly the refusal, so assert the property
        // rather than the old Pow-of-Pow shape.
        assert_ne!(e, x(view), "sign-erasing composition must refuse: {e:?}");
        // (x^3)^(1/3) must NOT become x in POW semantics: on x < 0 the source is
        // NaN (principal non-integer power of a negative value) while x is finite
        // -- a value fabrication; oddness of the inner exponent licenses
        // nothing about the OUTER principal root. The sound spelling of the
        // inverse composition is rootn (contract s5), tested below.
        let e = pow(
            pow(x(view), Ex::int(3), &cx),
            Ex::Num(Rat::new(1, 3).unwrap()),
            &cx,
        );
        assert!(
            matches!(&e, Ex::Pow(b, _) if matches!(&**b, Ex::Pow(..))),
            "integer-product composition must refuse: {e:?}"
        );
        // (x^5)^0.2 must refuse too.
        let e = pow(
            pow(x(view), Ex::int(5), &cx),
            Ex::Num(Rat::new(1, 5).unwrap()),
            &cx,
        );
        assert!(matches!(&e, Ex::Pow(b, _) if matches!(&**b, Ex::Pow(..))));
        // (x^3)^(1/5) -> x^(3/5): odd inner AND non-integer product -- both sides
        // are NaN on the whole negative half-line, so the fold is exact.
        let e = pow(
            pow(x(view), Ex::int(3), &cx),
            Ex::Num(Rat::new(1, 5).unwrap()),
            &cx,
        );
        assert_eq!(
            e,
            Ex::Pow(
                Box::new(x(view)),
                Box::new(Ex::Num(Rat::new(3, 5).unwrap()))
            )
        );
        // rootn(x^n, n) -> x for the SAME odd n: total at every value including
        // nan and +-inf (SIMPLIFICATION_CONTRACT_v2 s5 Corollaries), so the
        // CONSTRUCTOR folds it -- the composition pow refuses lives here soundly.
        let rootn_tok = view.intern("rootn");
        let e = fun(
            rootn_tok,
            vec![pow(x(view), Ex::int(3), &cx), Ex::int(3)],
            &cx,
        );
        assert_eq!(e, x(view));
        let e = fun(
            rootn_tok,
            vec![pow(x(view), Ex::int(5), &cx), Ex::int(5)],
            &cx,
        );
        assert_eq!(e, x(view));
        // ...but only for the MATCHING index: rootn(x^3, 5) keeps the Fun.
        let e = fun(
            rootn_tok,
            vec![pow(x(view), Ex::int(3), &cx), Ex::int(5)],
            &cx,
        );
        assert!(matches!(&e, Ex::Fun(..)), "{e:?}");
        // (x^(1/2))^2 -> must NOT become x (would fabricate a value on x < 0). The
        // inner is spelled `rootn(x,2)` now, so assert the PROPERTY: the licence arm
        // refuses without a non-negativity certificate.
        let e = pow(
            pow(x(view), Ex::Num(Rat::new(1, 2).unwrap()), &cx),
            Ex::int(2),
            &cx,
        );
        assert_ne!(
            e,
            x(view),
            "unlicensed root/power inverse must refuse: {e:?}"
        );
        // ...but (x^(1/2))^3 -> x^(3/2) (still non-integer: NaN region preserved). The
        // composition arm reads the even root as a rational power via
        // `as_rational_power`, so the licence logic applies unchanged.
        let e = pow(
            pow(x(view), Ex::Num(Rat::new(1, 2).unwrap()), &cx),
            Ex::int(3),
            &cx,
        );
        assert_eq!(
            e,
            Ex::Pow(
                Box::new(x(view)),
                Box::new(Ex::Num(Rat::new(3, 2).unwrap()))
            )
        );
        // (2x)^2 -> 4x^2 (integer power distributes).
        let e = pow(mul(vec![Ex::int(2), x(view)], &cx), Ex::int(2), &cx);
        assert_eq!(
            e,
            Ex::Mul(vec![
                Ex::int(4),
                Ex::Pow(Box::new(x(view)), Box::new(Ex::int(2)))
            ])
        );
    });
}

/// A NEGATIVE exponent over an ODD `rootn` had no arm at all,
/// so the reduction was route-dependent -- the explicit projection re-spelled it and the
/// re-parse then reduced, leaving the projection outside its own fixpoint (and tripping
/// the serialization-stability `debug_assert` in `engine/ac.rs`).
#[test]
fn odd_rootn_negative_exponent_composes() {
    with_view(|view| {
        let cx = Cx::bare(view);
        let rootn_tok = view.intern("rootn");
        let root = |b: Ex, n: i128| fun(rootn_tok, vec![b, Ex::int(n)], &cx);
        let r = |p: i128, q: i128| Ex::Num(Rat::new(p, q).unwrap());

        // ODD-ROOT COMPOSITION: sound for every NON-INTEGER exponent, because both sides
        // are NaN on a negative base. `rootn(x,3)^(-1/2) -> x^(-1/6)`, matching what the
        // reciprocal spelling already produced.
        assert_eq!(
            pow(root(x(view), 3), r(-1, 2), &cx),
            Ex::Pow(Box::new(x(view)), Box::new(r(-1, 6)))
        );
        assert_eq!(
            pow(root(x(view), 5), r(-1, 2), &cx),
            Ex::Pow(Box::new(x(view)), Box::new(r(-1, 10)))
        );
        assert_eq!(
            pow(root(x(view), 3), r(-3, 2), &cx),
            Ex::Pow(Box::new(x(view)), Box::new(r(-1, 2)))
        );
        // The positive direction is unchanged and still lands on the root spelling.
        assert_eq!(pow(root(x(view), 3), r(1, 2), &cx), root(x(view), 6));

        // INTEGER exponents stay excluded -- `rootn(-8,3)^2 = 4` while `(-8)^(2/3)` is
        // NaN, a positive-measure disagreement. The `rootn` must survive.
        assert!(matches!(
            pow(root(x(view), 3), Ex::int(2), &cx),
            Ex::Pow(..)
        ));
        assert!(matches!(
            pow(root(x(view), 3), Ex::int(-2), &cx),
            Ex::Pow(..)
        ));

        // ROOT/POWER INVERSE, negative mirror: `rootn(x,n)^(-n) -> 1/x` for odd n.
        assert_eq!(
            pow(root(x(view), 3), Ex::int(-3), &cx),
            Ex::Pow(Box::new(x(view)), Box::new(Ex::int(-1)))
        );
        assert_eq!(
            pow(root(x(view), 5), Ex::int(-5), &cx),
            Ex::Pow(Box::new(x(view)), Box::new(Ex::int(-1)))
        );
        // ...positive mirror unchanged, and only for the MATCHING index.
        assert_eq!(pow(root(x(view), 3), Ex::int(3), &cx), x(view));
        assert!(matches!(
            pow(root(x(view), 3), Ex::int(-5), &cx),
            Ex::Pow(..)
        ));
        // EVEN index needs a certainly-non-negative base, in both directions.
        assert!(matches!(
            pow(root(x(view), 2), Ex::int(-2), &cx),
            Ex::Pow(..)
        ));
    });
}

/// Integers beyond 128 bits are integers. The map of the 128-bit boundary found the sites
/// that read "does not fit i128" as "is not an integer", each unsound once such an integer
/// reaches it. The f64 domain knows no parity beyond 2^53, and every site must then refuse
/// (in `f64`: permissive composes powers naively, ignoring the sign).
#[test]
fn big_integers_keep_their_integer_reading() {
    with_view(|view| big_integer_sites(view, RuleMode::Default));
}

fn big_integer_sites(view: &TokenView, mode: RuleMode) {
    {
        let mut cx = Cx::bare(view);
        cx.mode = mode;
        let big = |p: num_bigint::BigInt| Rat::from_big(p, num_bigint::BigInt::from(1)).unwrap();
        let ten40 = big(num_traits::pow(num_bigint::BigInt::from(10), 40));
        let odd = ten40.checked_add(&Rat::ONE).unwrap(); // 10^40 + 1
        assert!(ten40.small_int().is_none() && ten40.is_integer());
        assert!(ten40.parity().is_none() && odd.parity().is_none());
        let rootn = view.intern("rootn");

        // 1. rootn(x, 10^40) is a root with an integer index, not the invalid operation.
        let r = fun(rootn, vec![x(view), Ex::Num(ten40.clone())], &cx);
        assert!(matches!(&r, Ex::Fun(..)), "{r:?}");

        // 2. Pow-of-pow: (x^(10^40))^(1/3) must not become x^(10^40/3). The inner power is
        //    an even power, defined and non-negative at x < 0, where x^(10^40/3) is NaN.
        let third = Rat::new(1, 3).unwrap();
        let inner = pow(x(view), Ex::Num(ten40.clone()), &cx);
        let composed = pow(inner, Ex::Num(third.clone()), &cx);
        let wrong = pow(x(view), Ex::Num(ten40.checked_mul(&third).unwrap()), &cx);
        assert_ne!(composed, wrong);

        // 3. Power through an even root: rootn(x^(2(10^40+1)), 4) must not become
        //    x^((10^40+1)/2), for the same reason.
        let k = odd.checked_mul(&Rat::int(2)).unwrap();
        let through = fun(
            rootn,
            vec![pow(x(view), Ex::Num(k.clone()), &cx), Ex::int(4)],
            &cx,
        );
        let wrong = pow(
            x(view),
            Ex::Num(k.checked_mul(&Rat::new(1, 4).unwrap()).unwrap()),
            &cx,
        );
        assert_ne!(through, wrong);
    }
}
