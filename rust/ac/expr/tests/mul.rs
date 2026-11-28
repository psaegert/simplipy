//! The product constructor.

use super::*;

#[test]
fn mul_collects_exponents() {
    with_view(|view| {
        let cx = Cx::bare(view);
        // x * x -> x^2
        let e = mul(vec![x(view), x(view)], &cx);
        assert_eq!(e, Ex::Pow(Box::new(x(view)), Box::new(Ex::int(2))));
        // x^2 * x^-1 -> x (variable base: finite-nonzero a.e. licence is syntactic)
        let x2 = pow(x(view), Ex::int(2), &cx);
        let xm1 = pow(x(view), Ex::int(-1), &cx);
        assert_eq!(mul(vec![x2, xm1], &cx), x(view));
        // x * 1/x -> 1 (full cancel on a variable)
        let xm1 = pow(x(view), Ex::int(-1), &cx);
        assert_eq!(mul(vec![x(view), xm1], &cx), Ex::int(1));
        // log(x)^1 * log(x)^-1: composite base, no certificate -> refuses.
        let lg = fun(view.intern("log"), vec![x(view)], &cx);
        let lgm1 = pow(lg.clone(), Ex::int(-1), &cx);
        let e = mul(vec![lg.clone(), lgm1], &cx);
        assert!(
            matches!(e, Ex::Mul(_)),
            "uncertified exponent cancel: {e:?}"
        );
        // 2 * 3 * x -> 6x; coefficient exact.
        let e = mul(vec![Ex::int(2), Ex::int(3), x(view)], &cx);
        assert_eq!(e, Ex::Mul(vec![Ex::int(6), x(view)]));
        // Zero: 0 * x^2 -> 0 (finite and never negative); 0 * x keeps x in `f64`, whose sign
        // is the zero's (float64's 0*x is -0 for x < 0); 0 * log(x) stays.
        let sq = pow(x(view), Ex::int(2), &cx);
        assert_eq!(mul(vec![Ex::int(0), sq], &cx), Ex::int(0));
        assert_eq!(
            mul(vec![Ex::int(0), x(view)], &cx),
            Ex::Mul(vec![Ex::int(0), x(view)])
        );
        let e = mul(vec![Ex::int(0), lg.clone()], &cx);
        assert_eq!(e, Ex::Mul(vec![Ex::int(0), lg]));
        // ... and in `permissive`, which reads one zero, 0 * x -> 0.
        let mut lossy = Cx::bare(view);
        lossy.mode = RuleMode::Permissive;
        assert_eq!(mul(vec![Ex::int(0), x(view)], &lossy), Ex::int(0));
    });
}

#[test]
fn powers_of_e_collect_by_their_exponents() {
    with_view(|view| {
        let exp = view.intern("exp");
        let y = || Ex::Leaf(view.intern("x1"));
        for mode in [RuleMode::Strict, RuleMode::Permissive] {
            let mut cx = Cx::bare(view);
            cx.mode = mode;
            let e = |a: Ex| fun(exp, vec![a], &cx);
            let neg = |a: Ex| mul(vec![Ex::int(-1), a], &cx);
            // exp(a) exp(b) = exp(a + b)
            assert_eq!(
                mul(vec![e(x(view)), e(y())], &cx),
                e(add(vec![x(view), y()], &cx)),
                "{mode:?}"
            );
            // exp(u) exp(-u) = exp(0) = 1, u finite almost everywhere
            assert_eq!(mul(vec![e(x(view)), e(neg(x(view)))], &cx), Ex::int(1));
            // e is e^1, and a rational power of an exp is a power of e
            assert_eq!(
                mul(vec![Ex::E, e(x(view))], &cx),
                e(add(vec![x(view), Ex::int(1)], &cx))
            );
            let sq = pow(e(x(view)), Ex::int(2), &cx);
            assert_eq!(
                mul(vec![sq.clone(), e(y())], &cx),
                e(add(vec![mul(vec![Ex::int(2), x(view)], &cx), y()], &cx))
            );
            // a quotient of powers of e, inside a wider product
            let q = mul(vec![y(), e(x(view)), pow(e(y()), Ex::int(-1), &cx)], &cx);
            assert_eq!(q, mul(vec![y(), e(add(vec![x(view), neg(y())], &cx))], &cx));
            // f64's exp(800) exp(-800) is inf * 0 = NaN, an overflow, where the true value
            // is given: exp(0) = 1
            assert_eq!(
                mul(vec![e(Ex::int(800)), e(Ex::int(-800))], &cx),
                Ex::int(1)
            );
            // a lone power of e stays, and one base's powers collect over that base as
            // `pow` spells them: exp(x) exp(x) is exp(x)^2
            assert_eq!(
                mul(vec![sq.clone(), Ex::int(3)], &cx),
                Ex::Mul(vec![Ex::int(3), sq.clone()])
            );
            assert_eq!(mul(vec![e(x(view)), e(x(view))], &cx), sq);
        }
    });
}

#[test]
fn mul_branch_cut_gate() {
    with_view(|view| {
        let cx = Cx::bare(view);
        let half = Ex::Num(Rat::new(1, 2).unwrap());
        // x^(1/2) * x^(1/2) must NOT become x (wrong on x < 0). The PROPERTY is what
        // matters, not the shape: with even roots spelled `rootn`, equal factors now
        // collect into `pow(rootn(x,2), 2)`, which is value-exact (NaN on x < 0 both
        // sides) and judge-OK. Asserting "still a Mul" was a proxy that only held while
        // the sound merge did not exist.
        let s = pow(x(view), half.clone(), &cx);
        let e = mul(vec![s.clone(), s.clone()], &cx);
        assert_ne!(e, x(view), "branch-cut merge must refuse: {e:?}");
        // abs(x)^(1/2) * abs(x)^(1/2) -> abs(x): certainly non-negative base.
        let ax = fun(view.intern("abs"), vec![x(view)], &cx);
        let s = pow(ax.clone(), half.clone(), &cx);
        assert_eq!(mul(vec![s.clone(), s], &cx), ax);
        // x^(1/2) * x^1 -> x^(3/2): sum stays non-integer, sound. `factor_split` reads
        // the even root as `x^(1/2)`, so the root spelling does not cost the merge.
        let s = pow(x(view), half, &cx);
        let e = mul(vec![s, x(view)], &cx);
        assert_eq!(
            e,
            Ex::Pow(
                Box::new(x(view)),
                Box::new(Ex::Num(Rat::new(3, 2).unwrap()))
            )
        );
    });
}

#[test]
fn abs_elimination_on_certainly_nonneg() {
    with_view(|view| {
        let cx = Cx::bare(view);
        let abs_t = view.intern("abs");
        // |x^2| -> x^2 and |x^-2| -> x^-2: even exponents of EITHER sign (the
        // old serve-order shadow lived exactly at the -2 spelling).
        for n in [2, -2] {
            let p = pow(x(view), Ex::int(n), &cx);
            assert_eq!(fun(abs_t, vec![p.clone()], &cx), p);
        }
        // |1/|x|| -> 1/|x|: certainly-non-negative base under ANY exponent.
        let ax = fun(abs_t, vec![x(view)], &cx);
        let inv_ax = pow(ax.clone(), Ex::int(-1), &cx);
        assert_eq!(fun(abs_t, vec![inv_ax.clone()], &cx), inv_ax);
        // |cos(cos x)| -> cos(cos x): range positivity via the unit-bounded argument.
        let cos_t = view.intern("cos");
        let cc = fun(cos_t, vec![fun(cos_t, vec![x(view)], &cx)], &cx);
        assert_eq!(fun(abs_t, vec![cc.clone()], &cx), cc);
        // `|sinh |x||` settles at `|sinh x|`, NOT at `sinh |x|`. Both are
        // the same function and both price 24,000 -- the abs elimination this test
        // guards is not what changed. What changed is which representative wins: the
        // parity arm reaches the argument FIRST and strips the inner `abs` (an `abs`
        // inside a sign-blind context is a no-op), leaving `sinh x`, whose sign is
        // indefinite, so the OUTER abs correctly stays. Before, the inner `abs` survived
        // and made the argument certainly-non-negative, so the outer one dropped
        // instead. Same node count, same mu, one canonical form rather than two.
        let sh_inner = fun(view.intern("sinh"), vec![ax.clone()], &cx);
        let sh_outer = fun(
            abs_t,
            vec![fun(view.intern("sinh"), vec![x(view)], &cx)],
            &cx,
        );
        assert_eq!(fun(abs_t, vec![sh_inner], &cx), sh_outer);
        // Bag closure: a product of non-negatives sheds the outer abs.
        let prod = mul(vec![pow(x(view), Ex::int(2), &cx), ax.clone()], &cx);
        assert_eq!(fun(abs_t, vec![prod.clone()], &cx), prod);
        // Controls: sign-indefinite operands KEEP the abs. (`abs Const` folding is the
        // Const channel's absorption in the pipeline, deliberately NOT this fold --
        // certainly_nonneg(Const) is false because a fitted value can be negative.)
        for keep in [
            x(view),
            fun(view.intern("sin"), vec![x(view)], &cx),
            fun(cos_t, vec![x(view)], &cx), // cos of an UNBOUNDED argument
            pow(x(view), Ex::int(3), &cx),
            // periodic, not monotone -- so `certainly_nonneg` refuses and the abs
            // stays. The argument is a BARE variable: it used to be
            // `|x|`, and the parity arm now strips that inner `abs` (a no-op inside a
            // sign-blind context), which was testing the parity arm rather than this
            // control's subject.
            fun(view.intern("tan"), vec![x(view)], &cx),
            Ex::ANON,
        ] {
            let e = fun(abs_t, vec![keep.clone()], &cx);
            assert_eq!(e, Ex::Fun(abs_t, vec![keep]), "abs must stay");
        }
    });
}

#[test]
fn mul_extended_and_const() {
    with_view(|view| {
        let cx = Cx::bare(view);
        assert_eq!(mul(vec![Ex::NaN, x(view)], &cx), Ex::NaN);
        assert_eq!(mul(vec![Ex::int(0), Ex::PosInf], &cx), Ex::NaN);
        assert_eq!(mul(vec![Ex::int(-2), Ex::PosInf], &cx), Ex::NegInf);
        assert_eq!(mul(vec![Ex::NegInf, Ex::NegInf], &cx), Ex::PosInf);
        // C * C -> C; C * 5 -> C; C * 0 keeps C in `f64`, whose sign is the zero's, and is 0
        // in `permissive`.
        assert_eq!(mul(vec![Ex::ANON, Ex::ANON], &cx), Ex::ANON);
        assert_eq!(mul(vec![Ex::ANON, Ex::int(5)], &cx), Ex::ANON);
        assert_eq!(
            mul(vec![Ex::ANON, Ex::int(0)], &cx),
            Ex::Mul(vec![Ex::int(0), Ex::ANON])
        );
        let mut lossy = Cx::bare(view);
        lossy.mode = RuleMode::Permissive;
        assert_eq!(mul(vec![Ex::ANON, Ex::int(0)], &lossy), Ex::int(0));
        // inf * C: C = 0 reachable -> keep both factors.
        let e = mul(vec![Ex::PosInf, Ex::ANON], &cx);
        assert_eq!(e, Ex::Mul(vec![Ex::PosInf, Ex::ANON]));
    });
}
