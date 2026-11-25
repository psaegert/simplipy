//! Sign placement and the orientation of classes.

use super::*;

#[test]
fn a_constant_absorbs_the_orientation_sign_of_every_class() {
    with_view(|view| {
        let sin = view.intern("sin");
        for mode in [RuleMode::Strict, RuleMode::Permissive] {
            let mut cx = Cx::bare(view);
            cx.mode = mode;
            let neg = |a: Ex| mul(vec![Ex::int(-1), a], &cx);
            // A sum holding a `<constant>` takes a sign into its terms: -(x0 + x1^c) is
            // -x0 - x1^c.
            let s = add(vec![x(view), pow(y(view), Ex::ANON, &cx)], &cx);
            let neg_s = neg(s.clone());
            assert!(matches!(&neg_s, Ex::Add(_)), "{mode:?}: {neg_s:?}");
            // Beside a bare `<constant>`, which absorbs every constant factor of its
            // product, that sign is the constant's, whatever route built the product, and the
            // sum takes the orientation with the fewer signs: c*-(x0 + x1^c) is c*(x0 + x1^c).
            let want = mul(vec![Ex::ANON, s.clone()], &cx);
            assert_eq!(want, Ex::Mul(vec![Ex::ANON, s.clone()]), "{mode:?}");
            assert_eq!(mul(vec![neg_s.clone(), Ex::ANON], &cx), want, "{mode:?}");
            assert_eq!(
                mul(vec![Ex::ANON, Ex::int(-1), s.clone()], &cx),
                want,
                "{mode:?}"
            );
            // So does an odd function of one: c*sin(c - x0) is c*sin(x0 + c).
            let t = add(vec![x(view), Ex::ANON], &cx);
            assert_eq!(
                mul(vec![Ex::ANON, fun(sin, vec![neg(t.clone())], &cx)], &cx),
                mul(vec![Ex::ANON, fun(sin, vec![t], &cx)], &cx),
                "{mode:?}"
            );
            // Without a bare `<constant>` the sum keeps the sign the product gave it.
            let kept = mul(vec![y(view), neg_s.clone()], &cx);
            assert!(
                matches!(&kept, Ex::Mul(v) if v.contains(&neg_s)),
                "{mode:?}: {kept:?}"
            );
        }
    });
}

#[test]
fn a_lone_site_takes_the_orientation_of_a_products_one_factor() {
    with_view(|view| {
        let sin = view.intern("sin");
        let z = || Ex::Leaf(view.intern("x2"));
        let half = Ex::Num(Rat::new(1, 2).unwrap());
        for mode in [RuleMode::Strict, RuleMode::Permissive] {
            let mut cx = Cx::bare(view);
            cx.mode = mode;
            let neg = |a: Ex| mul(vec![Ex::int(-1), a], &cx);
            // x0 - x1 - x2: two signs; -(x1 + x2 - x0) has two too, the outer one
            let s = add(vec![x(view), neg(y(view)), neg(z())], &cx);
            let raw_cube = Ex::Pow(Box::new(s.clone()), Box::new(Ex::int(3)));
            let raw_sin = Ex::Fun(sin, vec![s.clone()]);
            // alone, as the one factor of a product: (x0 - x1 - x2)^3 is
            // -(x1 + x2 - x0)^3, and sin(x0 - x1 - x2) is -sin(x1 + x2 - x0)
            let cube = pow(s.clone(), Ex::int(3), &cx);
            assert_eq!(
                cube,
                mul(vec![Ex::int(1), raw_cube.clone()], &cx),
                "{mode:?}"
            );
            assert!(
                matches!(&cube, Ex::Mul(v) if v.contains(&Ex::int(-1))),
                "{mode:?}: {cube:?}"
            );
            let sine = fun(sin, vec![s.clone()], &cx);
            assert_eq!(
                sine,
                mul(vec![Ex::int(1), raw_sin.clone()], &cx),
                "{mode:?}"
            );
            // a power over a lone site reads its class, whichever orientation it stands in
            assert_eq!(
                pow(cube.clone(), Ex::int(-1), &cx),
                pow(s.clone(), Ex::int(-3), &cx),
                "{mode:?}"
            );
            assert_eq!(
                pow(cube, half.clone(), &cx),
                pow(s.clone(), Ex::Num(Rat::new(3, 2).unwrap()), &cx),
                "{mode:?}"
            );
            assert_eq!(
                pow(sine, Ex::int(-1), &cx),
                pow(raw_sin, Ex::int(-1), &cx),
                "{mode:?}"
            );
        }
    });
}

/// ONE owner for the odd-function literal sign, on every construction
/// route. Before the hoist, `term_join` (the Add-collector's rebuild) fused
/// `-5 * sin(2)` to `5 * sin(-2)` while `mul()`'s direct assembly did not -- two
/// fixpoints for one value, differing by construction history (the invariance
/// defect class). And the collector could not SPLIT the sign back out, so
/// `sin(-2) + sin(2)` needed a render/re-parse cycle to cancel.
#[test]
fn odd_literal_sign_is_route_independent() {
    with_view(|view| {
        let cx = Cx::bare(view);
        let sin = view.intern("sin");
        let sin2 = fun(sin, vec![Ex::int(2)], &cx);
        let sin_neg2 = fun(sin, vec![Ex::int(-2)], &cx);

        // The two routes must agree, and on the ruled direction (the
        // literal owns the sign): -5 * sin(2) spells 5 * sin(-2).
        let via_mul = mul(vec![Ex::int(-5), sin2.clone()], &cx);
        let via_join = term_join(Coef::of(&Rat::int(-5)), sin2.clone(), &cx);
        assert_eq!(
            via_mul, via_join,
            "mul() and term_join disagree on the sign"
        );
        assert_eq!(
            via_mul,
            mul(vec![Ex::int(5), sin_neg2.clone()], &cx),
            "the fused spelling is not the canonical one"
        );
        // The other example: -1 * sin(2) files as sin(-2) (the Mul node itself
        // dissolves).
        assert_eq!(mul(vec![Ex::int(-1), sin2.clone()], &cx), sin_neg2);

        // The collector conflates the pair IN ONE PASS -- sin(-2)
        // and sin(2) share a key with opposite coefficients and cancel exactly.
        assert_eq!(
            add(vec![sin_neg2.clone(), sin2.clone()], &cx),
            Ex::int(0),
            "sin(-2) + sin(2) must cancel at collection, not via re-parse"
        );

        // The collision guard: flipping sin(-2) beside sin(2) would mint an
        // unmerged duplicate-base bag; the orbit must skip such masks and keep
        // the entry spelling stable (idempotent).
        let prod = mul(vec![sin_neg2.clone(), sin2.clone()], &cx);
        assert_eq!(canon(prod.clone(), &cx), prod, "ground product not stable");
    });
}

/// The sign-placement tie-break is a function of CONTENT, never of token
/// interning order. The mu-tied orbit of `-1 * (a-b) * (c-d)` (its two mirror
/// pairings price alike) must land on the same resolved spelling
/// when the same four opaque literals are interned in OPPOSITE orders and the
/// factors arrive swapped -- an older `ex_struct_cmp` compared `Leaf`/`Fun`
/// tokens by raw id and flipped the mirror decision with the interning (extreme
/// fuzz rows 829655/866090, the two regressions at 1M rows).
#[test]
fn f72_tiebreak_is_interning_independent() {
    fn build(intern_first: &[&str], swap_factors: bool) -> String {
        with_view(|view| {
            for s in intern_first {
                view.intern(s);
            }
            let cx = Cx::bare(view);
            let leaf = |s: &str| Ex::Leaf(view.intern(s));
            let neg = |e: Ex, cx: &Cx| mul(vec![Ex::int(-1), e], cx);
            let ab = add(vec![leaf("lit_a"), neg(leaf("lit_b"), &cx)], &cx);
            let cd = add(vec![leaf("lit_c"), neg(leaf("lit_d"), &cx)], &cx);
            let items = if swap_factors {
                vec![Ex::int(-1), cd, ab]
            } else {
                vec![Ex::int(-1), ab, cd]
            };
            ex_str(&mul(items, &cx), view)
        })
    }
    let base = build(&["lit_a", "lit_b", "lit_c", "lit_d"], false);
    assert_eq!(base, build(&["lit_d", "lit_c", "lit_b", "lit_a"], false));
    assert_eq!(base, build(&["lit_c", "lit_d", "lit_a", "lit_b"], true));
    assert_eq!(base, build(&["lit_b", "lit_a", "lit_d", "lit_c"], true));
}

/// With the rational content PARTITIONED across several `Num` members (the
/// product is not representable), the sign has exactly ONE canonical host, a function
/// of the value -- not of which member carried it on arrival. Before the
/// sign-factored accumulation, `-N * P` kept the sign on the `-N` partial while
/// the negation of `N * P` hosted it on the coefficient slot: two stable states
/// for one value (the extreme lane's idempotence family, 88 -> 30 rows).
#[test]
fn f72_partition_sign_host_is_route_invariant() {
    with_view(|view| {
        let cx = Cx::bare(view);
        // 0.9999999999999999 and 1e-300: the product's denominator 10^316 exceeds the
        // f64 domain's 2^1022 (phase 2c; it used to be an i128 overflow), so the bag keeps
        // BOTH as members.
        let n = Rat::new(9999999999999999, 10000000000000000).unwrap();
        let p = Rat::from_big(
            num_bigint::BigInt::from(1),
            num_traits::pow(num_bigint::BigInt::from(10), 300),
        )
        .unwrap();
        let route_signed_member = mul(
            vec![Ex::Num(n.checked_neg().unwrap()), Ex::Num(p.clone())],
            &cx,
        );
        let route_negated_bag =
            negate_term(&mul(vec![Ex::Num(n.clone()), Ex::Num(p.clone())], &cx), &cx).unwrap();
        let route_other_host =
            term_join(Coef::of(&p.checked_neg().unwrap()), Ex::Num(n.clone()), &cx);
        let route_swapped = mul(vec![Ex::Num(p), Ex::Num(n.checked_neg().unwrap())], &cx);
        assert_eq!(route_signed_member, route_negated_bag);
        assert_eq!(route_signed_member, route_other_host);
        assert_eq!(route_signed_member, route_swapped);
        // The chosen state is a fixpoint of its own constructor.
        let Ex::Mul(v) = route_signed_member.clone() else {
            panic!("partition bag expected, got {route_signed_member:?}");
        };
        assert_eq!(mul(v, &cx), route_signed_member);
    });
}

#[test]
fn an_odd_head_of_nan_trades_no_sign_and_an_even_head_over_it_terminates() {
    with_view(|view| {
        let cx = Cx::bare(view);
        let sin_nan = fun(view.intern("sin"), vec![Ex::NaN], &cx);
        // NaN is no negatable argument: a negation leaves it as it is, so the odd head has
        // no other member in its class and no sign trades into it.
        assert_eq!(multiple_literal(&Ex::NaN), None);
        assert_eq!(sign_trade_flip(&sin_nan, &cx), None);
        assert_eq!(
            mul(vec![Ex::int(-2), sin_nan.clone()], &cx),
            Ex::Mul(vec![Ex::int(-2), sin_nan.clone()])
        );
        // `sin(nan) / -inf` beside the written negative zero, under each even head: the arm
        // drops the sign and does not move again.
        let neg_zero = Ex::Leaf(view.intern(NEG_ZERO));
        for even in ["abs", "cos", "cosh"] {
            for coefficient in [1, -1] {
                let arg = mul(
                    vec![Ex::int(coefficient), neg_zero.clone(), sin_nan.clone()],
                    &cx,
                );
                let built = fun(view.intern(even), vec![arg], &cx);
                let Ex::Fun(_, a) = &built else {
                    panic!("not a call: {built:?}")
                };
                assert_eq!(even_argument(&a[0], &cx), None, "{built:?}");
            }
        }
    });
}
