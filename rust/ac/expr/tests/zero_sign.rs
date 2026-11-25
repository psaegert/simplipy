//! Signed zeros.

use super::*;

#[test]
fn a_zero_keeps_its_ieee_sign_where_it_is_inverted() {
    with_view(|view| {
        let lg = |cx: &Cx| fun(view.intern("log"), vec![x(view)], cx);
        for mode in [RuleMode::Strict, RuleMode::Permissive] {
            let mut cx = Cx::bare(view);
            cx.mode = mode;
            let ieee = matches!(mode, RuleMode::Strict);
            let inverse = |members: Vec<Ex>| -> Ex {
                let raised = raised_members(members, &Rat::NEG_ONE, &cx)
                    .unwrap_or_else(|held| vec![pow(held, Ex::int(-1), &cx)]);
                mul(raised, &cx)
            };
            // 1/(x0*0) is inf/x0 in f64 (-inf wherever x0 < 0); in permissive 1/0 has no
            // value (division by zero is undefined however written)
            let want = if ieee {
                mul(vec![Ex::PosInf, pow(x(view), Ex::int(-1), &cx)], &cx)
            } else {
                Ex::NaN
            };
            assert_eq!(inverse(vec![x(view), Ex::int(0)]), want, "{mode:?}");
            // 1/(-1*0) = 1/-0 is -inf in f64
            let want = if ieee { Ex::NegInf } else { Ex::NaN };
            assert_eq!(inverse(vec![Ex::int(-1), Ex::int(0)]), want, "{mode:?}");
            // a product held whole around its zero distributes the inverse too
            let held = pow(Ex::Mul(vec![Ex::int(0), lg(&cx)]), Ex::int(-1), &cx);
            if ieee {
                let want = mul(vec![Ex::PosInf, pow(lg(&cx), Ex::int(-1), &cx)], &cx);
                assert_eq!(held, want, "{mode:?}");
            } else {
                // permissive cancels `0*log(x0)` to 0 without a definedness check, and
                // then reads 1/0 by the table: no value
                assert_eq!(held, Ex::NaN, "{mode:?}");
            }
        }
    });
}

#[test]
fn a_zero_term_stays_where_the_rest_may_be_neg_zero_on_a_region() {
    with_view(|view| {
        for mode in [RuleMode::Strict, RuleMode::Permissive] {
            let mut cx = Cx::bare(view);
            cx.mode = mode;
            let x1 = Ex::Leaf(view.intern("x1"));
            let ab = fun(view.intern("abs"), vec![x(view)], &cx);
            // t = x1*(x0 - |x0|): -0 wherever x0 >= 0 and x1 < 0, a region
            let s = add(vec![x(view), mul(vec![Ex::int(-1), ab], &cx)], &cx);
            let t = mul(vec![x1.clone(), s], &cx);
            let plus_zero = add(vec![t.clone(), Ex::int(0)], &cx);
            // 2t - t with both terms over the one key (a product holding the sum: `mul`
            // would turn the negated sum around, a class in its other orientation)
            let scaled = |c: i128| match &t {
                Ex::Mul(v) => Ex::Mul([vec![Ex::int(c)], v.clone()].concat()),
                other => panic!("a product expected: {other:?}"),
            };
            let collected = add(vec![scaled(2), scaled(-1)], &cx);
            if matches!(mode, RuleMode::Strict) {
                // t + 0 and 2t - t are +0 there, as float64 sums them
                let kept = Ex::Add(vec![t.clone(), Ex::int(0)]);
                assert_eq!(plus_zero, kept);
                assert_eq!(collected, kept);
                // the sum keeps its orientation: -(t + 0) is no sum of negated terms
                assert_eq!(negate_term(&Ex::int(0), &cx), None);
                // the class in its two orientations does not collect where it may vanish
                let turned = add(vec![t.clone(), mul(vec![Ex::int(-1), t.clone()], &cx)], &cx);
                assert!(matches!(&turned, Ex::Add(v) if v.len() == 2), "{turned:?}");
            } else {
                assert_eq!(plus_zero, t, "{mode:?}");
                assert_eq!(collected, t, "{mode:?}");
            }
            // x0 + 0 is x0 in every mode: its zero set is null
            assert_eq!(add(vec![x(view), Ex::int(0)], &cx), x(view));
            // x1 + t + 0 is x1 + t: x1 is -0 only at a point
            let sum = add(vec![x1.clone(), t.clone()], &cx);
            assert_eq!(add(vec![sum.clone(), Ex::int(0)], &cx), sum, "{mode:?}");
        }
    });
}
