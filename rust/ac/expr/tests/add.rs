//! The sum constructor.

use super::*;

#[test]
fn add_collects_and_orders() {
    with_view(|view| {
        let cx = Cx::bare(view);
        // x + x -> 2x (same-sign, no licence needed)
        let e = add(vec![x(view), x(view)], &cx);
        assert_eq!(e, Ex::Mul(vec![Ex::int(2), x(view)]));
        // 2x + 3x -> 5x
        let e = add(
            vec![
                mul(vec![Ex::int(2), x(view)], &cx),
                mul(vec![Ex::int(3), x(view)], &cx),
            ],
            &cx,
        );
        assert_eq!(e, Ex::Mul(vec![Ex::int(5), x(view)]));
        // Order invariance: y + x == x + y
        let a = add(vec![y(view), x(view)], &cx);
        let b = add(vec![x(view), y(view)], &cx);
        assert_eq!(a, b);
        // Flattening: (x + y) + x -> 2x + y
        let inner = add(vec![x(view), y(view)], &cx);
        let e = add(vec![inner, x(view)], &cx);
        // Stripped term order: the 2x term (key x0) sorts before y (key x1).
        assert_eq!(
            e,
            Ex::Add(vec![Ex::Mul(vec![Ex::int(2), x(view)]), y(view)])
        );
        // Literals fold exactly: 1/2 + 1/3 = 5/6
        let e = add(
            vec![
                Ex::Num(Rat::new(1, 2).unwrap()),
                Ex::Num(Rat::new(1, 3).unwrap()),
            ],
            &cx,
        );
        assert_eq!(e, Ex::Num(Rat::new(5, 6).unwrap()));
    });
}

#[test]
fn add_sign_cancel_gate() {
    with_view(|view| {
        let cx = Cx::bare(view);
        // x - x -> 0: variables are certainly finite, licence-free.
        let neg_x = mul(vec![Ex::int(-1), x(view)], &cx);
        assert_eq!(add(vec![x(view), neg_x], &cx), Ex::int(0));
        // log(x) - log(x): NO certificate in a bare context -> refuses to cancel.
        let lg = fun(view.intern("log"), vec![x(view)], &cx);
        let neg_lg = mul(vec![Ex::int(-1), lg.clone()], &cx);
        let e = add(vec![lg.clone(), neg_lg.clone()], &cx);
        assert!(
            matches!(e, Ex::Add(_)),
            "uncertified cancel must refuse: {e:?}"
        );
        // With the certificate it cancels.
        let yes = |_: &Ex| true;
        let cx2 = Cx {
            view,
            cert_fin: Some(&yes),
            cert_finnz: None,
            cert_nzae: None,
            cert_nce: None,
            cert_region: None,
            cert_encl: None,
            mode: RuleMode::Strict,
            fold_f64: Cx::folds_for(RuleMode::Strict),
            ground_first: None,
            zero_sign_unread: false,
            flat_unread: false,
            sign_held: None,
            confined: &[],
        };
        assert_eq!(add(vec![lg.clone(), neg_lg], &cx2), Ex::int(0));
        // Same-sign merging never needs the licence: log(x) + log(x) -> 2 log(x).
        let e = add(vec![lg.clone(), lg.clone()], &cx);
        assert_eq!(e, Ex::Mul(vec![Ex::int(2), lg]));
    });
}

#[test]
fn add_extended_values() {
    with_view(|view| {
        let cx = Cx::bare(view);
        assert_eq!(add(vec![Ex::NaN, x(view)], &cx), Ex::NaN);
        assert_eq!(add(vec![Ex::PosInf, Ex::NegInf], &cx), Ex::NaN);
        assert_eq!(add(vec![Ex::PosInf, Ex::int(5)], &cx), Ex::PosInf);
        assert_eq!(add(vec![Ex::PosInf, x(view)], &cx), Ex::PosInf); // variable: finite
                                                                     // inf + log(x): log(x) can be -inf (x=0) -- refuses without certificate.
        let lg = fun(view.intern("log"), vec![x(view)], &cx);
        let e = add(vec![Ex::PosInf, lg], &cx);
        assert!(matches!(e, Ex::Add(_)));
    });
}

#[test]
fn add_const_independence_and_absorption() {
    with_view(|view| {
        let cx = Cx::bare(view);
        // C + C -> C (one constant absorbs, forall-exists).
        assert_eq!(add(vec![Ex::ANON, Ex::ANON], &cx), Ex::ANON);
        // C + 5 -> C.
        assert_eq!(add(vec![Ex::ANON, Ex::int(5)], &cx), Ex::ANON);
        // 2C + x + 3 -> C + x.
        let e = add(
            vec![mul(vec![Ex::int(2), Ex::ANON], &cx), x(view), Ex::int(3)],
            &cx,
        );
        // Constant-like terms sort LAST under the stripped order (x + C, not C + x).
        assert_eq!(e, Ex::Add(vec![x(view), Ex::ANON]));
        // C*x + C*x must NOT merge (independent constants).
        let cx_term = || Ex::Mul(vec![Ex::ANON, x(view)]);
        let e = add(vec![cx_term(), cx_term()], &cx);
        assert_eq!(e, Ex::Add(vec![cx_term(), cx_term()]));
    });
}

/// A term's coefficient is its whole literal partition (`Coef`). With only the first member
/// as the coefficient, `-N*N*x` kept a key of its own beside `N*N*x`, `N*N*x` and `N*y`
/// shared the content `N`, and a sum's content and like terms depended on which terms a
/// construction saw together: a state and the parse of its own print differed.
#[test]
fn refused_partition_is_one_coefficient() {
    for mode in [RuleMode::Strict, RuleMode::Permissive] {
        with_view(|view| {
            let mut cx = Cx::bare(view);
            cx.mode = mode;
            // 10^300 * 10^300 is beyond 2^1022: refused in the f64 domain.
            let n = Rat::from_big(
                num_traits::pow(num_bigint::BigInt::from(10), 300),
                num_bigint::BigInt::from(1),
            )
            .unwrap();
            let leaf = |s: &str| Ex::Leaf(view.intern(s));
            let (x, y, z) = (leaf("x1"), leaf("x2"), leaf("x3"));
            let nn = |e: Ex| mul(vec![Ex::Num(n.clone()), Ex::Num(n.clone()), e], &cx);
            let neg = |e: &Ex| negate_term(e, &cx).unwrap();
            let t = nn(x.clone());
            assert!(
                matches!(&t, Ex::Mul(v) if v.len() == 3),
                "partition bag expected: {t:?}"
            );
            let (c, k) = term_split(t.clone(), view);
            assert_eq!((c.rat(), &k), (None, &x));
            assert_eq!(term_split(neg(&t), view), (c.negated(), x.clone()));
            // The sign is written on the smallest member, not as a `-1` beside them, however
            // it arrives.
            let minus_n = Ex::Num(n.checked_neg().unwrap());
            let written = Ex::Mul(vec![minus_n.clone(), Ex::Num(n.clone()), x.clone()]);
            assert_eq!(neg(&t), written);
            assert_eq!(mul(vec![Ex::int(-1), t.clone()], &cx), written);
            assert_eq!(nn(mul(vec![Ex::int(-1), x.clone()], &cx)), written);
            assert_eq!(mul(vec![written.clone()], &cx), written);
            assert_eq!(
                mul(vec![Ex::Num(n.clone()), minus_n, x.clone()], &cx),
                written
            );
            // Like terms cancel, and add over their shared member: {2N, N} - {N, N} = {N, N}.
            assert_eq!(add(vec![t.clone(), neg(&t)], &cx), Ex::Num(Rat::ZERO));
            let two_t = mul(vec![Ex::int(2), t.clone()], &cx);
            assert_eq!(add(vec![two_t, neg(&t)], &cx), t);
            // The grouping does not decide: flat, a sub-sum first, and factored.
            let flat = add(vec![nn(x.clone()), nn(y.clone()), z.clone()], &cx);
            let nested = add(
                vec![z.clone(), add(vec![nn(x.clone()), nn(y.clone())], &cx)],
                &cx,
            );
            let xy = add(vec![x.clone(), y.clone()], &cx);
            let factored = add(vec![nn(xy.clone()), z.clone()], &cx);
            assert_eq!(flat, nested);
            assert_eq!(flat, factored);
            // A member of a partition is no content shared with a one-member coefficient.
            let mixed = add(
                vec![nn(x.clone()), mul(vec![Ex::Num(n.clone()), y.clone()], &cx)],
                &cx,
            );
            assert!(matches!(&mixed, Ex::Add(_)), "{mixed:?}");
            // The negated pair's content is the partition, its sign the partition's.
            let negated = add(vec![neg(&nn(x.clone())), neg(&nn(y.clone()))], &cx);
            assert_eq!(negated, neg(&nn(xy)));
        });
    }
}
