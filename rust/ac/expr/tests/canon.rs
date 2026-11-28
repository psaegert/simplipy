//! Re-canonicalization.

use super::*;

#[test]
fn canon_is_idempotent_on_samples() {
    with_view(|view| {
        let cx = Cx::bare(view);
        let lg = fun(view.intern("log"), vec![x(view)], &cx);
        let samples = vec![
            add(vec![x(view), y(view), x(view)], &cx),
            mul(vec![Ex::int(2), x(view), lg.clone(), x(view)], &cx),
            pow(add(vec![x(view), Ex::int(1)], &cx), Ex::int(2), &cx),
            add(vec![Ex::PosInf, lg], &cx),
            mul(
                vec![Ex::int(0), fun(view.intern("log"), vec![y(view)], &cx)],
                &cx,
            ),
        ];
        for s in samples {
            assert_eq!(canon(s.clone(), &cx), s, "canon not idempotent on {s:?}");
        }
    });
}
