//! The measure.

use super::*;

/// ONE PRICE FOR A SIGN: one bit wherever it sits -- on a literal, on a product's
/// coefficient beside other factors, or a negation standing alone (a sign, not a product,
/// so no `Mul` head). `+1` is no sign, and a reciprocal is division, not a sign.
#[test]
fn a_sign_costs_one_bit_wherever_it_sits() {
    with_view(|view| {
        let cx = Cx::bare(view);
        let c = |e: &Ex| super::complexity(e, view);
        let (x_, y_) = (x(view), y(view));
        let neg = |e: Ex| mul(vec![Ex::int(-1), e], &cx);
        let sin = |e: Ex| fun(view.intern("sin"), vec![e], &cx);
        // a negation standing alone: its operand and one bit
        assert_eq!(c(&neg(x_.clone())), c(&x_) + 1_000);
        assert_eq!(c(&neg(x_.clone())), 7_000);
        // the term of a difference
        let diff = add(vec![x_.clone(), neg(y_.clone())], &cx);
        let sum = add(vec![x_.clone(), y_.clone()], &cx);
        assert_eq!(c(&diff), c(&sum) + 1_000);
        // the -1 coefficient of a product beside other factors
        let prod = mul(vec![x_.clone(), y_.clone()], &cx);
        assert_eq!(c(&neg(prod.clone())), c(&prod) + 1_000);
        // the sign of a literal, a coefficient and an exponent: unchanged, one bit each
        assert_eq!(c(&Ex::int(-2)), c(&Ex::int(2)) + 1_000);
        let three_x = mul(vec![Ex::int(3), x_.clone()], &cx);
        let minus_three_x = mul(vec![Ex::int(-3), x_.clone()], &cx);
        assert_eq!(c(&minus_three_x), c(&three_x) + 1_000);
        assert_eq!(
            c(&pow(x_.clone(), Ex::int(-2), &cx)),
            c(&pow(x_.clone(), Ex::int(2), &cx)) + 1_000
        );
        // a reciprocal is division, not a sign
        assert_eq!(c(&pow(x_.clone(), Ex::int(-1), &cx)), 9_000);
        // so the two placements of a constant's sign cost the same, and the sign key
        // decides: the sign stays on the constant's literal
        let outside = Ex::Mul(vec![
            Ex::int(-1),
            x_.clone(),
            Ex::Fun(view.intern("sin"), vec![Ex::int(3)]),
        ]);
        let inside = Ex::Mul(vec![
            x_.clone(),
            Ex::Fun(view.intern("sin"), vec![Ex::int(-3)]),
        ]);
        assert_eq!(c(&outside), c(&inside));
        let built = mul(vec![Ex::int(-1), x_.clone(), sin(Ex::int(3))], &cx);
        assert_eq!(built, inside);
    });
}

#[test]
fn complexity_weight_table() {
    // The mu weight table in MILLI-BITS, at the SYMBOL TABLE: a leaf 6
    // bits, Add/Mul/Pow 3, Pi/E 4, an elementary head 6, a transcendental head 8,
    // an infinity 8 -- where mu' charged a flat 8 for every one of them.
    // Unchanged by the table: a priced literal pays ONE selector bit + its cheaper codeword
    // (rational L(p)[+L(q)] vs decimal-scientific max(2,L(m))+L(|k|)) + a bit for a
    // negative sign, mantissa floored at two bits; magnitude-1 coefficient and
    // exponent slots stay free (a bare sign writes no number, so no selector
    // either); <constant> = MU_FREE_WORST_CASE_F64 = 67 bits (the sup-construction
    // re-derived under the mu' codebook -- DERIVED, not chosen).
    with_view(|view| {
        let cx = Cx::bare(view);
        let c = |e: &Ex| super::complexity(e, view);
        // x - y = 16: Add(3) + x(6) + the negation of y (a sign, one bit, and no Mul
        // head) + y(6).
        // x*y = 15; x/y = 18 (the Pow(y, -1) wrapper at 3, exponent -1 free).
        let x_ = x(view);
        let y_ = y(view);
        let x_minus_y = add(
            vec![x_.clone(), mul(vec![Ex::int(-1), y_.clone()], &cx)],
            &cx,
        );
        assert_eq!(c(&x_minus_y), 16_000);
        assert_eq!(c(&mul(vec![x_.clone(), y_.clone()], &cx)), 15_000);
        let x_over_y = mul(vec![x_.clone(), pow(y_.clone(), Ex::int(-1), &cx)], &cx);
        assert_eq!(c(&x_over_y), 18_000);
        // 2x = 11.585 (the coefficient PAYS selector + its bits: 3 + 2.585 + 6),
        // x^2 = 11.585, 1/x = 9, sin(x) = 12. The literal floor is ONE bit,
        // so cost(2) = 2.585 clears it and no longer ties cost(3).
        assert_eq!(c(&mul(vec![Ex::int(2), x_.clone()], &cx)), 11_585);
        assert_eq!(c(&pow(x_.clone(), Ex::int(2), &cx)), 11_585);
        // THE EXPONENTS ARE ORDERED AGAIN. Under the two-bit floor `x^2`, `x^3` and
        // `x^4` all priced alike, because the clamp swallowed cost(2), cost(3) and
        // cost(4) together -- the measure could not tell a square from a cube.
        assert!(
            c(&pow(x_.clone(), Ex::int(2), &cx)) < c(&pow(x_.clone(), Ex::int(3), &cx))
                && c(&pow(x_.clone(), Ex::int(3), &cx)) < c(&pow(x_.clone(), Ex::int(4), &cx))
        );
        assert_eq!(c(&pow(x_.clone(), Ex::int(-1), &cx)), 9_000);
        assert_eq!(c(&fun(view.intern("sin"), vec![x_.clone()], &cx)), 12_000);
        // x^-2 costs as x^2 plus the FULL sign bit (the -2 exponent is a priced
        // literal: selector + L(2) + sign = 1 + 1.585 + 1, all of it clear of the
        // one-bit floor; under the two-bit floor the clamp absorbed all but 0.585).
        assert_eq!(c(&pow(x_.clone(), Ex::int(-2), &cx)), 12_585);
        // Mul(3) + <constant>(67_000) + x(6) = 76_000. The free constant is a LITERAL
        // price, so the table does not touch it: it now outweighs its own structure by
        // more than it did, which is the direction the ordering wants.
        assert_eq!(
            c(&mul(vec![Ex::ANON, x_.clone()], &cx)),
            mu_mul() + MU_FREE_WORST_CASE_F64 + mu_leaf()
        );
        assert_eq!(mu_free(), 67 * MU_MILLI);
        assert_eq!(c(&Ex::int(-5)), 4_585); // selector + L(5) + 1 for the sign
                                            // x8 + 1.2*x3 = 3 + 6 + (3 + cost(6/5) + 6).
                                            // cost(6/5) = selector + min(L(6)+L(5) = 5.392, max(2,L(12))+L(1) = 4.700)
                                            // = 5.700: the decimal-scientific codeword takes it (the fraction-only
                                            // rule is gone).
        let t = mul(vec![Ex::Num(Rat::new(6, 5).unwrap()), y_.clone()], &cx);
        assert_eq!(c(&add(vec![x_.clone(), t], &cx)), 23_700);
        // The stage-2 point, in one line: an E factor undercuts its f64 image --
        // and by MORE than before, since the table cheapened the structure the
        // symbolic spelling is made of and left the image's 56.272 bits alone.
        let sym = c(&mul(vec![Ex::E, x_.clone()], &cx));
        let mat = c(&mul(
            vec![
                Ex::Num(Rat::new(2718281828459045, 1000000000000000).unwrap()),
                x_.clone(),
            ],
            &cx,
        ));
        assert!(sym == 13_000 && sym < mat, "{sym} vs {mat}");
    });
}

/// PIN: `mu_numeric_str` reads fraction-shaped beyond-`Rat`
/// spellings through the component model (L(p) + L(q) from the digit strings),
/// not through the decimal path with the slash counted as a significand digit.
/// The boundary properties that matter: far above every in-range literal, monotone
/// in either component's digit count, and the decimal path untouched.
#[test]
fn mu_numeric_str_fraction_component_model() {
    let big41 = "1".repeat(41); // beyond i128 (39 digits max)
    let frac = format!("{big41}/7");
    let c = mu_numeric_str(&frac);
    // Component model, both components PAID (the implicit-denominator discount applies
    // only when there is no genuine denominator), plus the mu' selector bit; the
    // denominator 7 offers no decimal codeword: 1 + L(111...1) + L(7) = 137.031 bits.
    assert_eq!(c, 137_031);
    // Far above any in-range literal; the boundary that matters is the ~105-BIT
    // 52-bit dyadics.
    assert!(c > 105 * MU_MILLI);
    // Monotone in each component.
    assert!(mu_numeric_str(&format!("{big41}1/7")) > c);
    assert!(mu_numeric_str(&format!("{big41}/71")) > c);
    // The sign costs exactly one bit here, as it does in range;
    // a zero numerator floors at the one-bit codeword floor (+ the selector) --
    // the one-bit floor, mirrored across the i128 boundary.
    assert_eq!(mu_numeric_str(&format!("-{big41}/7")), c + MU_MILLI);
    assert_eq!(mu_numeric_str("0/7"), 2 * MU_MILLI);
    // The decimal path is case-insensitive in the exponent marker.
    assert_eq!(
        mu_numeric_str("4.159653437657682e-35"),
        mu_numeric_str("4.159653437657682E-35")
    );
    // The second codeword RESTORES the minimum over codes: `1e-40` takes its
    // decimal-scientific codeword, selector + max(floor, L(1)) + L(40) = 7.358 bits
    // under the one-bit floor, not the 133.880 the fraction-only rule charged. The boundary property this test
    // exists for -- the token pricer agrees with `mu_rat`'s rule, and the SAME value
    // prices the SAME from both spellings -- now holds EXACTLY: the fraction-only code
    // pinned a one-milli-bit seam between these two spellings (133.880 vs 133.879,
    // the rounded-L10 residue of the fraction code); both now take the decimal
    // codeword through the same `l_millibits(40)` and the seam is CLOSED.
    let e_notation = mu_numeric_str("1e-40");
    let as_fraction = mu_numeric_str(&format!("1/1{}", "0".repeat(40)));
    assert_eq!(e_notation, 7_358);
    assert_eq!(as_fraction, 7_358);
    assert_eq!(
        e_notation, as_fraction,
        "the two spellings of 1e-40 drifted apart: {e_notation} vs {as_fraction}"
    );
}

/// c_free' derivation: `MU_FREE_WORST_CASE_F64` is the CEILED sup of mu'
/// over f64 round-trip spellings plus the sign bit -- recomputed here from the
/// engine's own token pricer, so the published constant cannot drift from the code
/// that defines it. The sweep: every f64 exponent field at six mantissa patterns
/// (binade edges and mid-grid probes), plus the recorded argmax
/// `8.9002954340287245e-308` (found offline by ulp-walking every binade top in the
/// boundary decades; the analytic feasibility cap -- a shortest repr's mantissa is
/// below `10 * 2^53` and its scale within +-343 -- bounds the sup under 64.92
/// bits, so the 67-bit ceiling is robust to argmax misses).
#[test]
fn c_free_prime_derivation() {
    let price = |x: f64| -> u64 {
        let tok = format!("{x:e}");
        mu_numeric_str(&tok) - MU_MILLI // the codeword min, without the selector
    };
    let mantissas: [u64; 6] = [
        0,                     // the binade bottom (a power of two)
        1,                     // one ulp up
        0x000F_FFFF_FFFF_FFFF, // the binade top
        0x0008_0000_0000_0000,
        0x0005_5555_5555_5555,
        0x000A_AAAA_AAAA_AAAA,
    ];
    let mut sup = 0u64;
    for exp in 0..2047u64 {
        for m in mantissas {
            let x = f64::from_bits((exp << 52) | m);
            if x == 0.0 || !x.is_finite() {
                continue;
            }
            sup = sup.max(price(x));
        }
    }
    // The recorded argmax reproduces the recorded supremum, and nothing in the
    // sweep prices above it (the load-bearing safety direction).
    let rec = price(8.9002954340287245e-308);
    assert_eq!(rec, 64_649);
    assert!(
        sup <= rec,
        "the sweep beat the recorded supremum: {sup} > {rec}"
    );
    assert!(
        sup > rec - 2_000,
        "the sweep never reached the extreme region: {sup}"
    );
    // + the sign bit + the selector bit, ceiled: the published constant, re-derived.
    let derived = (rec + MU_MILLI + MU_MILLI).div_ceil(MU_MILLI) * MU_MILLI;
    assert_eq!(derived, MU_FREE_WORST_CASE_F64);
    assert_eq!(mu_free(), 67 * MU_MILLI);
}

/// The astronomic knee, both halves. (1) PRICING: the linear schedule `|scale| * log2(10)` exhausts
/// u64 at |scale| ~ 5.553e15, and saturating there collided every larger scale --
/// and every beyond-i64 exponent -- on the single price `u64::MAX`, so mu could not
/// tell `1e5553000000000000` from `1e10^30`. The plan's acceptance line is asserted
/// verbatim, then a strict ladder across the whole astronomic range in both
/// exponent signs. (2) ACCUMULATION: a `u64::MAX`-priced leaf made the tree sums in
/// `complexity()` overflow -- a debug-build panic and a release-build WRAP, and a
/// wrapped sum prices a composite below its own parts, which is the direction that
/// licenses false mu-descents. Raw `Ex` shells (no canon) exercise the sums
/// directly.
#[test]
fn astronomic_literals_stay_ordered_and_sums_never_wrap() {
    // (1) the plan's acceptance, verbatim
    assert_ne!(mu_numeric_str("1e5553000000000000"), u64::MAX);

    // strictly ordered across the old ceiling, the i64 boundary, and into
    // digit-string-only exponents; both signs ride the same schedule
    let huge = format!("1e1{}", "0".repeat(30));
    let huge_neg = format!("1e-1{}", "0".repeat(30));
    let ladder = [
        "1e308", // f64's edge: the exact linear regime
        "1e4294967296",
        "1e5553000000000000", // the old saturating_mul ceiling
        "1e9223372036854775807",
        "1e99999999999999999999", // exponent past i64: digit string only
        huge.as_str(),
    ];
    for w in ladder.windows(2) {
        assert!(
            mu_numeric_str(w[0]) < mu_numeric_str(w[1]),
            "mu not strictly increasing: mu({}) = {} !< mu({}) = {}",
            w[0],
            mu_numeric_str(w[0]),
            w[1],
            mu_numeric_str(w[1])
        );
    }
    let neg_ladder = [
        "1e-308",
        "1e-4294967296",
        "1e-5553000000000000",
        "1e-99999999999999999999",
        huge_neg.as_str(),
    ];
    for w in neg_ladder.windows(2) {
        assert!(
            mu_numeric_str(w[0]) < mu_numeric_str(w[1]),
            "mu not strictly increasing on the denominator side: {} vs {}",
            w[0],
            w[1]
        );
    }

    // (2) no wrap: a composite always prices at or above every part, and never
    // below a bare variable, whatever its leaves cost
    with_view(|view| {
        let a = Ex::Leaf(view.intern("1e5553000000000000"));
        let b = Ex::Leaf(view.intern("1e9223372036854775807"));
        let (ca, cb) = (complexity(&a, view), complexity(&b, view));
        let bare = complexity(&x(view), view);
        let shells = [
            Ex::Add(vec![a.clone(), b.clone()]),
            Ex::Mul(vec![a.clone(), b.clone()]),
            Ex::Pow(Box::new(a.clone()), Box::new(b.clone())),
            Ex::Fun(view.intern("sin"), vec![a.clone(), b.clone()]),
        ];
        for s in &shells {
            let cs = complexity(s, view);
            assert!(
                cs >= ca.max(cb),
                "composite priced below a part: {cs} < max({ca}, {cb}) on {s:?}"
            );
            assert!(cs >= bare, "composite priced below a bare variable: {cs}");
        }
        // The saturation CROSSING is no longer materializable (mu'): under
        // the two-codeword pricing every leaf is bounded by ~3.32 bits per DIGIT
        // of its own print (the scientific codeword prices the scale as an
        // integer, so the old ~1.4e13-milli-bit astronomic leaves now price a few
        // dozen bits -- this leaf is {ca} ~ 55 bits), and reaching u64::MAX would
        // take a tree whose serialization is petabytes. The old crossing pin
        // (`vec![leaf; u64::MAX / ca]`, ~1.3M members under the fraction-only code) would today demand
        // ~3e14 members and is deleted as physically unreachable; the
        // `saturating_add` accumulation stays as pure defense-in-depth. What
        // remains checkable at reachable scale: totals are exact (no wrap, no
        // saturation) and strictly monotone in member count on the largest
        // feasible astronomic bags.
        let q = 100_000usize;
        let mut prev = 0u64;
        for n in [q - 1, q, q + 1] {
            let cs = complexity(&Ex::Add(vec![a.clone(); n]), view);
            assert!(
                cs > prev,
                "the Add total failed to grow at {n} members: {prev} -> {cs}"
            );
            assert_eq!(
                cs,
                mu_add() + ca * n as u64,
                "an astronomic Add bag mispriced at {n} members"
            );
            prev = cs;
        }
        for bag in [
            Ex::Mul(vec![a.clone(); q]),
            Ex::Fun(view.intern("sin"), vec![a.clone(); q]),
        ] {
            let cs = complexity(&bag, view);
            assert!(
                cs >= ca && cs >= bare,
                "a large astronomic bag priced below its parts: {cs}"
            );
        }
    });
}

/// Zero is priced by its fraction codeword alone, at the floor: it has no decimal codeword and
/// no reciprocal.
#[test]
fn zero_has_the_fraction_codeword_alone() {
    assert_eq!(
        mu_rat_codeword_totals(&Rat::ZERO),
        (MU_RAT_FLOOR, None, None)
    );
    assert_eq!(mu_rat_exact(&Rat::ZERO), MU_MILLI + MU_RAT_FLOOR);
}

/// Every number is priced as it is written: the long exact product costs its digits, and the
/// 16-digit decimal of 1/3 costs its 16 digits.
#[test]
fn a_literal_is_priced_as_it_is_written() {
    let long = {
        let _exact = crate::ac::rat::number_domain(false);
        Rat::parse_decimal("0.0766541268471677307861497420516056347394916180597539647375669841")
            .unwrap()
    };
    let short = Rat::shortest_reading_as(long.to_f64_nearest()).unwrap();
    assert_eq!(mu_rat(&long), mu_rat_exact(&long));
    assert!(mu_rat(&short) < mu_rat(&long));
    let third = Rat::parse_decimal("0.3333333333333333").unwrap();
    assert_eq!(mu_rat(&third), mu_rat_exact(&third));
    assert!(mu_rat(&Rat::new(1, 3).unwrap()) < mu_rat(&third));
    for t in ["1000", "0.2", "-7.5", "1e-40", "2.177697277405848"] {
        let r = Rat::parse_decimal(t).unwrap();
        assert_eq!(mu_rat(&r), mu_rat_exact(&r), "{t}");
    }
}

/// A literal held in pieces costs at most what its value costs as one number, alone or
/// beside other factors.
#[test]
fn a_literal_held_in_pieces_costs_at_most_its_value() {
    with_view(|view| {
        let cx = Cx::bare(view); // f64's number domain
        let b = Rat::parse_decimal("9.876543210987654e-10").unwrap();
        let inv_b = match pow(Ex::Num(b.clone()), Ex::int(-1), &cx) {
            Ex::Num(r) => r,
            other => panic!("not a number: {other:?}"),
        };
        let exact = {
            let _exact = crate::ac::rat::number_domain(false);
            Rat::parse_decimal("1.2345678901234567e-305")
                .unwrap()
                .checked_mul(&inv_b)
                .unwrap()
        };
        // a's denominator, 10^321, passes 2^1022: a stays a leaf as written, beside 1/b
        let leaf = Ex::Leaf(view.intern("1.2345678901234567e-305"));
        let quotient = mul(
            vec![leaf.clone(), pow(Ex::Num(b.clone()), Ex::int(-1), &cx)],
            &cx,
        );
        assert_eq!(
            quotient,
            Ex::Mul(vec![Ex::Num(inv_b.clone()), leaf.clone()])
        );
        // the product's head and its pieces: the leaf as written and 1/b, priced as its
        // divisor (the reciprocal codeword), below the one exact number
        let pieces = mu_mul() + mu_numeric_str("1.2345678901234567e-305") + mu_rat(&inv_b);
        assert!(pieces < mu_rat(&exact));
        assert_eq!(complexity(&quotient, view), pieces);
        // A product deciding opposite sums beside a piece prices its candidates against
        // bounds below their prices (`decide_opposite_sums`' debug assertion).
        let s = add(
            vec![x(view), Ex::Num(Rat::parse_decimal("-9.6").unwrap())],
            &cx,
        );
        let t = mul(vec![Ex::int(-1), s.clone()], &cx);
        let p = mul(vec![Ex::int(3), leaf, s, t], &cx);
        let Ex::Mul(v) = &p else {
            panic!("not a product: {p:?}")
        };
        assert_eq!(v.iter().filter(|f| is_piece(f, view)).count(), 2, "{p:?}");
        let literal = bag_literal_value(v, false, view).unwrap();
        let rest: u64 = v
            .iter()
            .filter(|f| !is_piece(f, view))
            .map(|f| complexity(f, view))
            .sum();
        let structural: u64 = v
            .iter()
            .filter(|f| is_piece(f, view))
            .map(|f| complexity(f, view))
            .sum();
        assert_eq!(
            complexity(&p, view),
            mu_mul() + structural.min(mu_rat(&literal)) + rest
        );
    });
}
