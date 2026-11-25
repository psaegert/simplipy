//! Where numbers land.

use super::*;

/// Permissive's one rounding lands on whichever of the two floats around the value prints
/// shorter: a value between 0.1 and the next float, nearer the next, lands on 0.1 (the
/// nearest, 0.10000000000000002, prints a digit more). A float's own shortest spelling never
/// moves.
#[test]
fn a_landed_literal_takes_the_shorter_of_its_two_floats() {
    let r = Rat::from_big(
        num_bigint::BigInt::from(28823037615171179u64),
        num_bigint::BigInt::from(288230376151711744u64),
    )
    .unwrap();
    assert_eq!(r.to_f64_nearest(), 0.10000000000000002);
    assert_eq!(lossy_literal(&r), Rat::parse_decimal("0.1"));
    assert_eq!(lossy_literal(&Rat::parse_decimal("0.1").unwrap()), None);
    assert_eq!(
        lossy_literal(&Rat::parse_decimal("0.10000000000000002").unwrap()),
        None
    );
}

/// Strict writes a number two literals make as the cheaper of its exact value and float64's
/// value of the operation, the exact one at an equal price.
#[test]
fn strict_writes_the_cheaper_of_the_exact_and_the_float64_number() {
    let d = |t: &str| Rat::parse_decimal(t).unwrap();
    with_view(|view| {
        // the simplify walk's context
        let mut cx = Cx::bare(view);
        cx.fold_f64 = true;
        let prod = |v: Vec<Rat>| literal_partition_product(v, false, &cx);
        let sum = |v: Vec<Rat>| literal_partition_sum(v, false, &cx);
        let a = d("0.30000000000000004");
        assert_eq!(prod(vec![a.clone(), a]), vec![d("0.09000000000000002")]);
        assert_eq!(
            prod(vec![d("0.7234567890123457"), d("1.3821548250604163")]),
            vec![d("0.9999292916561291")],
            "float64's product of the two doubles, not the double nearest the exact product"
        );
        assert_eq!(sum(vec![d("0.1"), d("0.2")]), vec![d("0.3")]);
        assert_eq!(prod(vec![d("0.1"), d("3")]), vec![d("0.3")]);
        // the partition is exact, and only its one combined number is rounded
        assert_eq!(
            prod(vec![Rat::new(1, 3).unwrap(), d("1e300"), d("1e-300")]),
            vec![Rat::new(1, 3).unwrap()]
        );
        // a fully numeric product stays exact: the part is valued whole
        let a = d("0.30000000000000004");
        assert_eq!(
            literal_partition_product(vec![a.clone(), a], true, &cx),
            vec![d("0.0900000000000000240000000000000016")]
        );
        // a conversion keeps the exact product
        let bare = Cx::bare(view);
        let a = d("0.30000000000000004");
        assert_eq!(
            literal_partition_product(vec![a.clone(), a], false, &bare),
            vec![d("0.0900000000000000240000000000000016")]
        );
        let mut p = Cx::bare(view);
        p.mode = RuleMode::Permissive;
        let a = d("0.30000000000000004");
        assert_eq!(
            literal_partition_product(vec![a.clone(), a], false, &p),
            vec![d("0.0900000000000000240000000000000016")],
            "permissive rounds where the number lands (`land`)"
        );
    });
    // no float64 value replaces an exact one past the range of the normal doubles
    let third = Rat::new(1, 3).unwrap();
    assert_eq!(strict_number(third.clone(), f64::INFINITY), third);
    assert_eq!(strict_number(third.clone(), 0.0), third);
    assert_eq!(strict_number(third.clone(), 1e-310), third);
    // an integer stays exact
    let big = Rat::int(12157665459056928801);
    assert_eq!(strict_number(big.clone(), 1.2157665459056929e19), big);
    // a written literal of more digits than its double needs, in the simplify walk
    with_view(|view| {
        let mut cx = Cx::bare(view);
        cx.fold_f64 = true;
        assert_eq!(
            strict_literal(d("0.1234567890123456789"), &cx),
            d("0.12345678901234568")
        );
        assert_eq!(
            strict_literal(d("0.1234567890123456789"), &Cx::bare(view)),
            d("0.1234567890123456789"),
            "a conversion keeps the literal as written"
        );
        assert_eq!(strict_literal(d("0.1"), &cx), d("0.1"));
        let mut p = Cx::bare(view);
        p.mode = RuleMode::Permissive;
        assert_eq!(
            strict_literal(d("0.1234567890123456789"), &p),
            d("0.1234567890123456789"),
            "permissive rounds where a literal lands (`land`)"
        );
    });
}

/// The shortest decimal reads back as the float.
#[test]
fn the_shortest_decimal_reads_back_as_its_double() {
    for y in [
        0.1,
        1.0 / 3.0,
        std::f64::consts::PI,
        1e-300,
        5e-324,
        f64::MAX,
        -0.75,
        1e22,
        0.45445993041202365,
    ] {
        let d = Rat::shortest_reading_as(y).unwrap();
        assert_eq!(d.to_f64_nearest(), y, "{y:?}");
    }
    assert_eq!(Rat::shortest_reading_as(f64::INFINITY), None);
}
