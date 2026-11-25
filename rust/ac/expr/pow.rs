//! THE POWER CONSTRUCTOR [`pow`]: its folds, the distribution of an integer power over a product
//! and its licence, and the rejoin of reciprocals at the output boundary.

use super::*;

/// Is `e` an odd root (`rootn` of an odd index), or a power of one? Its collection can leave its
/// class (`cbrt(u)^3` is `u`), which the certificate-free parse already decides; a divisor
/// holding one is distributed by its members' form there too (`power_bag`), so its members meet
/// that decision as the same members spelled one by one do.
fn holds_odd_root(e: &Ex, cx: &Cx) -> bool {
    match e {
        Ex::Pow(b, _) => holds_odd_root(b, cx),
        Ex::Fun(op, a) => {
            a.len() == 2
                && cx.view.tok_is(*op, "rootn")
                && matches!(&a[1], Ex::Num(k) if k.is_odd_integer())
        }
        _ => false,
    }
}

/// Is `e` an infinity, or a sum holding one (an infinite or NaN value, never zero)?
fn holds_infinity(e: &Ex) -> bool {
    match e {
        Ex::PosInf | Ex::NegInf => true,
        Ex::Add(ts) => ts
            .iter()
            .any(|t| matches!(t, Ex::PosInf | Ex::NegInf) || is_inf_carrying(t)),
        _ => false,
    }
}

// The residual equal-mu same-coefficient-sign tie in the
// sign-placement owner breaks on `cmp_ex` -- the ONE canonical total order, which
// compares Leaf/Fun tokens by their STRINGS. The dedicated `ex_struct_cmp` this
// replaced compared them by raw token ID, i.e. by INTERNING ORDER, so on a fully
// mu-tied orbit (the two mirror pairings price alike) the winner was a function of which
// literal the input stream interned first: `-1 * (a-b)*(c-d)` and its factor-swapped
// spelling picked OPPOSITE mirror pairings (extreme-lane fuzz rows 829655/866090, the
// two regressions at 1M rows).

/// LOSSY simplification priced reciprocal products WORSE
/// than sound's on the sound-REFUSED set. The blanket zero-set licence makes the lossy
/// `pow` distribute `(a*b)^-1 -> a^-1 * b^-1` unconditionally -- the match-enabling
/// normal form the `$`-cancellation needs (`x * inv(x*y) -> inv y` only cancels through
/// the flat bag) -- but each distributed factor carries its own `^-1` wrapper, so where
/// NO partner cancels, the lossy endpoint priced strictly above sound's licence-refusal
/// form `(a*b)^-1` (registered row 1414: mu 80 vs 72). Neither mode consulted `mu` when
/// choosing the spelling; sound's smaller form was a fail-closed accident.
///
/// This is an OUTPUT-BOUNDARY projection (`perm.reciprocal_projection`), deliberately NOT a
/// constructor arm. The first
/// build of this fix put the join into `mul`'s final assembly, making the joined form
/// the WORKING representation -- disqualified by the debug stability assert: `pow`'s
/// distribution arm settles its two parts in a nested vacuum `mul` that joined them
/// before the enclosing bag could splice, so construction order leaked into the state
/// (and joined mid-chain states also shift rule reachability and the descent gate's mu
/// ordering). The distributed form is the right working representation (match-enabling,
/// cancellation-visible); the joined form is only ever a better FINAL SPELLING. So the
/// rewrite chain runs entirely in the distributed canon, byte-identical to pre-fix, and
/// [`rejoin_projection`] applies this decision once, bottom-up, to the returned
/// endpoint (measured on the 200k battery: mode-ordering rows 822 -> 780, 42 fixed,
/// 0 added).
///
/// The decision, per `Mul` node: offer the negative-integer-power members back as one
/// joined `(prod f^|n|)^-1`; the strictly `mu`-smaller of the two spellings wins (tie:
/// distributed, the incumbent). The `mu` arithmetic makes the win set exact: for k
/// joined factors the join saves (k-1) `Pow` wrappers and pays one inner `Mul` symbol,
/// so k = 2 INSIDE a wider bag is a tie (distributed stays) while the whole-bag pair
/// (the outer `Mul` unwraps: row 1414's shape) and every k >= 3 join strictly. Two
/// gates keep the decision honest:
///
/// * SOUND ALIGNMENT: the join is offered only where sound-mode `pow` would REFUSE to
///   re-distribute it (`nz_ae_certified` / `certainly_nonneg` per member -- the same
///   conjuncts as the distribution licence, minus the lossy blanket). Where sound
///   distributes, lossy stays distributed too: the two output forms agree there.
///   Idempotence is a funnel property: re-simplifying a joined output re-parses it
///   through `pow`'s blanket distribution back to a working endpoint whose
///   projection must reproduce the output. That endpoint is NOT in general the
///   state the join was priced against (the two construction routes -- positive
///   inner bag vs hidden `^-1` members -- orient trade-site mirrors differently),
///   so the decision is additionally gated on the candidate's RE-SETTLE IMAGE
///   (see the comparison at the return).
/// * ELIGIBILITY excludes BARE-`Const` bases (joining `c^-1 * c'^-1` would let the
///   inner bag's `has_const` flag conflate INDEPENDENT constants -- the flag collapses
///   multiplicity) and literal-`Num` bases (only `0^neg` survives `pow`, and a literal
///   zero never joins a denominator -- the inner product would collapse; cf. the
///   renderer's den rule). DEEP Consts (inside function arguments, `atanh(C*x2)`) are
///   eligible: the inner bag keeps Const-bearing factors verbatim in the opaque bucket,
///   never merged -- the first build excluded them wholesale (`contains_const`) and
///   silently forfeited the join on exactly the sound-refused shapes the projection
///   exists for (measured: whole-bag pairs under `sin(..)` stayed distributed at mu +8
///   over sound's refusal form).
///
/// Termination: the joined node is built RAW (`Ex::Pow`), never through `pow` (whose
/// lossy blanket would immediately re-distribute it); the inner `mul` sees only
/// positive-power members, so no recursion re-enters this decision. The decision
/// itself is guarded by the re-derivability comparison at the return (see there):
/// a join ships only when it strictly beats its own re-settle image, which is what
/// makes the shipped spelling a fixpoint under re-parse.
fn rejoin_reciprocals(settled: Ex, signed: bool, cx: &Cx) -> Ex {
    let Ex::Mul(members) = &settled else {
        return settled;
    };
    let eligible: Vec<usize> = members
        .iter()
        .enumerate()
        .filter_map(|(i, f)| {
            let Ex::Pow(b, e) = f else { return None };
            // A literal-infinity base never joins: `inv(inf)` is a DETERMINED zero
            // (`pow` folds it at construction). Joining it hides that determined
            // fold behind an opaque base -- measured: every net-regressed battery row
            // had `inf` as a direct member of the joined product. DEEP infinities
            // (inside function arguments, row 1414's `acos(inf * x4)`) stay eligible:
            // no fold sees through those either way.
            if matches!(**b, Ex::Num(_) | Ex::PosInf | Ex::NegInf | Ex::Const(_)) {
                return None;
            }
            let Ex::Num(r) = &**e else { return None };
            (r.is_integer() && r.is_negative()).then_some(i)
        })
        .collect();
    if eligible.len() < 2 {
        return settled;
    }
    let inner_members: Vec<Ex> = eligible
        .iter()
        .map(|&i| {
            let Ex::Pow(b, e) = &members[i] else {
                unreachable!()
            };
            let Ex::Num(r) = &**e else { unreachable!() };
            if *r == Rat::NEG_ONE {
                (**b).clone()
            } else {
                pow((**b).clone(), Ex::Num(r.checked_neg().unwrap()), cx)
            }
        })
        .collect();
    let inner = mul(inner_members.clone(), cx);
    // The joined base's own decision can trade a member's sign into its coefficient (a sum
    // whose other orientation is cheaper once it stands in a product); that sign leaves the base
    // (`joined_base_sign`).
    let (inner, base_sign) = joined_base_sign(inner, cx);
    {
        let Ex::Mul(v) = &inner else {
            // The positive-power bag collapsed below two factors: no joined SHAPE exists.
            return settled;
        };
        let sound_distributes =
            v.iter().all(|f| cx.nz_ae_certified(f)) || v.iter().all(|f| cx.certainly_nonneg(f));
        if sound_distributes {
            return settled;
        }
    }
    let mut rest: Vec<Ex> = members
        .iter()
        .enumerate()
        .filter(|(i, _)| !eligible.contains(i))
        .map(|(_, f)| f.clone())
        .collect();
    if base_sign {
        // The rest of the bag takes the sign, placed by its own decision (`mul()`): the parse of the
        // joined spelling decides the same members with the base's inverted members beside them,
        // which carry no sign of their own.
        rest.push(Ex::Num(Rat::NEG_ONE));
        rest = match mul(rest, cx) {
            Ex::Mul(v) => v,
            Ex::Num(r) if r.is_one() => Vec::new(),
            f => vec![f],
        };
        // A bag that may not take a new sign (`signed`) keeps its working spelling.
        if !signed && gains_a_sign(&rest, &settled) {
            return settled;
        }
    }
    let assemble = |mut cand: Vec<Ex>| match cand.len() {
        1 => cand.pop().unwrap(),
        _ => {
            cand.sort_by(|a, b| mul_factor_cmp(a, b, cx.view));
            Ex::Mul(cand)
        }
    };
    let joined = Ex::Pow(Box::new(inner.clone()), Box::new(Ex::Num(Rat::NEG_ONE)));
    let mut cand = rest.clone();
    cand.push(joined);
    let mut best = assemble(cand);
    // COEFFICIENT COMPLETION: the bag's rational
    // coefficient can ride INSIDE the joined base, reciprocated -- `0.5 * (x4*S)^-1`
    // IS `(2*x4*S)^-1` -- and when the bag then collapses to the lone joined node, the
    // outer `Mul` unwraps (mu -8, and mu(1/c) vs mu(c) can differ too). Sound's
    // licence-refusal forms keep their coefficient inside the base exactly like this
    // (`inv(-1 * x0 * S)` from `inv(neg(..))` spellings), so without the completion
    // the projection stopped one symbol short of sound's spelling on every such row
    // (23/200k, all gap 8). The funnel stays closed: re-parse distributes the joined
    // base, `pow` folds the literal's reciprocal back into the coefficient, and this
    // decision re-derives.
    //
    // RE-DERIVABILITY GATE: the completion is
    // funnel-stable only if the reciprocated coefficient SURVIVES VERBATIM as a `Num`
    // member of the joined base. `mul()` inside the completion runs the full assembly,
    // and the Const sign fold may CONSUME a negative `cinv` into a Const-bearing Add
    // member -- a strict mu win whose source is the sign fold, not the join, and one
    // the funnel cannot re-derive: the re-parsed (absorbed) state has no Num left in
    // the bag, the completion cannot re-fire, the plain join ties, and
    // tie -> distributed -- so the joined spelling shipped exactly once and the next
    // lossy pass moved (lossy idempotence, 2/200k, fuzz rows 92892 + 115616).
    // When the coefficient survives (positive c always; negative c beside sums the
    // fold does not touch), re-parse re-extracts it through pow's distribution and
    // the completion re-derives -- including sound-aligned `inv(-1 * x0 * S)` joins
    // (blanket refusal of negative c donated a uniform mu gap of 8 on 8/200k rows).
    if let Some(ci) = rest.iter().position(|f| matches!(f, Ex::Num(_))) {
        let Ex::Num(c) = &rest[ci] else {
            unreachable!()
        };
        if let Some(cinv) = c.checked_inv() {
            let inner2 = mul(vec![Ex::Num(cinv.clone()), inner.clone()], cx);
            let survives = matches!(&inner2, Ex::Mul(v)
                if v.iter().any(|f| matches!(f, Ex::Num(n) if *n == cinv)));
            // A NEGATIVE reciprocated coefficient
            // beside a sign-trade site is no longer funnel-stable -- pow()'s
            // odd-negative pre-fold now folds that sign into the site on re-parse
            // (the row-310 pole-orientation cure), so the joined spelling cannot
            // re-derive verbatim (measured: corpus row 33 lossy idempotence). The
            // completion skips exactly the pre-fold's firing condition.
            let funnel_stable = !cinv.is_negative()
                || !matches!(&inner2, Ex::Mul(v)
                    if v.iter().any(|f| is_sign_trade_site(f, cx)));
            if survives && funnel_stable {
                let joined2 = Ex::Pow(Box::new(inner2), Box::new(Ex::Num(Rat::NEG_ONE)));
                let mut cand2: Vec<Ex> = rest
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != ci)
                    .map(|(_, f)| f.clone())
                    .collect();
                cand2.push(joined2);
                let candidate2 = assemble(cand2);
                if complexity(&candidate2, cx.view) < complexity(&best, cx.view) {
                    best = candidate2;
                }
            }
        }
    }
    // RE-DERIVABILITY COMPARISON (fuzz rows 303116/472156/603382 and, after the
    // first-cut fix, 59022/194208/308376/593418/713415/790992 -- all introduced or
    // exposed by the sign-placement owner): the join used to be judged against the WORKING state, but the
    // working state is not what a shipped join re-parses to. Parse builds the joined
    // base's inner bag POSITIVELY first, where bare trade sites get re-oriented (the
    // Coeff orbit's argmin, or the Const-side Free flip beside a Const-bearing member)
    // before the blanket re-distributes -- while the division route builds the same
    // value member-by-member with every site HIDDEN inside a `^-1` wrapper (not a
    // carrier: the one-zero pole trilemma), preserving arrival mirrors. Two routes,
    // two working states, and a join priced against one route re-derived into the
    // other: the strict win evaporated into a tie on the second pass and the lossy
    // endpoint moved (lossy idempotence). The gate is the re-derivability principle applied to the
    // WHOLE decision: a join ships only if it strictly beats BOTH the working state
    // (the improvement test, as before) AND its own RE-SETTLE IMAGE -- the state its
    // re-parse provably lands on (inner is `mul`-built == what parse rebuilds, `mul`
    // is idempotent on it, so the image is exact). A shipped join then re-picks
    // itself on every later pass (fixpoint by construction), and a refused join
    // ships the working state, whose rendering re-parses member-by-member through
    // the hidden-site route back to itself.
    let resettle = |cand: &Ex| -> Ex {
        let members: Vec<Ex> = match cand {
            Ex::Mul(v) => v.clone(),
            other => vec![other.clone()],
        };
        let mut parts: Vec<Ex> = Vec::with_capacity(members.len() + 2);
        for m in members {
            match m {
                Ex::Pow(b, e)
                    if matches!(&*e, Ex::Num(r) if *r == Rat::NEG_ONE)
                        && matches!(&*b, Ex::Mul(_)) =>
                {
                    let Ex::Mul(bm) = *b else { unreachable!() };
                    for f in bm {
                        parts.push(pow(f, Ex::Num(Rat::NEG_ONE), cx));
                    }
                }
                other => parts.push(other),
            }
        }
        mul(parts, cx)
    };
    let mu_best = complexity(&best, cx.view);
    let mu_settled = complexity(&settled, cx.view);
    let image = resettle(&best);
    let mu_image = complexity(&image, cx.view);
    if mu_best < mu_settled && mu_best < mu_image {
        best
    } else if base_sign
        && mu_image < mu_settled
        && (signed || !gains_a_sign(std::slice::from_ref(&image), &settled))
    {
        // The base's decision found a cheaper orientation for a member the working state holds
        // under its inverse: the re-settle image keeps it, distributed, and re-derives itself.
        image
    } else {
        settled
    }
}

/// Does `members` (a bag's members, or the bag itself) carry a negative coefficient where the bag
/// `settled` carries none?
fn gains_a_sign(members: &[Ex], settled: &Ex) -> bool {
    let negative = |f: &Ex| match f {
        Ex::Num(r) => r.is_negative(),
        Ex::Mul(v) => v.iter().any(|g| matches!(g, Ex::Num(r) if r.is_negative())),
        _ => false,
    };
    members.iter().any(negative) && !negative(settled)
}

/// The joined base without a negative coefficient of its own, and whether one left it: a sign the
/// joined base's own product placed in its coefficient leaves the base for the rest of the bag
/// (`rejoin_reciprocals`). The base is decided as a product of its own, so a sum the distributed
/// bag held under its inverse can take its cheaper orientation there and hand the sign to the
/// base's coefficient, while the bag keeps the sign it had. Kept in the base, the two signs would
/// print as `(-a)/(-(b (x - u)))`, which the parse reads as `a/(b (x - u))`, so a second call would
/// print another spelling. In the lossy context the projection runs in, the inverse of `-P` is
/// minus the inverse of `P` (they differ only at `P`'s zero), so the sign joins the rest of the bag,
/// whose own decision places it. A bag the parse reads with its sign kept (the answer, or a term of
/// a sum) takes it as it comes; inside a function's argument or a power's base, where the parse
/// moves or drops a product's sign, a bag that would gain a sign keeps its working spelling.
fn joined_base_sign(inner: Ex, cx: &Cx) -> (Ex, bool) {
    if !cx.permits(Permissive::Agreement) {
        return (inner, false);
    }
    let Ex::Mul(mut v) = inner else {
        return (inner, false);
    };
    if !matches!(v.first(), Some(Ex::Num(r)) if r.is_negative()) {
        return (Ex::Mul(v), false);
    }
    let Ex::Num(r) = &v[0] else { unreachable!() };
    match r.checked_neg() {
        Some(p) if p.is_one() => {
            v.remove(0);
        }
        Some(p) => v[0] = Ex::Num(p),
        None => return (Ex::Mul(v), false),
    }
    let base = if v.len() == 1 {
        v.pop().expect("one member")
    } else {
        Ex::Mul(v)
    };
    (base, true)
}

/// The output-boundary rejoin walk (see [`rejoin_reciprocals`]): apply the reciprocal
/// join decision bottom-up to every `Mul` node of the final lossy state. STRUCTURAL
/// rebuild on purpose -- routing through the constructors would re-distribute the joins
/// (`pow`'s lossy blanket). Bags whose members changed are re-sorted under their own
/// comparator so the projected tree spells deterministically.
pub fn rejoin_projection(e: Ex, cx: &Cx) -> Ex {
    rejoin_walk(e, true, cx)
}

/// [`rejoin_projection`] at a node; `signed`: does the parse keep a sign the node's bag takes (the
/// answer itself, a term of a sum, a member of a product), or move or drop it (a function's
/// argument, a power's base or exponent)?
fn rejoin_walk(e: Ex, signed: bool, cx: &Cx) -> Ex {
    match e {
        Ex::Add(v) => {
            let mut terms: Vec<Ex> = v.into_iter().map(|x| rejoin_walk(x, true, cx)).collect();
            terms.sort_by(|a, b| add_term_cmp(a, b, cx.view));
            Ex::Add(terms)
        }
        Ex::Mul(v) => {
            let mut m: Vec<Ex> = v.into_iter().map(|x| rejoin_walk(x, true, cx)).collect();
            m.sort_by(|a, b| mul_factor_cmp(a, b, cx.view));
            rejoin_reciprocals(Ex::Mul(m), signed, cx)
        }
        Ex::Pow(b, ex) => Ex::Pow(
            Box::new(rejoin_walk(*b, false, cx)),
            Box::new(rejoin_walk(*ex, false, cx)),
        ),
        // `atan2` reads its arguments' zero signs, and the parse keeps their signs (`atan2_of`).
        Ex::Fun(f, v) if v.len() == 2 && cx.ieee_zeros() && cx.view.tok_is(f, "atan2") => {
            Ex::Fun(f, v.into_iter().map(|x| rejoin_walk(x, true, cx)).collect())
        }
        // A lattice bag keeps its members in the canonical order (`lattice`).
        Ex::Fun(f, v) if lattice_head(f, cx.view) => {
            let mut m: Vec<Ex> = v.into_iter().map(|x| rejoin_walk(x, false, cx)).collect();
            m.sort_by(|a, b| cmp_ex(a, b, cx.view));
            Ex::Fun(f, m)
        }
        Ex::Fun(f, v) => Ex::Fun(
            f,
            v.into_iter().map(|x| rejoin_walk(x, false, cx)).collect(),
        ),
        leaf => leaf,
    }
}

/// Rebuild `base^exp` (rational or symbolic-with-count) through the power constructor so
/// integer-power distribution and exact folds apply; as a member of the product being built
/// ([`pow_member`]), whose sign placement orients it.
pub(super) fn rebuild_factor(base: Ex, sym: Option<Ex>, r: Rat, cx: &Cx) -> Ex {
    match sym {
        None => pow_member(base, Ex::Num(r), cx),
        Some(e) => {
            if r.is_one() {
                pow_member(base, e, cx)
            } else {
                let scaled = mul(vec![Ex::Num(r), e], cx);
                pow_member(base, scaled, cx)
            }
        }
    }
}

/// `pow`'s licence to distribute the integer power `e` over a product of `v`: the bag it distributes
/// over (a negative coefficient folded into a sign-trade site first), or `None` where it
/// refuses. The divisor parse and `canon` ask it too (`distributes_power`, `inverted_members`),
/// so a quotient is distributed exactly where `pow` would distribute it.
/// Does `pow` distribute the integer power `e` over the product of `v` (`power_bag`'s licence)?
pub(crate) fn distributes_power(v: &[Ex], e: &Rat, cx: &Cx) -> bool {
    power_bag(v, e, cx).is_some()
}

/// THE PREMISE OF `pow.distribute`: may the integer power `e` distribute over the product of
/// `v`? A non-negative or even power always; an odd negative one where every member is nonzero
/// almost everywhere, a member is infinite, all are certainly `>= 0`, or a member's zero is one
/// whose sign an inverse reads (`inverts_a_signed_zero`).
pub(crate) fn distribution_licensed(v: &[Ex], e: &Rat, cx: &Cx) -> bool {
    !e.is_negative()
        || e.is_even_integer()
        || inverts_a_signed_zero(v, cx)
        || v.iter().any(holds_infinity)
        || v.iter().all(|f| zero_set_null(f, cx))
        || (v.iter().any(|f| holds_odd_root(f, cx))
            && v.iter()
                .all(|f| cx.nz_ae_licensed(f) || zero_set_null_by_form(f, cx)))
        || v.iter().all(|f| cx.certainly_nonneg(f))
}

pub(super) fn power_bag(v: &[Ex], e: &Rat, cx: &Cx) -> Option<Vec<Ex>> {
    if !Law::PowDistribute.holds_powers(Powers::Product { members: v, n: e }, cx) {
        return None;
    }
    Some(distributed_bag(v, e, cx))
}

/// The licences a distributed reading's bindings must meet: for each product read distributed,
/// its members and power ([`pattern_distributed`]).
pub(crate) type Licences = Vec<(Vec<Ex>, Rat)>;

/// THE DISTRIBUTED READING OF A PATTERN (`pow.distribute` at bind time). A slot may bind a member
/// whose zero set holds a region, so a pattern's product under an odd negative power holds where
/// its members cannot license the distribution (`distribution_licensed`), while its instances
/// distribute wherever their members do: `1/(x1*?0)` stays one member, `1/(x1*x3)` is
/// `x1^-1*x3^-1`. This is `e` with every such product distributed as `pow` distributes a licensed
/// one, and, for each, its members and power: the licence a binding must meet for the reading to
/// match (`AcRule::licences`). `None` where `e` holds no such product.
pub(crate) fn pattern_distributed(e: &Ex, cx: &Cx) -> Option<(Ex, Licences)> {
    fn walk(e: &Ex, cx: &Cx, out: &mut Licences) -> Option<Ex> {
        let kids = |v: &[Ex], out: &mut Licences| -> Option<Vec<Ex>> {
            let mut kids: Option<Vec<Ex>> = None;
            for (i, c) in v.iter().enumerate() {
                if let Some(r) = walk(c, cx, out) {
                    kids.get_or_insert_with(|| v.to_vec())[i] = r;
                }
            }
            kids
        };
        match e {
            Ex::Pow(b, x) => {
                if let (Ex::Mul(v), Ex::Num(n)) = (&**b, &**x) {
                    if n.is_integer()
                        && n.is_negative()
                        && v.iter().any(|f| f.contains_wildcard(cx.view))
                        && !distribution_licensed(v, n, cx)
                    {
                        let members = kids(v, out).unwrap_or_else(|| v.clone());
                        out.push((members.clone(), n.clone()));
                        let raised = distributed_bag(&members, n, cx)
                            .into_iter()
                            .map(|f| pow_member(f, Ex::Num(n.clone()), cx))
                            .collect();
                        return Some(mul(raised, cx));
                    }
                }
                let nb = walk(b, cx, out);
                let nx = walk(x, cx, out);
                (nb.is_some() || nx.is_some()).then(|| {
                    pow(
                        nb.unwrap_or_else(|| (**b).clone()),
                        nx.unwrap_or_else(|| (**x).clone()),
                        cx,
                    )
                })
            }
            Ex::Add(v) => kids(v, out).map(|k| add(k, cx)),
            Ex::Mul(v) => kids(v, out).map(|k| mul(k, cx)),
            Ex::Fun(f, v) => kids(v, out).map(|k| fun(*f, k, cx)),
            _ => None,
        }
    }
    let mut licences = Vec::new();
    walk(e, cx, &mut licences).map(|d| (d, licences))
}

/// The members `pow` raises where it distributes the integer power `e` over the product of `v`
/// ([`power_bag`] without its licence).
fn distributed_bag(v: &[Ex], e: &Rat, cx: &Cx) -> Vec<Ex> {
    // For a NEGATIVE ODD exponent, fold a negative bag coefficient
    // into the first sign-trade site BEFORE distributing (exact at the
    // finite bag level: -c * S == c * (-S)). After distribution the sum
    // sits under a REFUSED odd-negative carrier -- (S)^-n and -((-S))^-n
    // are a.e.-equal but POLE-DIFFERENT, so no owner may reconcile them
    // later; whether the sign had already met the bag decided which
    // rendering the a.e. licence produced (corpus row 310: chain held
    // (1-x4)^-1 while parse of its own serialization built
    // -(x4-1)^-1). The pre-fold is UNCONDITIONAL (not mu-judged):
    // post-distribute there is no legal trade, so entry-independence
    // requires one fixed pre-state.
    let vv: Vec<Ex>;
    let bag: &[Ex] = if e.is_negative() && e.is_odd_integer() && !inverts_a_signed_zero(v, cx) {
        let ci = v
            .iter()
            .position(|f| matches!(f, Ex::Num(r) if r.is_negative()));
        let si = v.iter().position(|f| is_sign_trade_site(f, cx));
        match (ci, si) {
            (Some(ci), Some(si)) => {
                let nc = match &v[ci] {
                    Ex::Num(r) => r.checked_neg().map(Ex::Num),
                    _ => None,
                };
                match (sign_trade_flip(&v[si], cx), nc) {
                    (Some(nf), Some(nc)) => {
                        let mut w = v.to_vec();
                        w[si] = nf;
                        w[ci] = nc;
                        if matches!(&w[ci], Ex::Num(r) if r.is_one()) {
                            w.remove(ci);
                        }
                        vv = w;
                        &vv
                    }
                    _ => v,
                }
            }
            _ => v,
        }
    } else {
        v
    };
    bag.to_vec()
}

/// The canonical power constructor. Folds and gates:
///
/// * `e == 1` -> base. `e == 0` -> 1 (TOTAL under the engine's C99-aligned pow: `pow(x, 0) = 1`
///   for EVERY x including NaN and the infinities).
/// * `base == 1` -> 1 (TOTAL: `pow(1, y) = 1` for every y including NaN).
/// * rational^integer folds exactly (checked; overflow keeps the symbolic form).
///   rational^(p/q) folds exactly when the q-th root is exact (`4^(1/2) -> 2`,
///   `(-8)^(1/3)` is NOT folded: the engine's rational pow is NaN on negative bases with
///   non-integer exponents -- the REAL odd root is the separate `pow1_3` operator).
///   `0^negative` and inexact roots stay symbolic (the rules-pass numeric fold owns IEEE edge
///   parity with the shipped engine).
/// * `(a^r)^s -> a^(r*s)` under the composition licence: sound iff (`r` integer AND (`s` integer
///   OR `r` odd)) OR (`r` non-integer AND `r*s` non-integer) OR the base is certainly
///   non-negative. Counterexamples pinned in tests: `(a^2)^(1/2) = |a| != a`;
///   `(a^(1/2))^2 = a` only on `a >= 0`.
/// * `(a*b)^n` for INTEGER n distributes over the factors (TOTAL as extended-real evaluations;
///   for non-integer exponents `(ab)^(1/2) != a^(1/2) b^(1/2)` on `a, b < 0`).
/// * a power that is a sign-trade site takes the orientation it takes as a product's one factor
///   (`lone_site`).
pub fn pow(base: Ex, exp: Ex, cx: &Cx) -> Ex {
    match pow_member(base, exp, cx) {
        node @ Ex::Pow(..) => lone_site(node, cx),
        other => other,
    }
}

/// The SPELLING `pow(b, e)`, as the parse reads it: [`pow`], except numpy's square root in `f64`.
/// Float64's `b ** 0.5` is `sqrt(b)` (`numeric::aligned_pow`), which keeps a zero's sign
/// (`sqrt(-0) = -0`) where `rootn(b, 2)`, the engine's reading of `b^(1/2)`, gives +0. Where `b`
/// may be a zero with a sign of its own (`sqrt_parts_from_rootn`) the two are different functions
/// there, as an inverse or `atan2` reads them: the written negative zero's root is the written
/// negative zero, and any other such `b` keeps the call as written, a node no exponent arithmetic
/// reads (`1/pow(ceil(x), 1/2)` is -inf on (-1, 0), `1/rootn(ceil(x), 2)` +inf).
pub(crate) fn pow_spelled(base: Ex, exp: Ex, cx: &Cx) -> Ex {
    let pow_tok = cx.view.intern("pow");
    if matches!(&exp, Ex::Num(h) if *h == Rat::new(1, 2).expect("1/2"))
        && cx.view.arity(pow_tok) == Some(2)
        && !Law::PowHalfRoot.holds(&base, cx)
    {
        // `sqrt` keeps a zero, its sign and NaN: the root of the written negative zero, or of a
        // zero product (`mul.zero`), is that zero.
        if is_neg_zero(&base, cx.view) || zero_product(&base, cx) {
            return base;
        }
        return Ex::Fun(pow_tok, vec![base, exp]);
    }
    pow(base, exp, cx)
}

/// [`pow`] for a member of a product, whose sign placement decides the member's orientation with
/// the product's other members (`sign_place`): the same power, a sign-trade site left in the
/// orientation it was built in.
///
/// The laws applied here (`ac::laws::Law`): `pow.refused_power`, `num.zero_pole`, `num.inf_power`,
/// `num.literal_base_inf_exponent`, `num.table_fold`.
pub(crate) fn pow_member(base: Ex, exp: Ex, cx: &Cx) -> Ex {
    // A ZERO PRODUCT UNDER A POSITIVE POWER (`mul.zero`): the power 1/2 is float64's `sqrt`, which
    // keeps a zero, its sign and NaN, so the power is the product itself; any other that is no
    // odd integer is `+0` (or NaN) whatever the zero's sign, so the product drops the factors that
    // only gave it one (`unsigned_zero`).
    let base = match &exp {
        Ex::Num(q) if *q == Rat::new(1, 2).expect("1/2") && zero_product(&base, cx) => {
            return base;
        }
        Ex::Num(q) if q.signum() > 0 && !q.is_odd_integer() => {
            unsigned_zero(&base, cx).unwrap_or(base)
        }
        _ => base,
    };
    // A literal base or exponent beside a non-literal one LANDS here (permissive's ρ,
    // [`land`]), where that one is not ground; a power of two literals folds exactly or stays
    // as written.
    let (base, exp) = if matches!(base, Ex::Num(_)) != matches!(exp, Ex::Num(_))
        && !in_ground([&base, &exp], cx)
    {
        (land_operand(base, cx), land_operand(exp, cx))
    } else {
        (base, exp)
    };
    // In `permissive` a power of two literals is the table's exact value
    // (`table_literal_fold`):
    // `pow(0, -1)` and `pow(0, -2)` are NaN (a division by zero), `pow(1, inf)`, `pow(inf, 0)`
    // and `pow(nan, 0)` NaN, `pow(-inf, 0)` 1 -- before the one-zero arms and the `x^0` and `1^x`
    // identities below, which hold for every finite real.
    if cx.table_values() && table_literal(&base) && table_literal(&exp) {
        if let Some(folded) = table_literal_fold("pow", &[base.clone(), exp.clone()]) {
            return folded;
        }
    }
    if let Ex::Num(e) = &exp {
        if e.is_zero() && Law::PowZero.holds(&base, cx) {
            return Ex::Num(Rat::ONE);
        }
        if e.is_one() && Law::PowOne.holds(&base, cx) {
            return base;
        }
    }
    if matches!(&base, Ex::Num(b) if b.is_one()) && Law::PowBaseOne.holds(&exp, cx) {
        return Ex::Num(Rat::ONE);
    }
    // THE WRITTEN NEGATIVE ZERO under a negative integer power is float64's pole of its sign:
    // `(-0)^-1` is -inf, `(-0)^-2` is +inf (`NEG_ZERO`; `f64` only).
    if cx.f64_semantics() && is_neg_zero(&base, cx.view) {
        if let Ex::Num(k) = &exp {
            if k.is_integer() && k.is_negative() {
                return if k.is_odd_integer() {
                    Ex::NegInf
                } else {
                    Ex::PosInf
                };
            }
        }
    }
    // THE NEGATIVE HALF POWER OF A SIGNED ZERO (`f64` only). Float64 reads `b^(1/2)` the way
    // `rootn(b, 2)` is, +0 at -0 (C's `pow` at every non-integer exponent but numpy's 0.5), and
    // prints `b^(-1/2)` as `inv pow b 1/2`, whose `pow b 0.5` numpy computes as `sqrt(b)`, -0
    // at -0 (`numeric::aligned_pow`): where `b` may be a zero with a sign of its own
    // (`sqrt_parts_from_rootn`), the power is `1/rootn(b, 2)`, which prints as the value it
    // is (+inf at -0). (The SPELLING `pow b 0.5` is numpy's `sqrt`: `pow_spelled`.)
    if let Ex::Num(e) = &exp {
        if *e == Rat::new(-1, 2).expect("-1/2") && !Law::PowHalfRoot.holds(&base, cx) {
            if is_neg_zero(&base, cx.view) {
                return Ex::PosInf;
            }
            let root = fun(cx.view.intern("rootn"), vec![base, Ex::int(2)], cx);
            return pow_member(root, Ex::int(-1), cx);
        }
    }
    // rational^rational exact folds.
    if let (Ex::Num(b), Ex::Num(e)) = (&base, &exp) {
        // DETERMINED POLE: a literal-zero base with a
        // negative literal exponent is exact non-finite arithmetic under the
        // one-zero convention (0^n = +0 for n > 0, so 0^{-n} = 1/+0 = +inf; the
        // same convention the signed-inf licence settled for `inv 0`). Without
        // this arm, exponent collection could BUILD `Pow(0, -2)` (from
        // `inv 0 * inv 0`) -- a state whose serialization `inv pow 0 2` re-reads
        // to `inv 0` because the inner `pow 0 2` folds on the way in: a genuine
        // violation of the canonical-form uniqueness contract, caught by the
        // debug `stable()` assertion (release outputs were value-correct; the
        // contract was not).
        if b.is_zero() && e.is_negative() {
            return Ex::PosInf;
        }
        if e.is_integer() {
            if let Some(r) = b.checked_pow_integer(e) {
                return Ex::Num(r);
            }
        } else if !b.is_negative() {
            // b^(p/q): exact iff the q-th root of b is rational.
            if let Some(root) = b.checked_root_integer(&e.denom()) {
                if let Some(r) = root.checked_pow_integer(&e.numer()) {
                    return Ex::Num(r);
                }
            }
        }
    }
    // DETERMINED RECIPROCAL OF INFINITY (dual of the pole arm): an
    // infinite literal base under a negative literal exponent is the same
    // reciprocal-convention arithmetic as the pole arm above -- inf^q = 1/inf^|q| = 0
    // for every rational q < 0, and (-inf)^{-n} = 1/(+-inf) = 0 for every integer
    // n > 0 (unsigned zero). It is the unique determined case where an infinite
    // ground yields a FINITE value, so the interval classification arm (which folds
    // only non-finite classes) cannot carry it; without it `inv float("inf")`
    // survives as a composite and the mine re-learns 1/inf as data. (-inf)^q with
    // q < 0 NON-integer stays unfolded here: the principal real power of a negative
    // base is NaN territory, which interval classification certifies instead.
    // Infinite EXPONENTS are untouched: b^inf is a limit notion, not arithmetic --
    // that family (exp(-inf) -> 0 etc.) arrives as certified rules by doctrine.
    // In `f64` the zero keeps float64's sign, the dual of the written negative zero's pole above:
    // `(-inf)^-1` is `1/-inf = -0` and `(-inf)^-3` is -0, `(-inf)^-2` is +0, so `1/(x1/-inf)` is
    // `-inf/x1` (a product's zero carries its sign, `mul.zero`). And `+inf` under a positive exponent is `+inf`
    // (`inf^(1/2)` is `sqrt(inf)`), decided here, so a product holding it inverts to the zero of
    // its sign: `1/-(inf^(1/2))` is -0.
    if let Ex::Num(e) = &exp {
        if matches!(base, Ex::PosInf) && e.signum() > 0 {
            return Ex::PosInf;
        }
        if e.is_negative() {
            match &base {
                Ex::PosInf => return Ex::Num(Rat::ZERO),
                Ex::NegInf if e.is_odd_integer() && f64_zero_signs(cx) => {
                    return Ex::Leaf(cx.view.intern(NEG_ZERO))
                }
                Ex::NegInf if e.is_integer() => return Ex::Num(Rat::ZERO),
                _ => {}
            }
        }
    }
    // A NEGATED SITE UNDER A POWER IS THE POWER OF ITS FLIP: `-F` and its flip (`sign_trade_flip`)
    // are one class in its two orientations, one value wherever the flip is a site's other spelling,
    // so `(-F)^s` is `flip(F)^s` for every exponent. A site standing alone may take the negated
    // orientation (`lone_site`); a power over it reads the class as it reads the site spelled
    // without the sign: `(S^3)^-1` is `S^-3` and `(S^3)^(1/2)` is `S^(3/2)` whichever orientation
    // `S^3` stands in. (A negative coefficient beside other members folds into a site where the
    // power distributes, `power_bag`.)
    // An odd negative power inverts the zero of a site that vanishes on a region with the sign
    // its orientation gives it (`region_signed_zero`): `1/-(S^3)` is `-inf` there and
    // `1/flip(S)^3` is `+inf`, so the power distributes instead (`inverts_a_signed_zero`).
    let negated_site = |f: &Ex| match &exp {
        Ex::Num(s) => Law::PowNegatedSite.holds_powers(Powers::Site { site: f, s }, cx),
        _ => true,
    };
    if let Ex::Mul(v) = &base {
        if let [Ex::Num(c), f] | [f, Ex::Num(c)] = v.as_slice() {
            if *c == Rat::NEG_ONE && negated_site(f) {
                if let Some(flipped) = sign_trade_flip(f, cx) {
                    return pow_member(flipped, exp, cx);
                }
            }
        }
    }
    // Integer powers distribute over products -- TOTAL for non-negative and for EVEN
    // integer exponents (at a zero of the product both spellings give 0 resp. +inf; at
    // infinities they agree for every integer). For ODD NEGATIVE exponents the identity
    // (a*b)^n = a^n * b^n FAILS under the one-zero contract wherever the product VANISHES
    // with a negative co-factor: (-1 * (x-|x|))^-1 is +inf on the whole half-line x >= 0
    // (1/0 = +inf) while -1 * (x-|x|)^-1 is -inf there -- a positive-measure
    // infinity-sign change (surfaced by the 1M corpus gate, 2 ppm). The failure set
    // lives entirely on the factors' ZERO SETS (every infinite/NaN configuration agrees
    // on both sides), so the licence is per-factor NONZERO-A.E. (`nz_ae_licensed`:
    // variables and Const qualify -- null hyperplane resp. null {c=0} in c-space --
    // abs-style factors with half-line zero sets are refused by the zero-set
    // certificate; a polynomial or an analytic function of one passes by its form,
    // `zero_set_null_by_form`, which the certificate-free parse reads too), a bag with an
    // infinite member (never zero), or an all-certainly-nonneg bag (at a zero every
    // co-factor is >= 0, so both spellings agree on +inf). Refusal keeps Pow(Mul[..], n):
    // on the refused set the two spellings are genuinely DIFFERENT functions, so
    // canonicalization must not identify them. The licence is `power_bag`.
    if let (Ex::Mul(v), Ex::Num(e)) = (&base, &exp) {
        if e.is_integer() {
            if let Some(bag) = power_bag(v, e, cx) {
                let parts: Vec<Ex> = bag
                    .into_iter()
                    .map(|f| pow_member(f, Ex::Num(e.clone()), cx))
                    .collect();
                return mul(parts, cx);
            }
            // REFUSED: normalize the kept form to the exponent -1 shape so refusal is
            // CONFLUENT. The same object reaches this point structured two ways -- the
            // exponent composed before meeting the bag (pow-of-pow: Pow(Mul[-1,A], -3))
            // or after (parse of the emitted spelling: Pow(Mul[-1, A^3], -1)) -- and a
            // form-dependent refusal oscillates between them across passes (observed:
            // idempotence broke on 3/400 corpus rows). (prod f)^n == (prod f^|n|)^-1 is
            // TOTAL for odd n < 0: x^n == (x^|n|)^-1 pointwise (at 0 both +inf, at
            // +-inf both 0, finite exact) and the inner |n| is a NON-NEGATIVE integer
            // distribution (total, licensed above).
            if e.is_integer() && e.cmp_int(-1) == Ordering::Less {
                let m = e.checked_neg().expect("a negative integer negates");
                let parts: Vec<Ex> = v
                    .iter()
                    .cloned()
                    .map(|f| pow_member(f, Ex::Num(m.clone()), cx))
                    .collect();
                return pow_member(mul(parts, cx), Ex::int(-1), cx);
            }
        }
    }
    // Power-of-power composition under the licence. (x^r)^s -> x^(r*s) is a THEOREM
    // only where both sides agree on the negative half-line; the three sound cases:
    //   * s integer: an integer power of a total function, exact everywhere;
    //   * r odd AND POSITIVE AND r*s NON-integer: on x < 0 the source is NaN (x^r < 0
    //     under a non-integer outer exponent) and the target is NaN too (r*s
    //     non-integer). The non-integer-product conjunct is LOAD-BEARING: with r*s
    //     integer the target is FINITE on the whole negative half-line while the source
    //     is NaN ((x^5)^0.2 -> x would fabricate values). Oddness alone licenses nothing
    //     about the outer principal root.
    //
    //     For a NEGATIVE odd r the base must additionally be unable to reach an infinity
    //     (`never_infinite`), and without that conjunct the step is unsound. The "x^r < 0 on
    //     x < 0" step is false at x = -inf when r is NEGATIVE: there x^r is ZERO (one
    //     zero, contract §9.2 -- an f64 zero's sign is measurement rendering, never a
    //     value), so the source is DEFINED (0^s) while the target x^(r*s) is NaN on a
    //     negative base with a non-integer exponent. That is defined -> undefined, which
    //     soundness forbids at ZERO measure tolerance: the null-set licence covers
    //     undefined -> defined only, "the one asymmetry". Reachable at a REAL valuation,
    //     not merely at the point -inf, because a slot binds an arbitrary SUBTREE and
    //     `neg(inv(x0))` is -inf at x0 = 0 -- the Dirac on a null set that contract v2's
    //     own repair #2 warns about. A LEAF base keeps its fold: §9.1 quantifies variables
    //     over the REALS and §10.8 makes a free constant's infinity a limit, never an
    //     attained value, so only a COMPOUND can reach one. Counterexample:
    //         x1 - rootn(inv(neg(inv(x0))), 4)  ->  x1 - inv(pow(-1/x0, 1/4))
    //         at x0 = 0:  source 1.0,  target nan.
    //     `cert_fin` cannot rescue the negative case: it certifies finiteness ALMOST
    //     EVERYWHERE, and the whole defect lives on the null set.
    //   * a certainly-non-negative base, where every spelling agrees.
    // (r even with r*s non-integer must stay refused: (x^2)^(1/4) = |x|^(1/2) is
    // DEFINED on x < 0 while x^(1/2) is NaN -- the dual failure direction.)
    if let (Some((inner_base, r)), Ex::Num(s)) = (as_rational_power(&base, cx), &exp) {
        {
            let r = &r;
            // A composition does not hide a reader of a held pair's zero
            // (`compose_hides_a_flat_read`): `1/(1/log(exp(x0)))` stays.
            let flat = compose_hides_a_flat_read(inner_base, r, s, cx);
            let licence = !flat
                && Law::PowCompose.holds_powers(
                    Powers::Rational {
                        base: inner_base,
                        r,
                        s,
                    },
                    cx,
                );
            if licence {
                if let Some(p) = r.checked_mul(s) {
                    return pow_member(inner_base.clone(), Ex::Num(p), cx);
                }
            }
        }
    }
    // The SAME composition where the power is spelled as a FUNCTION: `pow(exp a, b)` and
    // `pow(e, b)` collapse into one `exp`. See `compose_e_power` -- the base is positive
    // everywhere, so no licence is needed, and the fold condition is what keeps the arm
    // descending. Placement is load-bearing: it runs BEFORE the unit-fraction -> `rootn`
    // conversion at the bottom of this function, the rewrite that blinded both
    // `pow np.e _0 -> exp _0` and `pow exp (-1) _0 -> exp neg _0` at a unit-fraction
    // exponent (measured before this arm: `pow(exp(-1), 1/2)` shipped as
    // `rootn(exp(-1), 2)` at mu 20,000 where `exp(-1/2)` is 11,585).
    if let Some(composed) = compose_e_power(&base, &exp, cx) {
        return composed;
    }
    // ROOT/POWER INVERSE under licence: `pow(rootn(b, n), n) -> b`. TOTAL for odd n (the
    // signed root is a bijection on the extended reals: `rootn(-8,3)^3 = -8`); for even n
    // it needs a certainly-non-negative base, because `rootn(b,2)^2` is NaN on `b < 0`
    // where `b` itself is defined -- unlicensed, that would be a positive-measure
    // extension. This restores the licence the pow-of-pow arm used to supply back when
    // even roots were spelled as powers; without it `rootn(abs(x),2)^2` stopped folding
    // to `abs(x)`.
    //
    // The NEGATIVE mirror `pow(rootn(b, n), -n) -> 1/b` carries the same licence by the same
    // argument: reciprocating a total identity is total (`1/nan = nan`, `1/0 = +inf`,
    // `1/+inf = 0`), and the even case needs the same non-negative base. Without it the
    // reduction was ROUTE-DEPENDENT -- `pow(rootn(x,3), -3)` stayed put while its own
    // explicit spelling `inv pow rootn x 3 3` reduced to `inv x`, so the explicit projection
    // was not a fixpoint of the parser.
    if let (Ex::Fun(rop, rargs), Ex::Num(e)) = (&base, &exp) {
        if rargs.len() == 2 && cx.view.tok_is(*rop, "rootn") {
            if let (Ex::Num(idx), true) = (&rargs[1], e.is_integer()) {
                let n = e.abs();
                if n.cmp_int(2) != Ordering::Less
                    && *idx == n
                    && Law::PowRootInverse.holds_powers(
                        Powers::Rational {
                            base: &rargs[0],
                            r: idx,
                            s: e,
                        },
                        cx,
                    )
                {
                    if !e.is_negative() {
                        return rargs[0].clone();
                    }
                    return pow_member(rargs[0].clone(), Ex::int(-1), cx);
                }
            }
        }
    }
    // ODD-ROOT COMPOSITION under licence: `pow(rootn(b, m), s) -> pow(b, s/m)` for an ODD
    // index `m >= 3` and a NON-INTEGER rational `s`.
    //
    // An odd `rootn` is deliberately NOT readable as `b^(1/m)` (`factor_split`), because the
    // signed total root and the principal power are different functions on `b < 0`, so the
    // generic composition licence cannot reach this shape. It is nonetheless TOTAL exactly
    // when `s` is not an integer, and the case split is short: on `b < 0` the left side is a
    // negative real raised to a non-integer power, i.e. NaN, and the right side is `b^(s/m)`
    // whose exponent is also non-integer -- `s = p/q` in lowest terms with `q >= 2` gives
    // `s/m = p/(qm)` with reduced denominator `qm/gcd(p,m) >= q >= 2` -- hence NaN too; on
    // `b > 0` both are `b^(s/m)`; on `b == 0` both are `0^(s/m)`, agreeing at the pole by the
    // sign of the exponent; `+-inf` and NaN follow the same two lines.
    //
    // The INTEGER `s` case is genuinely unsound and stays excluded: `rootn(-8,3)^2 = 4` while
    // `(-8)^(2/3)` is NaN, a positive-measure disagreement.
    if let (Ex::Fun(rop, rargs), Ex::Num(s)) = (&base, &exp) {
        if rargs.len() == 2 && cx.view.tok_is(*rop, "rootn") {
            if let Ex::Num(idx) = &rargs[1] {
                if idx.is_odd_integer()
                    && idx.cmp_int(3) != Ordering::Less
                    && Law::PowOddRootCompose.holds_powers(
                        Powers::Rational {
                            base: &rargs[0],
                            r: idx,
                            s,
                        },
                        cx,
                    )
                {
                    if let Some(t) = idx.checked_inv().and_then(|inv| s.checked_mul(&inv)) {
                        return pow_member(rargs[0].clone(), Ex::Num(t), cx);
                    }
                }
            }
        }
    }
    // A NEGATIVE EVEN UNIT FRACTION over a `rootn` whose index has no known parity (f64 mode,
    // beyond 2^53): `rootn(b, m)^(-1/n) -> 1/rootn(rootn(b, m), n)` for even
    // n. The odd-root composition above needs the parity and refuses, while the printed
    // `inv pow rootn b m 1/n` re-reads through the parity-free even-unit-fraction and
    // root-composition arms; construction takes that route directly. Sound: on R = rootn(b, m)
    // >= 0 both sides are 1/R^(1/n), on R < 0 both are NaN, at R = 0 both are +inf.
    if let (Ex::Fun(rop, rargs), Ex::Num(s)) = (&base, &exp) {
        if rargs.len() == 2 && cx.view.tok_is(*rop, "rootn") && s.is_negative() && !s.is_integer() {
            if let Ex::Num(idx) = &rargs[1] {
                let n = s.denom();
                if idx.is_integer()
                    && idx.parity().is_none()
                    && s.numer() == Rat::NEG_ONE
                    && n.is_even_integer()
                {
                    if let Some(unit) = s.checked_neg() {
                        let root = pow_member(base.clone(), Ex::Num(unit), cx);
                        return pow_member(root, Ex::int(-1), cx);
                    }
                }
            }
        }
    }
    // A LITERAL BASE AT AN INFINITE EXPONENT IS DETERMINED: `b^(+-inf)` is a STEP in |b|,
    // and with `b` a literal the step is decided, so this is exact arithmetic and not a
    // limit notion (the spike-flattener refusal is about a SYMBOLIC base, where the step can
    // be straddled). Judge: CERTIFIED at every case below.
    //
    // The engine already folded the halves that go to +inf (`2^inf`, `(1/2)^(-inf)`) through
    // the determined-reciprocal arm; the halves that go to ZERO were MISSING, and it was the
    // MINER that said so -- the reciprocal-base arm below made `inv pow 2 _0 -> pow 0.5 _0`
    // native, and the re-mine promptly minted eighteen ground rules
    // (`inv pow k float("inf") -> 0`, k in -10..10) to cover what the wildcard rule had been
    // reaching through. An incomplete arm shows up as rules: the same signal that found
    // a missing sign half, a missing write half and the parity arm's abs propagation.
    //
    // |b| = 1 and b = 0 stay OUT: `(-1)^inf` is 1 and `0^inf` is 0, both certified, but
    // neither is reached by the reciprocal arm and neither was asked for here.
    if matches!(&exp, Ex::PosInf | Ex::NegInf) {
        if let Ex::Num(bv) = &base {
            if !bv.is_zero() {
                // |b| against 1, exactly: `Rat` is normalized with a positive denominator,
                // so comparing |num| with den decides it without any division.
                let mag = bv.abs().cmp_int(1);
                let to_zero = matches!(
                    (&exp, mag),
                    (Ex::PosInf, std::cmp::Ordering::Less)
                        | (Ex::NegInf, std::cmp::Ordering::Greater)
                );
                if to_zero {
                    return Ex::Num(Rat::ZERO);
                }
            }
        }
    }
    // A RECIPROCAL BASE AT AN INFINITE EXPONENT MOVES INTO THE EXPONENT'S SIGN:
    // `(1/t)^(+-inf) -> t^(-+inf)`. Judge: CERTIFIED (the general symbolic-base version
    // `pow(1/t, s) -> pow(t, -s)` is only TOLERATED, so this is deliberately the infinite
    // case ALONE, where the pole configurations collapse). The descent is unconditional --
    // the `inv` node disappears and the two infinities cost the same, 32,000 -> 24,000.
    if matches!(&exp, Ex::PosInf | Ex::NegInf) {
        if let Ex::Pow(inner, k) = &base {
            if matches!(&**k, Ex::Num(r) if *r == Rat::NEG_ONE) {
                let flipped = if matches!(&exp, Ex::PosInf) {
                    Ex::NegInf
                } else {
                    Ex::PosInf
                };
                return pow_member((**inner).clone(), flipped, cx);
            }
        }
    }
    // A LITERAL BASE ABSORBS THE EXPONENT'S SIGN: `b^(-t) -> (1/b)^t` for a literal `b`.
    // Judge: CERTIFIED at every literal, including the two the one-zero convention decides
    // (`0^(-t) -> inf^t`, `inf^(-t) -> 0^t`) and negative bases.
    //
    // GATED ON THE FLIP REMOVING A NODE, which is a structural test and NOT a mu guard --
    // the direction is genuinely context-dependent and this is where it turns. Measured:
    //   exponent `Mul[-1, x0]`        26,000 -> 18,585   the Mul wrapper collapses     TAKE
    //   exponent `Mul[-1, sin x0]`    34,000 -> 26,585   same                          TAKE
    //   exponent `Num(-1/2)`          13,585 -> 12,585   sign is free, rootn pays      TAKE
    //   exponent `Num(-1/3)`          14,000 -> 13,585   sign is free                  TAKE
    //   exponent `Mul[-1, x0, x1]`    34,000 -> 34,585   the Mul SURVIVES              REFUSE
    // The reciprocal costs the base exactly one bit (`mu(1/n) = mu(n) + 1`), so the
    // flip pays iff it buys back a whole node. With two or more members the bag survives the
    // flip, nothing is bought, and the arm would ASCEND -- which is why the shipped rules
    // could never be a blanket constructor arm and why this is not one either.
    // The literal's reciprocal, under the one-zero convention the judge priced these against
    // (`1/0 = +inf`). RATIONAL LITERALS ONLY, and both exclusions were paid for:
    //
    //   * `-inf` is REFUSED because the judge KILLs it, bc-positive-measure: `(-inf)^(-t)` is
    //     NaN at every non-integer `t` while `0^t` is 0 there, so the flip would extend
    //     undefined to defined on positive measure. I had written `PosInf | NegInf` from
    //     symmetry WITHOUT asking, and the h045 wall caught it -- the judge-first rule exists
    //     precisely because symmetry is not an argument.
    //   * `+inf` is judge-CERTIFIED but owned by the determined-reciprocal arm above, which
    //     folds `Pow(inf, negative)` to 0 before this arm is reached.
    let literal_reciprocal = |e: &Ex| -> Option<Ex> {
        match e {
            Ex::Num(b) if b.is_zero() => Some(Ex::PosInf),
            Ex::Num(b) => b.checked_inv().map(Ex::Num),
            _ => None,
        }
    };
    if let Some(positive) = cx.carried_negative(&exp) {
        let single_member = match &exp {
            Ex::Num(_) => true,
            Ex::Mul(v) => v.iter().filter(|m| !matches!(m, Ex::Num(_))).count() == 1,
            _ => false,
        };
        if single_member {
            if let Some(r) = literal_reciprocal(&base) {
                return pow_member(r, positive, cx);
            }
        }
    }
    // The `inv` SPELLING of the same fact: `1/(b^t) -> (1/b)^t` at a literal `b`. It arrives
    // as `Pow(Pow(b, t), -1)`, which the pow-of-pow composition above cannot touch -- that
    // arm needs a RATIONAL inner exponent and here `t` is symbolic. No structural gate is
    // needed: the exponent is carried across untouched, so the outer `Pow` node disappears
    // against the base's one bit and the descent is unconditional (26,000 -> 18,585).
    // Judge: CERTIFIED at a literal base (`inv pow 3 _0 -> pow (1/3) _0`), TOLERATED at a
    // symbolic one -- literal only, exactly as above.
    if matches!(&exp, Ex::Num(r) if *r == Rat::NEG_ONE) {
        if let Ex::Pow(ib, ie) = &base {
            if let Some(r) = literal_reciprocal(ib) {
                return pow_member(r, (**ie).clone(), cx);
            }
        }
    }
    // EVEN UNIT-FRACTION EXPONENT -> `rootn`. For an even index the two
    // spellings are the SAME function -- both are the principal root, NaN on negatives --
    // so the orientation is free, and mu now settles it: `rootn x 2` prices 18 against
    // `pow x 1/2` at 19, because a genuine fraction pays for its denominator while the
    // integer index does not. It is also 3 tokens against 7 in the tagged form, and the
    // index stays inside a consumer's integer vocabulary.
    //
    // ODD n must NOT convert, in either direction: `rootn(-8,3) = -2` where `pow(-8,1/3)`
    // is NaN, so those are different functions. Only UNIT fractions convert: `pow(x, 3/2)`
    // is not `rootn(x,2)`, and rewriting it as `pow(rootn(x,2),3)` is a separate question.
    //
    // Runs LAST, so every exact fold, pole, infinity and composition arm above has already
    // had its chance -- `pow(4, 1/2)` still folds to `2` and never reaches here.
    if let Ex::Num(e) = &exp {
        let n = e.denom();
        if e.numer().is_one() && n.is_even_integer() && n.cmp_int(2) != Ordering::Less {
            return fun(cx.view.intern("rootn"), vec![base, Ex::Num(n)], cx);
        }
    }
    // AN EVEN POWER'S BASE: an even integer exponent erases the base's sign, `(-S)^2k = S^2k`
    // exactly, at the poles too (a negative even power is `+inf` at `S = 0` in either orientation,
    // 0 at an infinity), so a mixed-sign sum's orientation is free and both orientations are one
    // value. The base takes the orientation the sum's normal-form choice gives a free site
    // (`flipped_orientation_wins`, `Tie::Lex`: neither spelling distinguished). Const-bearing and
    // absorbing sums keep their absorption owners. Base and exponent are final here, and the flip
    // re-enters `pow` once (the flipped orientation stands, the order being one on the class).
    if let (Ex::Add(ts), Ex::Num(r)) = (&base, &exp) {
        if r.is_even_integer()
            && !r.is_zero()
            && !ts.iter().any(Ex::contains_const)
            && !ts.iter().any(term_absorbs_negation)
            && !ts.iter().any(|x| matches!(x, Ex::PosInf | Ex::NegInf))
            && flipped_orientation_wins(ts, Wrapper::Neither, Tie::Lex, cx)
        {
            if let Some(mut flipped) = ts
                .iter()
                .map(|t| negate_term(t, cx))
                .collect::<Option<Vec<Ex>>>()
            {
                // FILES-BARE gate (same rule as sign_trade_flip): only adopt an
                // orientation primitive_sum itself would file bare.
                if !flipped_orientation_wins(&flipped, Wrapper::Flipped, Tie::Keep, cx) {
                    flipped.sort_by(|a, b| add_term_cmp(a, b, cx.view));
                    return pow_member(Ex::Add(flipped), exp, cx);
                }
            }
        }
    }
    // The same orientation for an odd function of a sum under an even exponent: f(-S)^2k =
    // (-f(S))^2k = f(S)^2k exactly, so `tanh(x2 - x1)^2` and `tanh(x1 - x2)^2` are one value,
    // filed with the argument the sum itself would take.
    if let (Ex::Fun(op, args), Ex::Num(r)) = (&base, &exp) {
        let pos = odd_head(cx.view, *op, args).map(|(p, _)| p);
        let odd = pos.is_some();
        let pos = pos.unwrap_or(0);
        if let (true, Ex::Add(ts)) = (odd && r.is_even_integer() && !r.is_zero(), &args[pos]) {
            if !ts.iter().any(Ex::contains_const)
                && !ts.iter().any(term_absorbs_negation)
                && !ts.iter().any(|x| matches!(x, Ex::PosInf | Ex::NegInf))
                && flipped_orientation_wins(ts, Wrapper::Neither, Tie::Lex, cx)
            {
                if let Some(mut flipped) = ts
                    .iter()
                    .map(|t| negate_term(t, cx))
                    .collect::<Option<Vec<Ex>>>()
                {
                    if !flipped_orientation_wins(&flipped, Wrapper::Flipped, Tie::Keep, cx) {
                        flipped.sort_by(|a, b| add_term_cmp(a, b, cx.view));
                        let mut args = args.clone();
                        args[pos] = Ex::Add(flipped);
                        return pow_member(Ex::Fun(*op, args), exp, cx);
                    }
                }
            }
        }
        // And over a negative literal multiple, the positive one: `sin(-2 x1)^2` is
        // `sin(2 x1)^2`, as the class {sin(2 x1), sin(-2 x1)} is written (`collect_opposite_sums`).
        if r.is_even_integer() && !r.is_zero() {
            if let Some(Ex::Fun(fk, fa)) = positive_head(&base, cx.view) {
                return pow_member(fun(fk, fa, cx), exp, cx);
            }
        }
    }
    Ex::Pow(Box::new(base), Box::new(exp))
}

/// ONE ORIENTATION FOR A CLASS, WHEREVER IT STANDS. A sign-trade site (`sign_trade_flip`: an odd
/// power or an odd root of a sum, an odd function of one) spells its class in either
/// orientation, `f(S)` or `-f(-S)`; as a factor of a product the sign placement picks between
/// them over the whole product (`sign_place`: the cheaper, then the sign key, fewest signs and
/// then the outer ones). A site standing alone takes the spelling that placement gives it as the
/// one factor of a product, so `S^3` and `1*S^3` are one answer, and so are `1/S` and `1/(1*S)`:
/// `(x3*(x3 - x1) - x2)^3` is `-(x2 + x3*(x1 - x3))^3`, whose two signs stand further out.
pub(super) fn lone_site(node: Ex, cx: &Cx) -> Ex {
    // A site standing alone may be the one an inverse reads (`1/S^3`): over a sum that vanishes on
    // a region it keeps the orientation it was written in, which the inverse keeps
    // (`region_signed_zero`, `inverts_a_signed_zero`).
    if region_signed_zero(&node, cx) {
        return node;
    }
    // One scope for the site test and the placement, so the flip is computed once.
    let _memo = FlipScope::open();
    if sign_trade_flip(&node, cx).is_some() {
        sign_place(Rat::ONE, vec![node], cx)
    } else {
        node
    }
}
