//! THE SUM CONSTRUCTOR [`add`]: like terms, shared numbers, the primitive sum and its orientation,
//! and the absorption of infinities.

use super::*;

/// The canonical n-ary addition constructor. See the module doc for the invariants; the gates:
///
/// * NaN element -> NaN (TOTAL: `nan + t = nan` for every extended-real `t`).
/// * `+inf` and `-inf` both present -> NaN (TOTAL: the pair alone forces it pointwise).
/// * One infinity present: it absorbs the rational accumulator and every term with a
///   finite-a.e. licence; absorbing an UNLICENSED term is refused (`inf + log(x)` stays a bag:
///   at `x = 0` the truth is `inf + -inf = nan`, not `inf`).
/// * Like terms merge SAME-SIGN freely (TOTAL: same-sign coefficient addition commutes with the
///   `t = +-inf` points). OPPOSITE-SIGN merging cancels mass and needs the `!`-licence on the
///   key (`2t - t` at `t = +inf` is `nan`; `t` is `+inf`; sound only where that hole is null).
/// * `Const`-containing keys never merge with anything (independence), EXCEPT bare `r*Const`
///   terms, which all collapse into ONE `Const` absorbing the rational accumulator
///   (`c1 + c2 + 5 = c3`, the contract's forall-exists direction).
///
/// The laws applied here (`ac::laws::Law`): `add.flatten`, `add.collect`, `add.unfactor`,
/// `num.literal_arith`, `num.inf_arith`.
pub fn add(items: Vec<Ex>, cx: &Cx) -> Ex {
    // a fully numeric sum is valued whole: its literals combine exactly
    let ground_bag = in_ground(&items, cx);
    // `x + (-0)` is `x` bit for bit in float64, the sign of a zero and NaN included: the written
    // negative zero leaves a sum that holds anything else.
    let items = if cx.f64_semantics()
        && items.len() > 1
        && items.iter().any(|e| is_neg_zero(e, cx.view))
        && !items.iter().all(|e| is_neg_zero(e, cx.view))
    {
        items
            .into_iter()
            .filter(|e| !is_neg_zero(e, cx.view))
            .collect()
    } else {
        items
    };
    // Where the absorption of infinities settles (`permissive`), a part holding an infinity
    // beside a part with a variable is one term to the like-term cancellation, which runs here,
    // before the flatten spreads its terms (`collect_infinite_parts`).
    let items = if items.len() > 1 && Law::InfinityAbsorbs.settles(cx) {
        collect_infinite_parts(items, cx)
    } else {
        items
    };
    // Flatten (Flat) + literal scan.
    let mut lits: Vec<Rat> = Vec::new();
    let mut acc_overflow: Vec<Ex> = Vec::new(); // Num partials an overflowed accumulator emitted
    let mut pos_inf = false;
    let mut neg_inf = false;
    let mut has_const = false;
    // The name the one fitted constant of the sum is written with (`ConstName::merge`).
    let mut const_name = ConstName::ANON;
    let mut terms: Vec<Ex> = Vec::new();
    // Where another addend is never -0 on a region (`add.zero`'s test), the sum is never -0 there
    // either: no zero of a factored sum is read through it, and a negative coefficient spreads
    // whatever the factored sum's zero set (`unfactors`).
    let settled_by_others = f64_zero_signs(cx)
        && items.iter().any(|t| {
            let negative_factored = |f: &Ex| {
                matches!(f, Ex::Mul(v) if factored_sum(v).is_some_and(|(r, _)| r.is_negative()))
            };
            negative_factored(t) || matches!(t, Ex::Add(ts) if ts.iter().any(negative_factored))
        })
        && items.iter().any(|t| {
            let kept = matches!(t, Ex::Mul(v) if factored_sum(v)
                .is_some_and(|(r, inner)| !r.is_zero() && !unfactors(&r, inner, cx)));
            !kept && signed_zero_null(t, true, cx)
        });
    let mut stack: Vec<Ex> = items;
    stack.reverse();
    while let Some(e) = stack.pop() {
        match e {
            Ex::NaN => return Ex::NaN,
            Ex::Add(v) => stack.extend(v.into_iter().rev()),
            // A FACTORED sum entering an Add context unfactors: `r * (a + b)` contributes the
            // terms `ra, rb` (TOTAL: distribution of a finite rational over extended-real
            // sums holds pointwise). Without this, extraction on an inner sub-sum would
            // produce a `Mul[r, Add]` term the enclosing bag cannot see through, and the
            // canonical form would depend on ASSOCIATION order (`(2a + 2b) + c` vs
            // `2a + (2b + c)`). Extraction decisions are thereby always made once, on the
            // complete bag. Scaled terms re-enter the flatten stack so every classification
            // (literals, Const, infinities, nested sums) reruns on them.
            // The literal may be a refused partition (`Mul[1e160, 1e170, Add]`): its members are
            // one coefficient. In `f64` a coefficient that is no positive number unfactors only
            // where the zero's sign stays (`unfactors`): a zero product keeps its sum until the
            // zero products settle below, and a negative coefficient keeps a sum that may vanish
            // on a region, unless another addend settles the sum (`settled_by_others`).
            Ex::Mul(v)
                if factored_sum(&v).is_some_and(|(r, inner)| {
                    unfactors(&r, inner, cx) || (r.is_negative() && settled_by_others)
                }) =>
            {
                let (r, inner) = match factored_sum(&v) {
                    Some((r, inner)) => (r, inner.to_vec()),
                    None => unreachable!("a factored sum"),
                };
                let mut scaled: Vec<Ex> = Vec::with_capacity(inner.len());
                let mut ok = true;
                for t in &inner {
                    match scale_term(t, &r, cx) {
                        Some(s) => scaled.push(s),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    stack.extend(scaled.into_iter().rev());
                } else {
                    terms.push(Ex::Mul(v)); // `0 * inf`: keep the factored term opaque
                }
            }
            Ex::Num(r) => lits.push(r),
            Ex::PosInf => pos_inf = true,
            Ex::NegInf => neg_inf = true,
            Ex::Const(n) => {
                has_const = true;
                const_name = const_name.merge(n, cx.view);
            }
            other => terms.push(other),
        }
    }
    if pos_inf && neg_inf {
        return Ex::NaN;
    }

    // CANONICAL literal partition (`Rat::partition_sum`): the literals summed as
    // far as the sums stay representable, as a fixed point a re-read reproduces.
    lits.sort_unstable_by(|a, b| a.cmp_exact(b));
    // ZERO TERMS IN `f64` (`signed_zero_null`). Literals that sum to zero are float64's `+0` term
    // (`t + 0`, `t + 2 - 2`), and two like terms of opposite signs are a `+0` wherever their key
    // vanishes (`2t - t` is `-0 + 0` where `t = -0`): either way the sum is never `-0`. Where what
    // remains may be `-0` on a region, the sum keeps a `+0` term.
    let mut owes_zero = f64_zero_signs(cx) && !lits.is_empty();
    let mut acc = Rat::ZERO;
    let mut lits = literal_partition_sum(lits, ground_bag, cx);
    if lits.len() == 1 {
        acc = lits.pop().unwrap();
    } else {
        acc_overflow.extend(lits.into_iter().map(Ex::Num));
    }

    // ZERO PRODUCTS IN `f64` (`zero_product`): a zero with a sign is a zero term, whose sign
    // nothing reads where the rest of the sum is not -0 on a region (`add.zero`): a literal, an
    // infinity, a `<constant>` or a term the region test clears. There it drops the factors that
    // only gave it its sign (`unsigned_zero`), and goes where nothing else is left: `x1 + 0*x0` is
    // `x1`, while `ceil(x1) + 0*x0` and `0*x0 + 0*x1` keep their zeros, whose signs decide the
    // sum's.
    let terms: Vec<Ex> = if terms.iter().any(|t| zero_product(t, cx)) {
        let settled = pos_inf
            || neg_inf
            || has_const
            || owes_zero
            || !acc.is_zero()
            || !acc_overflow.is_empty()
            || terms
                .iter()
                .any(|t| !zero_product(t, cx) && Law::AddZero.holds(t, cx));
        if settled {
            let mut out: Vec<Ex> = Vec::with_capacity(terms.len());
            for t in terms {
                settle_zero_term(t, &mut out, cx);
            }
            out
        } else {
            terms
        }
    } else {
        terms
    };
    // Like-term collection: (key, positive-coefficient sum, negative-coefficient sum), in
    // first-seen order for determinism (the final sort canonicalizes anyway).
    struct Bucket {
        key: Ex,
        /// ALL same-key coefficients, collected raw; the fold happens at rebuild in
        /// SORTED order (an arrival-order fold made the overflow partition depend
        /// on the input spelling).
        coeffs: Vec<Coef>,
    }
    let mut buckets: Vec<Bucket> = Vec::new();
    let mut const_terms = false; // any r*Const term seen (merges into the bare-Const pool)
    for t in terms {
        let (c, key) = term_split(t, cx.view);
        if let Ex::Const(n) = key {
            const_name = const_name.merge(n, cx.view);
            // r*Const: joins the bare-Const pool (c1*r1 + c2*r2 = c3, forall-exists;
            // `Law::AbsorbSum`).
            const_terms = true;
            continue;
        }
        if key.contains_const() {
            // Independent constants: never merge; keep verbatim (a one-element bucket).
            buckets.push(Bucket {
                key,
                coeffs: vec![c],
            });
            continue;
        }
        match buckets
            .iter_mut()
            .find(|b| !b.key.contains_const() && b.key == key)
        {
            Some(b) => b.coeffs.push(c),
            None => buckets.push(Bucket {
                key,
                coeffs: vec![c],
            }),
        }
    }
    if has_const || const_terms {
        has_const = true;
        acc = Rat::ZERO; // absorbed into the constant (c + 5 = c', forall-exists)
        acc_overflow.clear();
    }

    // Like terms are like up to sign: terms that differ only in the orientation of their classes
    // (`1/(x3 - x2)` and `-1/(x2 - x3)`) are one term, read with every class in its one orientation
    // and the sign in the coefficient (`orient_term_classes`), so `x + f - f` cancels whatever
    // spelling each `f` arrived in. A sign-blind hash of the keys' leaves gates it: only keys that
    // collide pay for the orientation. A collected term is rebuilt through the constructors, which
    // decide its spelling.
    let mut rebuilt = vec![false; buckets.len()];
    if buckets.len() >= 2 {
        fn holds_sum(e: &Ex) -> bool {
            match e {
                Ex::Add(_) => true,
                Ex::Mul(v) | Ex::Fun(_, v) => v.iter().any(holds_sum),
                Ex::Pow(b, q) => holds_sum(b) || holds_sum(q),
                _ => false,
            }
        }
        let sig: Vec<Option<u64>> = buckets
            .iter()
            .map(|b| {
                (!b.key.contains_const()
                    && matches!(b.key, Ex::Mul(_) | Ex::Pow(..) | Ex::Fun(..))
                    && holds_sum(&b.key))
                .then(|| leaf_sig(&b.key))
            })
            .collect();
        let hit: Vec<usize> = (0..buckets.len())
            .filter(|&i| {
                sig[i].is_some() && sig.iter().enumerate().any(|(j, x)| j != i && *x == sig[i])
            })
            .collect();
        if !hit.is_empty() {
            let read: Vec<(Ex, bool)> = hit
                .iter()
                .map(|&i| match orient_term_classes(&buckets[i].key, cx) {
                    Some(o) => match term_split(o, cx.view) {
                        (c, k) if c.is_one() => (k, false),
                        (c, k) if c.is_neg_one() => (k, true),
                        _ => (buckets[i].key.clone(), false),
                    },
                    None => (buckets[i].key.clone(), false),
                })
                .collect();
            let mut gone = vec![false; buckets.len()];
            for a in 0..hit.len() {
                if gone[hit[a]] {
                    continue;
                }
                let group: Vec<usize> = (a..hit.len())
                    .filter(|&b| !gone[hit[b]] && read[b].0 == read[a].0)
                    .collect();
                // In `f64` a class turned around is no negation where its sum vanishes on a region
                // (both orientations of `x1 - |x1|` are `+0` for `x1 >= 0`), so terms whose key may
                // vanish on a region stay apart (`signed_zero_null`).
                if group.len() < 2 || vanishes_on_a_region(&read[a].0, cx) {
                    continue;
                }
                let mut coeffs: Vec<Coef> = Vec::new();
                for &b in &group {
                    for c in &buckets[hit[b]].coeffs {
                        coeffs.push(if read[b].1 { c.negated() } else { c.clone() });
                    }
                }
                let target = hit[group[0]];
                buckets[target].key = read[a].0.clone();
                buckets[target].coeffs = coeffs;
                rebuilt[target] = true;
                for &b in &group[1..] {
                    gone[hit[b]] = true;
                }
            }
            if gone.iter().any(|&g| g) {
                let mut keep = gone.iter().map(|g| !g);
                buckets.retain(|_| keep.next().unwrap_or(true));
                let mut keep = gone.iter().map(|g| !g);
                rebuilt.retain(|_| keep.next().unwrap_or(true));
            }
        }
    }

    // Rebuild terms, applying the opposite-sign gate per bucket. A rebuilt term can itself be
    // an `Add`: when merged coefficients sum to exactly 1 on an Add-valued key
    // (`(1/4)A + (3/4)A -> A`), `term_join` returns the bare key -- SPLICE it, or the outer bag
    // would nest a same-kind bag and break the Flat invariant.
    let mut out: Vec<Ex> = Vec::new();
    let mut spliced = false;
    let push_term = |out: &mut Vec<Ex>, spliced: &mut bool, t: Ex| match t {
        Ex::Add(v) => {
            out.extend(v);
            *spliced = true;
        }
        // `term_join` on a partition-bag key routes through `mul()`, which may
        // COMPLETE the fold (the regrouped product fits i128 after all) and hand back
        // a bare literal. That literal must re-enter literal accumulation beside
        // `acc` -- parking it as a term would leave two unfolded literal members in
        // one bag. Ride the splice re-run (same termination argument: the completed
        // fold strictly reduced node count).
        t @ Ex::Num(_) => {
            out.push(t);
            *spliced = true;
        }
        // A collected coefficient on a sum the flatten kept whole (a negative one beside a sum
        // that may vanish on a region) may be one the sum unfactors: the re-run spreads it.
        Ex::Mul(v)
            if factored_sum(&v)
                .is_some_and(|(r, inner)| !r.is_zero() && unfactors(&r, inner, cx)) =>
        {
            out.push(Ex::Mul(v));
            *spliced = true;
        }
        t => out.push(t),
    };
    for (b, rebuilt) in buckets.into_iter().zip(rebuilt) {
        // DETERMINISTIC per-key fold (`Rat::partition_sum`). A bucket that mixes
        // signs on a key that is finite a.e. may cancel across them (`x - x = 0` needs the
        // licence), so its coefficients fold as ONE signed partition; otherwise each sign pool
        // folds on its own (a same-sign sum is total). Either way no two members left in a
        // pool fold, so the bucket re-reads to itself in any printed order. Folding the pools
        // first and cancelling only two folded totals did not: a printed bag
        // `x1 - C*x1 - D*x1` whose negative pool refused re-read `x1 - C*x1` first and
        // cancelled it.
        let Bucket { key, coeffs } = b;
        let mixed =
            coeffs.iter().any(|c| c.is_negative()) && coeffs.iter().any(|c| !c.is_negative());
        let pools: Vec<Vec<Coef>> = if mixed && Law::AddCancel.holds(&key, cx) {
            owes_zero |= f64_zero_signs(cx);
            vec![coeffs]
        } else {
            let (negs, poss): (Vec<Coef>, Vec<Coef>) =
                coeffs.into_iter().partition(|c| c.is_negative());
            vec![poss, negs]
        };
        for mut pool in pools {
            // Each coefficient may be a refused partition, folded by `Coef::checked_add`.
            pool.retain(|c| !c.is_zero());
            for c in
                crate::ac::rat::partition_by(pool, Coef::order, Coef::checked_add, Coef::is_zero)
            {
                // a fully cancelled bucket leaves nothing: 0 * (finite-a.e. t) -> 0 licensed
                let t = if rebuilt && !c.is_zero() {
                    let mut items = c.items();
                    items.push(canon(key.clone(), cx));
                    mul(items, cx)
                } else {
                    term_join(c, key.clone(), cx)
                };
                push_term(&mut out, &mut spliced, t);
            }
        }
    }
    out.extend(acc_overflow);
    if spliced {
        // A spliced Add's terms must COLLECT against the outer bag's other terms (the splice
        // alone leaves e.g. a bag-resident `2x` and a spliced `x` unmerged, breaking canon
        // idempotence). Re-run the constructor over everything; terminates because each splice
        // removes one bag node.
        if pos_inf {
            out.push(Ex::PosInf);
        }
        if neg_inf {
            out.push(Ex::NegInf);
        }
        if has_const {
            out.push(Ex::Const(const_name));
        }
        if !acc.is_zero() {
            out.push(Ex::Num(acc));
        } else if owes_zero {
            out.push(Ex::Num(Rat::ZERO));
        }
        return add(out, cx);
    }

    // Sign-fold congruence at the TERM level: `term_join` builds raw products (no
    // `mul()` pass -- that is its point), so a collected, scaled or negated term can
    // carry a negative coefficient beside an Add factor that `mul()`'s sign-fold
    // clauses ban from canonical states: Const-bearing (the stability battery
    // caught a chain term `-1 * (x15 + C*x3) * (..C..)` whose re-parse folded while
    // the term did not), bare-inf-bearing (the bare-inf arm; `-(P + inf)*(T)` shipped as a
    // subtracted section while re-parse routed the same value through `mul()` and
    // folded the sign INTO the inf sum -- 2/1M fuzz rows 212777/914946), or holding a
    // negation-ABSORBING member (the absorbing arm, incl. its recursive nestings). The
    // predicate mirrors the UNION of the three `mul()` arms -- a re-settle that
    // matched fewer shapes would ship terms `mul()` itself refuses to build.
    // Re-settle exactly those terms through `mul()`, which owns the fold. Terminates
    // by structural descent: the nested settle works on strictly deeper subterms.
    let mut resettled_reshaped = false;
    for t in out.iter_mut() {
        if let Ex::Mul(v) = t {
            let neg_coeff = v.iter().any(|f| matches!(f, Ex::Num(r) if r.is_negative()));
            let banned_add = v.iter().any(|f| {
                matches!(f, Ex::Add(ts) if ts.iter().any(|m| {
                    m.contains_const()
                        || matches!(m, Ex::PosInf | Ex::NegInf)
                        || term_absorbs_negation(m)
                }))
            });
            if neg_coeff && banned_add {
                let v = std::mem::take(v);
                *t = mul(v, cx);
                resettled_reshaped |= !matches!(t, Ex::Mul(_));
            }
        }
    }
    // A re-settled term can leave the Mul shape only from the exact input
    // `Mul[-1, Add[..Const..]]` (any other coefficient magnitude or extra factor keeps
    // the Mul; the collapse is the bare flipped Add) -- believed UNREACHABLE here: the
    // flatten distributes every 2-element `Mul[Num, Add]` (its scale never overflows at
    // r = -1: no Rat holds i128::MIN), and Const-bearing bucket keys never merge to
    // mint the shape at rebuild (independence keeps them one-element). Handle it anyway
    // so the bag invariants (I1/I3) are local to this function rather than a
    // consequence of that argument: re-enter the constructor exactly like the splice
    // path (same pending-flag tail; terminates because the Const sign fold's winning
    // orientation does not re-fire).
    if resettled_reshaped {
        if pos_inf {
            out.push(Ex::PosInf);
        }
        if neg_inf {
            out.push(Ex::NegInf);
        }
        if has_const {
            out.push(Ex::Const(const_name));
        }
        if !acc.is_zero() {
            out.push(Ex::Num(acc));
        } else if owes_zero {
            out.push(Ex::Num(Rat::ZERO));
        }
        return add(out, cx);
    }

    // One infinity: absorb what is licensed; keep the bag if anything is not.
    // `primitive_sum` is DELIBERATELY skipped for the kept bag, and loses nothing: the
    // infinity pins the magnitude multiset at 1 (it absorbs scalar magnitudes exactly),
    // so no unanimous content != 1 can exist, and the sign family closes without the
    // pass -- +-1 coefficients distribute totally (`scale_term` flips the infinity) and
    // per-term signs live in the sections, so every negation spelling lands in this same
    // flat bag (docs/formal.md, "infinity-bearing sums are stored flat"). Consequence:
    // bare infinities NEVER reach `primitive_sum`.
    if pos_inf || neg_inf {
        let inf = if pos_inf { Ex::PosInf } else { Ex::NegInf };
        // Rational accumulator and Const absorb totally (finite by construction). Where the
        // absorption is a settling step (`permissive`), every other term stays for the pass.
        let unlicensed: Vec<Ex> = out
            .drain(..)
            .filter(|t| !Law::InfinityAbsorbs.constructs(t, cx))
            .collect();
        if unlicensed.is_empty() {
            return inf;
        }
        let mut v = unlicensed;
        v.push(inf);
        v.sort_by(|a, b| add_term_cmp(a, b, cx.view));
        return Ex::Add(v);
    }
    // INFINITE-TERM ABSORPTION (class C): a term that is
    // an infinity-carrying product takes values only in {±inf, NaN} (a canonical Mul
    // keeps at most one ±inf factor; inf * 0 = NaN, inf * finite = ±inf, NaN
    // propagates), and a sum of such terms stays in {±inf, NaN}. Every co-term F
    // with the finite-a.e. licence is absorbed: F + I = I wherever F is finite (a
    // finite F vanishes against I = ±inf; I = NaN forces both sides NaN), so the
    // disagreement set lives inside {F not finite} -- null under the `!`-licence.
    // The rational accumulator and a fitted Const are finite by construction and
    // absorb totally. This generalizes the literal-infinity arm above to
    // variable-bearing infinite terms (`x + inf*x -> inf*x`); the kept bag falls
    // through to the ordinary tail (inf-carrying PRODUCTS, unlike bare infinities,
    // always went through `primitive_sum` -- their coefficient is pinned at 1
    // because `mul` folds any scalar into the infinity, so no unanimous content
    // can arise and the pass is a no-op on them).
    if out.iter().any(is_inf_carrying) {
        out.retain(|f| is_inf_carrying(f) || !Law::InfinityAbsorbs.constructs(f, cx));
        acc = Rat::ZERO;
        has_const = false;
        owes_zero = false;
    }
    if has_const {
        out.push(Ex::Const(const_name));
    } else if !acc.is_zero() {
        // The constant term LANDS beside other terms here (permissive's ρ, [`land`]); a sum
        // of literals alone stays exact, and a sum of ground terms.
        let acc = if out.is_empty() || in_ground(&out, cx) {
            acc
        } else {
            land(acc, cx)
        };
        out.push(Ex::Num(acc));
    } else if owes_zero && !out.is_empty() && !out.iter().any(|t| Law::AddZero.holds(t, cx)) {
        // Every remaining term may be `-0` on a region: the zero term stays (`ceil(x1) + 0`), so
        // the sum is `+0` there as float64's is. Where one term cannot be (`x1 + 0` is `x1`, -0 at
        // the one point `x1 = -0`), it goes.
        out.push(Ex::Num(Rat::ZERO));
    }

    match out.len() {
        0 => Ex::Num(Rat::ZERO),
        1 => out.pop().unwrap(),
        _ => {
            out.sort_by(|a, b| add_term_cmp(a, b, cx.view));
            primitive_sum(out, cx)
        }
    }
}

/// The TERMS OF A SUM AS ITS FLATTEN READS THEM: where a sum writes the number its terms share
/// once ([`group_shared_numbers`]), each group `c*(a + b)` is its terms `c*a`, `c*b`, which is how
/// `add` reads it on entering a sum. The search proposes its factorings on these terms, so a group
/// never hides a term from a factoring across it (`x14*x16 - 2*(x16 + x6)` offers
/// `x16*(x14 - 2) - 2*x6`). A group whose coefficient the flatten keeps whole (`add.unfactor`'s
/// premise) stays one term.
pub(crate) fn spread_groups<'t>(terms: &'t [Ex], cx: &Cx) -> std::borrow::Cow<'t, [Ex]> {
    let group = |t: &Ex| matches!(t, Ex::Mul(v) if factored_sum(v).is_some());
    if !terms.iter().any(group) {
        return std::borrow::Cow::Borrowed(terms);
    }
    let mut out: Vec<Ex> = Vec::with_capacity(terms.len() + 2);
    for t in terms {
        if let Ex::Mul(v) = t {
            if let Some((r, inner)) = factored_sum(v).filter(|(r, inner)| unfactors(r, inner, cx)) {
                let scaled: Option<Vec<Ex>> = inner.iter().map(|s| scale_term(s, &r, cx)).collect();
                if let Some(scaled) = scaled {
                    out.extend(scaled);
                    continue;
                }
            }
        }
        out.push(t.clone());
    }
    std::borrow::Cow::Owned(out)
}

/// A SUM GROUPS A SHARED NUMBER (`add.group`), in every mode: the terms of a
/// flattened sum, grouped by the exact value of their coefficient up to sign (a literal term's
/// coefficient is itself), each group of two or more written `c*(group)` with its content and
/// orientation decided as a whole sum's ([`whole_primitive_sum`]), the sign in the coefficient:
/// `2*x1 + 2*x2 + x3` is `2*(x1 + x2) + x3`, `3.71649*x1 - 3.71649*x2 + x3` is
/// `3.71649*(x1 - x2) + x3`, and `3.71649*x1 + 3.71649*x2 + 5.11927*x3 + 5.11927*x4` writes each
/// number once. The key is the exact value, so the grouping is deterministic, and every
/// construction decides it on the complete flattened bag (`add` spreads a group entering a sum
/// first), so it does not depend on how the sum was built. A coefficient of magnitude one is no
/// number to share, and those terms stay as they are; so do a term holding a fitted constant
/// (absorption owns its coefficient) and a factored sum the flatten kept whole.
///
/// IN `f64` A GROUP IS THE WHOLE-SUM FACTORING OF ITS TERMS, with that factoring's premise
/// (`add.content`): `c*a + c*b` and `c*(a + b)` have one real value and differ in float64 by
/// rounding alone, which strict admits (float64's kinds are kept: both sides finite, the same
/// infinity or both undefined), and a positive content moves no zero's sign; a sign leaves the
/// group only where the group's sum holds none (`Law::AddContent`, read by
/// [`whole_primitive_sum`]), so `x2 - 2*x0 - 2*x1` is `x2 - 2*(x0 + x1)`, while under an inverse
/// `ceil(x2) - 2*ceil(x0) - 2*floor(x1)` is `ceil(x2) + 2*(-ceil(x0) - floor(x1))`. A group the
/// flatten meets again spreads as `add.unfactor` says, and is grouped again here: the two
/// premises are one test.
///
/// A GROUP IS WRITTEN ONLY WHERE IT COSTS NO MORE than its terms distributed: the price of
/// `c*(group)` against the prices of its terms, the members a sum prices one by one. A number
/// written once saves its own price each time it is not written again, which an inexact number
/// always pays for (`3.71649*(x1 + x2)` saves a constant); an exact number whose group costs
/// more stays distributed (`x13/2 + x1/(4*x13) + 1/4` keeps its quarters, where
/// `(x1/x13 + 1)/4` is dearer), so a grouping never prices a sum above its distributed terms.
/// The price is a function of the group's terms alone, so the choice is deterministic. The
/// remaining sum then takes its orientation as any sum does. `None` where no group is written
/// beside other terms: with the whole sum as one group the sum's content is the content
/// [`whole_primitive_sum`] extracts.
fn group_shared_numbers(terms: &[Ex], cx: &Cx) -> Option<Ex> {
    if terms.len() < 3 {
        return None;
    }
    let magnitude = |t: &Ex| -> Option<Coef> {
        if t.contains_const() || is_inf_carrying(t) {
            return None;
        }
        let c = match t {
            Ex::Num(r) => Coef::of(r),
            Ex::Mul(v) if factored_sum(v).is_some() => return None,
            _ => term_split(t.clone(), cx.view).0,
        };
        let m = c.abs();
        (!m.is_one() && !m.is_zero()).then_some(m)
    };
    let mut groups: Vec<(Coef, Vec<usize>)> = Vec::new();
    for (i, t) in terms.iter().enumerate() {
        let Some(m) = magnitude(t) else { continue };
        match groups.iter_mut().find(|(g, _)| *g == m) {
            Some((_, members)) => members.push(i),
            None => groups.push((m, vec![i])),
        }
    }
    groups.retain(|(_, members)| members.len() >= 2);
    if groups.is_empty() || (groups.len() == 1 && groups[0].1.len() == terms.len()) {
        return None;
    }
    let mut grouped = vec![false; terms.len()];
    let mut out: Vec<Ex> = Vec::with_capacity(terms.len());
    for (_, members) in &groups {
        let mut group: Vec<Ex> = members.iter().map(|&i| terms[i].clone()).collect();
        group.sort_by(|a, b| add_term_cmp(a, b, cx.view));
        let distributed = group
            .iter()
            .fold(0u64, |p, t| p.saturating_add(complexity(t, cx.view)));
        // A content that does not divide out (an overflow) leaves the group's terms as they are,
        // and so does a group that costs more than its terms.
        let written = whole_primitive_sum(group, cx);
        if matches!(written, Ex::Add(_)) || complexity(&written, cx.view) > distributed {
            continue;
        }
        for &i in members {
            grouped[i] = true;
        }
        out.push(written);
    }
    if out.is_empty() {
        return None;
    }
    out.extend(
        terms
            .iter()
            .zip(&grouped)
            .filter(|(_, &g)| !g)
            .map(|(t, _)| t.clone()),
    );
    out.sort_by(|a, b| add_term_cmp(a, b, cx.view));
    Some(whole_primitive_sum(out, cx))
}

/// The PRIMITIVE-SUM normal form: a canonical `Add` carries the ORIENTATION-MAXIMAL sign and
/// no UNANIMOUS common coefficient magnitude. The extracted unit/content wraps the sum in a
/// coefficient (`-a - b -> -1 * (a + b)`; `a/3 + b/3 -> 1/3 * (a + b)`; `-2a + 3b ->
/// -1 * (2a - 3b)`), which is exactly the classical content/primitive-part decomposition
/// restricted to the sign-and-unanimity case -- the restriction keeps extraction
/// length-neutral-or-shortening (`x8 + 1.2*x3` has no unanimous content and stays put).
///
/// This closes every sign/grouping spelling family in the CONSTRUCTORS, with no iteration:
/// the class {u * A : u rational != 0, A primitive} has exactly one representative
/// `Mul[u, A]` (or `A` itself when u == 1), and every constructor path lands on it because
/// the sign decision is a function of the orientation CLASS {A, -A} (see
/// `flipped_orientation_wins`), never of any single lead term. Soundness: dividing every
/// term by a finite nonzero rational and multiplying the sum by it is TOTAL over the
/// extended reals (`u*(x+y)` and `ux + uy` agree at every point, including infinities and
/// NaN); `Const` terms pass through unchanged (`c / u` re-fits to `c'`, the contract's
/// forall-exists direction).
///
/// A sum first groups the numbers its terms share ([`group_shared_numbers`]); the whole sum as
/// one group is the content extracted here.
///
/// The laws applied here (`ac::laws::Law`): `add.content`, `add.group`.
fn primitive_sum(terms: Vec<Ex>, cx: &Cx) -> Ex {
    // Bare infinities never arrive here: the `add()` constructor's one-infinity exit
    // returns its flat bag before this call (deliberately -- see the closure argument
    // at that exit).
    debug_assert!(
        !terms.iter().any(|t| matches!(t, Ex::PosInf | Ex::NegInf)),
        "primitive_sum received a bare infinity: the add() inf-exit must precede it"
    );
    if let Some(grouped) = group_shared_numbers(&terms, cx) {
        return grouped;
    }
    whole_primitive_sum(terms, cx)
}

/// [`primitive_sum`] of the whole sum: its unanimous content and its orientation.
fn whole_primitive_sum(terms: Vec<Ex>, cx: &Cx) -> Ex {
    // A Const-bearing sum NEVER extracts content: absorption owns it (`mul`'s Const
    // fold arm unwraps `-1 x Add[Const-bearing]` through per-term `negate_term`,
    // where Const-carrying terms absorb the sign by the forall-exists refit), so
    // wrapping one here would ping-pong with that arm (measured: stack overflow on
    // the degenerate-config battery when the exemption was gated on the FIRST term
    // being constlike -- a Const-bearing sum led by an ordinary term slipped through
    // to extraction and re-wrapped what the fold arm had just unwrapped). The flat
    // bag is already unique for the class: Const terms are sign-neutral at term
    // level, every other term carries its true sign as ordinary structure.
    // ATOM-bearing constlike sums (pi/e with numerals) do NOT fold -- exact folds
    // are mu-refused, the bag persists -- so they need the orientation/content
    // decision like every other sum (`-2 - e` was the unoriented
    // all-negative bag while parse of its own display built the `Mul[-1, Add[2, e]]`
    // wrap: two states, one spelling).
    if terms.is_empty() || terms.iter().any(Ex::contains_const) {
        return Ex::Add(terms);
    }
    // Unanimity: every term's |coefficient| equal (Const counts as 1; NaN cannot appear --
    // it collapses the whole sum earlier).
    // A coefficient a fold refused is one magnitude with all its members (`Coef`).
    let mut magnitudes: Vec<Coef> = Vec::with_capacity(terms.len());
    for t in &terms {
        let c = match t {
            Ex::Const(_) => Coef::ONE,
            Ex::Num(r) => Coef::of(r),
            _ => term_split(t.clone(), cx.view).0,
        };
        magnitudes.push(c.abs());
    }
    let unanimous = magnitudes.windows(2).all(|w| w[0] == w[1]);
    // A sum holding a negation-absorbing member (see `term_absorbs_negation`)
    // never takes a SIGN decision here -- absorption owns the sign, exactly like the
    // Const-bearing skip above (and with the same ping-pong risk: `mul()`'s absorbing
    // sign arm folds a negative coefficient into this class unconditionally, so
    // wrapping one here would re-mint what that arm just unwrapped). Magnitude-only
    // content extraction stays: a positive u moves no signs, so the absorbing member's
    // canonical spelling is untouched.
    let g = if unanimous {
        magnitudes[0].clone()
    } else {
        Coef::ONE
    };
    // The orientation is a normal-form choice (`flipped_orientation_wins`): the cheaper spelling,
    // the tie order on an equal price. For a content-1 sum the flip's wrapper costs one sign (the
    // negation `-1 x Add` exists only in the flipped spelling); for a non-unit content the
    // wrapper exists in both spellings and the flip pays only the coefficient's sign bit, the
    // per-term deltas priced on the divided terms (the spellings stored), so `g`'s sign bit
    // counts once.
    let mut primitive: Option<Vec<Ex>> = None;
    let sign_neg = if terms.iter().any(term_absorbs_negation) {
        false
    } else if g.is_one() {
        flipped_orientation_wins(&terms, Wrapper::Flipped, Tie::Keep, cx)
    } else {
        match divide_terms(&terms, &g, cx) {
            Some(divided_pos) => {
                let wsd = match g.rat() {
                    Some(r) => {
                        mu_rat(&r.checked_neg().expect("a Rat negates")) as i128
                            - mu_rat(&r) as i128
                    }
                    // A partition's sign is a member of its own (`Mul[-1, 1e160, 1e170, S]`):
                    // the two wrappers priced as they stand.
                    None => {
                        let wrap = |c: &Coef| {
                            let mut v = c.items();
                            v.push(Ex::Add(divided_pos.clone()));
                            complexity(&Ex::Mul(v), cx.view) as i128
                        };
                        wrap(&g.negated()) - wrap(&g)
                    }
                };
                let wins =
                    flipped_orientation_wins(&divided_pos, Wrapper::Both(wsd), Tie::Keep, cx);
                primitive = Some(divided_pos);
                wins
            }
            None => return Ex::Add(terms),
        }
    };
    // `add.content`'s premise in `f64`: the sign leaves the sum only where its zero's sign stays.
    // Float64's `-a - b` is `+0` wherever the sum cancels, and `-(a + b)` is `-0` there, so a sum
    // that may vanish on a region keeps its orientation (`sum_holds_its_sign`):
    // `-ceil(x0) - floor(x1)` stays as it is, `-x0 - x1` is `-(x0 + x1)`. A positive content moves
    // no sign. A content held out of the sum leaves its primitive part, the sum the state holds
    // (`2*(-ceil(x0) - floor(x1))` holds `-ceil(x0) - floor(x1)`), and the root's walk names the
    // sums whose zero is read by the spelling it finds (`release_unread`): the premise reads both
    // spellings, so a group a flatten spreads and gathers again keeps the sign it held
    // (`ceil(x2) + 2*(-ceil(x0) - floor(x1))` under an inverse).
    let sign_neg = sign_neg
        && Law::AddContent.holds(&Ex::Add(terms.clone()), cx)
        && primitive
            .as_ref()
            .is_none_or(|p| Law::AddContent.holds(&Ex::Add(p.clone()), cx));
    let u = if sign_neg { g.negated() } else { g };
    if u.is_one() || u.is_zero() {
        return Ex::Add(terms);
    }
    // Divide every term by u (exact; total -- see the doc above).
    let Some(mut divided) = divide_terms(&terms, &u, cx) else {
        return Ex::Add(terms);
    };
    // Re-sort (coefficient ties may reorder) and wrap. `mul`'s sign-orientation arm
    // re-enters `add` only when the FLIPPED orientation wins -- and `divided`
    // is the winning orientation by construction (sign_neg came from the same
    // decision), so this tail call cannot recurse back into `add`.
    divided.sort_by(|a, b| add_term_cmp(a, b, cx.view));
    // Defense-in-depth: with the class-level sign decision a self-reproducing division can
    // no longer be SELECTED (a fully flip-symmetric bag compares equal in both orientations
    // and keeps sign +, and `u == 1` returned above) -- but refuse it outright if one ever
    // slips through, since wrapping a bag in a coefficient that divides back to the same
    // bag is a non-terminating spelling. (A residual, documented non-uniqueness remains for
    // ENCLOSING coefficients of self-negative sums: `Mul[2, S]` and `Mul[-2, S]` denote the
    // same 0-or-NaN function and both persist; measure-zero corner, the certificates cancel
    // these bags wherever they are licensed.)
    if divided == terms {
        return Ex::Add(terms);
    }
    let mut items = u.items();
    items.push(Ex::Add(divided));
    mul(items, cx)
}

/// Divide every term by `u` exactly (`None` on any overflow) -- the same loop as
/// `primitive_sum`'s tail, factored so the orientation decision can price the DIVIDED
/// candidates for a non-unit content.
fn divide_terms(terms: &[Ex], u: &Coef, cx: &Cx) -> Option<Vec<Ex>> {
    let mut divided: Vec<Ex> = Vec::with_capacity(terms.len());
    for t in terms {
        let d = match t {
            Ex::Const(n) => Ex::Const(*n),
            Ex::Num(r) => Ex::Num(Coef::of(r).over(u)?.rat()?),
            _ => {
                let (c, key) = term_split(t.clone(), cx.view);
                term_join(c.over(u)?, key, cx)
            }
        };
        divided.push(d);
    }
    Some(divided)
}

/// The structural part of the tie order between a sum's two orientations, where the price and the
/// sign key (`sign_key`) leave them tied.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Tie {
    /// At a sign-trade site the current spelling is the distinguished one (bare, its coefficient
    /// positive), and it stands.
    Keep,
    /// At a free-orientation site (an even power's base, an even function's argument) neither
    /// spelling is distinguished, both bare: the sorted coefficient sequences compare
    /// lexicographically (exact rational order) and the larger stands, an order on the orientation
    /// class, so both entry spellings land on one representative. Position by position the
    /// stripped keys agree (their multiset is negation-invariant), and the first asymmetric position
    /// decides, also with a flip-symmetric pair `{t, -t}` the certificates do not cancel: `-a - b`
    /// flips to `-1 * (a + b)`, `-2a + 3b` to `-1 * (2a - 3b)`, `a - b` stays. A `Const` term is
    /// sign-free (a negation re-fits it) and compares equal at its position; an infinity carries its
    /// sign.
    Lex,
}

/// Where the wrapper of a sum's flipped orientation stands (`flipped_orientation_wins`).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Wrapper {
    /// In neither spelling: a free-orientation site (an even power's base, an even function's
    /// argument), where both orientations are bare.
    Neither,
    /// In the flipped spelling only: a content-1 sum, whose flip is the negation `Mul[-1, .]`,
    /// one sign (`mu_sign`).
    Flipped,
    /// In both: a non-unit content `g`, written `g x .` and flipped `-g x .`; the flip pays the
    /// price of `-g` over `g`.
    Both(i128),
}

/// THE ORIENTATION OF A SUM, `A` or `-A`: does the flipped spelling win? A normal-form choice,
/// like every sign and orientation choice the constructors make: at the node, given its children,
/// the cheaper admitted spelling (`complexity`), and on an equal price the tie order, the sign key
/// (`sign_key`: fewer signs, then the outer ones), then the structure (`Tie`). The candidates are
/// admitted by the premises, never by the price: a sum that holds its sign
/// (`sum_holds_its_sign`) is not turned around by its callers. Every other price comparison
/// belongs to the settling and the search. `false` (the spelling stands) where a negation
/// overflows.
pub(super) fn flipped_orientation_wins(terms: &[Ex], wrapper: Wrapper, tie: Tie, cx: &Cx) -> bool {
    // A decision of the call's own contexts is made once in the call (`CallDecisions`).
    let Some(ctx) = call_decision_key(cx) else {
        return orientation_decided(terms, wrapper, tie, cx);
    };
    let key = (ctx, terms.to_vec(), wrapper, tie);
    if let Some(hit) = CALL_DECISIONS.with(|m| {
        m.borrow()
            .as_ref()
            .and_then(|d| d.orientations.get(&key).copied())
    }) {
        return hit;
    }
    let r = orientation_decided(terms, wrapper, tie, cx);
    CALL_DECISIONS.with(|m| {
        if let Some(d) = m.borrow_mut().as_mut() {
            d.orientations.insert(key, r);
        }
    });
    r
}

/// [`flipped_orientation_wins`], decided.
fn orientation_decided(terms: &[Ex], wrapper: Wrapper, tie: Tie, cx: &Cx) -> bool {
    // The cheaper orientation wins. The wrapper's own price
    // delta, from the caller's side: one sign for a content-1 sum (the negation exists only
    // in the flipped spelling), the coefficient's sign-bit delta for a non-unit content
    // (wrapper in both spellings), nothing at a free-orientation site.
    let mut delta = match wrapper {
        Wrapper::Neither => 0,
        Wrapper::Flipped => mu_sign() as i128,
        Wrapper::Both(d) => d,
    };
    let mut flipped: Vec<Ex> = Vec::with_capacity(terms.len());
    for t in terms {
        match negate_term(t, cx) {
            Some(n) => {
                delta += complexity(&n, cx.view) as i128 - complexity(t, cx.view) as i128;
                flipped.push(n);
            }
            None => return false,
        }
    }
    if delta != 0 {
        return delta < 0;
    }
    // PERMISSIVE'S TWO SPELLINGS TIE IN PRICE: the one strict prices lower is taken
    // ([`strict_price`]: the spelling read by strict's constructors, priced), so where the modes
    // read a spelling apart (a sign strict holds in a sum, a pole's orientation) a permissive
    // answer is no dearer than strict's on strict's yardstick. Where strict prices them alike
    // too, the tie order below.
    if matches!(cx.mode, RuleMode::Permissive) {
        let negated = !matches!(wrapper, Wrapper::Neither);
        let (a, b) = (
            strict_price(terms, false, cx),
            strict_price(&flipped, negated, cx),
        );
        if a != b {
            return b < a;
        }
    }
    // EXACT mu tie -> the one sign key (`sign_key`): fewer signs, then outer ones. The flipped
    // spelling carries the wrapper's sign (`-1 x Add` for a content-1 sum, `-g x Add` beside a
    // current `g x Add`), except at the free-orientation sites (`Lex`: an even power's base),
    // where neither spelling has one.
    let terms_key = |ts: &[Ex], depth: u32| {
        ts.iter()
            .map(|t| sign_key(t, cx.view))
            .fold((0u32, 0u32), |a, (c, s)| (a.0 + c, a.1 + s + c * depth))
    };
    let (current, flip) = match tie {
        Tie::Lex => (terms_key(terms, 1), terms_key(&flipped, 1)),
        _ => {
            let depth = if matches!(wrapper, Wrapper::Both(_)) {
                2
            } else {
                1
            };
            let (c, s) = terms_key(&flipped, 2);
            (terms_key(terms, depth), (c + 1, s + 1))
        }
    };
    if current != flip {
        return flip < current;
    }
    // Equal signs too: the structure (`Tie`). At a free-orientation site the sorted coefficient
    // sequences compare, an order on the orientation class (both entry spellings land on one
    // representative).
    match tie {
        Tie::Keep => return false,
        Tie::Lex => {}
    }
    flipped.sort_by(|a, b| add_term_cmp(a, b, cx.view));
    for (a, b) in terms.iter().zip(flipped.iter()) {
        let ca = orientation_coeff(a, cx.view);
        let cb = orientation_coeff(b, cx.view);
        match ca.order(&cb) {
            std::cmp::Ordering::Less => return true,
            std::cmp::Ordering::Greater => return false,
            std::cmp::Ordering::Equal => {}
        }
    }
    false // fully flip-symmetric: both orientations are the same multiset
}

/// STRICT'S PRICE of a spelling: the sum of `terms` (negated where `negated`: `-1 * (terms)`)
/// read by strict's constructors, certificate-free (the bare canon, the reading `complexity`
/// gives strict's answers), priced. Permissive breaks a tie between two equally priced spellings
/// by it ([`flipped_orientation_wins`]). Remembered for the simplify call where one holds the
/// memo open (`RegionScope`): the sums a search visits recur from state to state.
fn strict_price(terms: &[Ex], negated: bool, cx: &Cx) -> u64 {
    let sum = Ex::Add(terms.to_vec());
    let e = if negated {
        Ex::Mul(vec![Ex::Num(Rat::NEG_ONE), sum])
    } else {
        sum
    };
    let key = (e, cx.f64_numbers());
    if let Some(p) =
        STRICT_PRICE_MEMO.with(|m| m.borrow().as_ref().and_then(|m| m.get(&key).copied()))
    {
        return p;
    }
    let p = complexity(&canon(key.0.clone(), &Cx::bare(cx.view)), cx.view);
    STRICT_PRICE_MEMO.with(|m| {
        if let Some(m) = m.borrow_mut().as_mut() {
            m.insert(key, p);
        }
    });
    p
}

/// The signed rational coefficient a term contributes to the orientation comparison
/// (`Const` counts +1 -- sign-free; infinities carry their sign as +/-1).
/// LIKE PARTS BEFORE THE INFINITY, where the absorption of infinities settles
/// ([`Law::InfinityAbsorbs`], `permissive`): a sum holding an infinity beside a part with a
/// variable is one term to the like-term cancellation (`add.cancel`), and so is a product holding
/// such a sum. The cancellation collects such terms of one sum by their factors before the
/// flatten spreads them and before the infinity absorbs anything: `(x0 + inf) - (x0 + inf)` is 0,
/// where the spread terms would meet `inf - inf`, `x1*(x0 + inf) - x1*(x0 + inf)` is 0, and
/// `2*(x0 + inf) - (x0 + inf)` is `x0 + inf`. A sum read with its infinity negative is the
/// negation of the sum with every term negated, so `-(x0 + inf)`, built as `-x0 - inf`, is the
/// part `x0 + inf` with the factor -1. The other items pass unchanged, in their order.
fn collect_infinite_parts(items: Vec<Ex>, cx: &Cx) -> Vec<Ex> {
    let infinite_sum = |t: &Ex| {
        matches!(t, Ex::Add(v) if v.iter().any(|m| matches!(m, Ex::PosInf | Ex::NegInf))
            && v.iter().any(|m| m.has_measure_space(cx.view)))
    };
    // (the factor, the part read with its infinity positive)
    let part = |e: &Ex| -> Option<(Rat, Ex)> {
        let (mut r, sum, mut others) = match e {
            Ex::Add(v) if infinite_sum(e) => (Rat::ONE, v, Vec::new()),
            Ex::Mul(m) => {
                let mut sums = m.iter().filter(|f| infinite_sum(f));
                let (Some(Ex::Add(v)), None) = (sums.next(), sums.next()) else {
                    return None;
                };
                let mut r = Rat::ONE;
                let mut others = Vec::with_capacity(m.len());
                for f in m {
                    match f {
                        Ex::Num(q) => r = r.checked_mul(q)?,
                        f if infinite_sum(f) => {}
                        f => others.push(f.clone()),
                    }
                }
                (r, v, others)
            }
            _ => return None,
        };
        let oriented = if sum.iter().any(|t| matches!(t, Ex::PosInf)) {
            Ex::Add(sum.clone())
        } else {
            r = r.checked_neg()?;
            let flipped: Option<Vec<Ex>> = sum.iter().map(|t| negate_term(t, cx)).collect();
            add(flipped?, cx)
        };
        let key = if others.is_empty() {
            oriented
        } else {
            others.push(oriented);
            mul(others, cx)
        };
        Some((r, key))
    };
    let parts: Vec<Option<(Rat, Ex)>> = items.iter().map(part).collect();
    if parts.iter().flatten().count() < 2 {
        return items;
    }
    let mut taken = vec![false; items.len()];
    let mut out: Vec<Ex> = Vec::with_capacity(items.len());
    for i in 0..items.len() {
        if taken[i] {
            continue;
        }
        taken[i] = true;
        let Some((r, key)) = &parts[i] else {
            out.push(items[i].clone());
            continue;
        };
        let group: Vec<usize> = (i + 1..items.len())
            .filter(|&j| !taken[j] && parts[j].as_ref().is_some_and(|(_, k)| k == key))
            .collect();
        if group.is_empty() || !Law::AddCancel.holds(key, cx) {
            out.push(items[i].clone());
            continue;
        }
        let factor = group.iter().try_fold(r.clone(), |acc, &j| {
            parts[j].as_ref().and_then(|(s, _)| acc.checked_add(s))
        });
        let Some(factor) = factor else {
            out.push(items[i].clone());
            continue;
        };
        for &j in &group {
            taken[j] = true;
        }
        if !factor.is_zero() {
            out.push(mul(vec![Ex::Num(factor), key.clone()], cx));
        }
    }
    out
}

/// THE ABSORPTION OF INFINITIES as a settling step (`add.infinity_absorbs`, where
/// [`Law::InfinityAbsorbs`] settles): a built sum holding an infinity absorbs every term the law's
/// premise admits, as the constructor does where the law is one of its steps. The pass takes it
/// after the constructors' like-term cancellations, so a part with a variable beside an infinity
/// is one term to them: `(x0 + inf) - (x0 + inf)` is 0, and `x0 + inf` alone is `inf`. `None`
/// where the sum absorbs nothing.
pub(crate) fn absorb_infinities(e: &Ex, cx: &Cx) -> Option<Ex> {
    let Ex::Add(v) = e else {
        return None;
    };
    let infinite = |t: &Ex| matches!(t, Ex::PosInf | Ex::NegInf);
    let kept: Vec<Ex> = if v.iter().any(infinite) {
        v.iter()
            .filter(|t| infinite(t) || !Law::InfinityAbsorbs.holds(t, cx))
            .cloned()
            .collect()
    } else if v.iter().any(is_inf_carrying) {
        v.iter()
            .filter(|t| is_inf_carrying(t) || !Law::InfinityAbsorbs.holds(t, cx))
            .cloned()
            .collect()
    } else {
        return None;
    };
    (kept.len() < v.len()).then(|| add(kept, cx))
}

/// Is this Add term an infinity-carrying product? Such a term is never finite: its
/// values are ±inf (infinity times a nonzero finite co-product) or NaN (times zero,
/// or a NaN co-factor). Bare `Ex::PosInf`/`Ex::NegInf` never reach the term list
/// (the flatten scan diverts them to the sign flags), so `Mul` is the only carrier.
pub(super) fn is_inf_carrying(t: &Ex) -> bool {
    matches!(t, Ex::Mul(v) if v.iter().any(|f| matches!(f, Ex::PosInf | Ex::NegInf)))
}

/// Does this Add member absorb a negation WITHOUT keeping a
/// coefficient sign? Carriers: a product with a bare infinity factor (`term_join`
/// folds the sign into the infinity, exactly), a product with an inf-bearing Add
/// factor (`mul()`'s bare-inf arm folds the sign INTO that sum, for any negative
/// coefficient), and -- RECURSIVELY -- a product whose Add factor holds an absorbing
/// member (the sign chases the absorber through `mul()`'s absorbing arm at each level,
/// mirroring the depth `Ex::contains_const` gives the Const side). A sum
/// holding such a member can never park a class sign in a `-1 x Add` wrapper AT ANY
/// factor multiset: the display splits co-factors (divisor-side literals render as
/// `-(S)/den`), so re-parse associates the sign with the sum alone, routes it
/// through `mul()`, and the absorbing member eats it -- the wrapper spelling is
/// never re-parse-stable (the bare-inf conflation, one level up; fuzz row 18943 pinned
/// the non-lone shape). `mul()`'s absorbing sign arm distributes unconditionally on this
/// class, and `primitive_sum` never makes a sign decision for it (the same division
/// of ownership as Const-bearing sums: absorption owns the sign).
pub(super) fn term_absorbs_negation(t: &Ex) -> bool {
    matches!(t, Ex::Mul(v) if v.iter().any(|f| {
        matches!(f, Ex::PosInf | Ex::NegInf)
            || matches!(f, Ex::Add(ts) if ts.iter().any(|m| {
                matches!(m, Ex::PosInf | Ex::NegInf) || term_absorbs_negation(m)
            }))
    }))
}

fn orientation_coeff(t: &Ex, view: &TokenView) -> Coef {
    match t {
        Ex::Const(_) => Coef::ONE,
        Ex::PosInf => Coef::ONE,
        Ex::NegInf => Coef::ONE.negated(),
        Ex::Num(r) => Coef::of(r),
        // Sign-normalized split: an odd-literal term's sign is SEMANTIC and the
        // orientation comparison must see it -- sin(-2) contributes -1, not +1.
        _ => term_split(t.clone(), view).0,
    }
}
