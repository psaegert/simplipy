//! THE SIGN PLACEMENT of a product ([`sign_place`]): the sign orbit and its argmin, the sign trade
//! of odd heads, and the decisions a simplify call makes once.

use super::*;

/// The sites of one class in a product (`sign_place`): the class, and per site its index among the
/// sites and whether it spells the class against the class's own orientation.
type ClassGroup = (Option<Ex>, Vec<(usize, bool)>);

/// Assemble a product from `(coefficient, factor bag)` in its CANONICAL sign
/// placement -- the shared owner behind `mul()`'s final assembly and `term_join`'s
/// negative-coefficient joins, so every site that mints a product builds the SAME
/// spelling the orientation machinery prices (the row-120 law). With no trade site
/// (or a negation overflow, or no configuration that keeps the product's values at its zeros)
/// this is exactly the plain assembly: push the non-unit coefficient, sort, wrap.
pub(super) fn sign_place(coeff: Rat, out: Vec<Ex>, cx: &Cx) -> Ex {
    sign_place_signed(coeff, out, None, cx)
}

/// The zero of sign `negative` in `f64` (`Some`): `+0` is the literal zero and `-0` the written
/// negative zero. `None`: the one zero.
pub(super) fn signed_zero(negative: Option<bool>, cx: &Cx) -> Ex {
    match negative {
        Some(true) => Ex::Leaf(cx.view.intern(NEG_ZERO)),
        _ => Ex::Num(Rat::ZERO),
    }
}

/// [`sign_place`] for a product whose coefficient is a zero: in `f64` (`Some(negative)`) the zero
/// carries the product's sign, `+0` or `-0`, and a flip of a site moves it, unless a bare `Const`
/// is beside it (the refit eats every sign); `None`, the one zero eats it.
pub(super) fn sign_place_zero(negative: Option<bool>, out: Vec<Ex>, cx: &Cx) -> Ex {
    if out.iter().any(|f| matches!(f, Ex::Const(_))) {
        return sign_place(Rat::ZERO, out, cx);
    }
    sign_place_signed(Rat::ZERO, out, negative, cx)
}

/// [`sign_place`], with the zero's sign where the coefficient is a zero that carries one.
fn sign_place_signed(coeff: Rat, out: Vec<Ex>, zero: Option<bool>, cx: &Cx) -> Ex {
    // A placement under the call's own contexts is made once in the call (`CallDecisions`).
    let Some(ctx) = call_decision_key(cx) else {
        return sign_placed(coeff, out, zero, cx);
    };
    let key = (ctx, coeff.clone(), out.clone(), zero);
    if let Some(hit) = CALL_DECISIONS.with(|m| {
        m.borrow()
            .as_ref()
            .and_then(|d| d.placements.get(&key).cloned())
    }) {
        return hit;
    }
    let r = sign_placed(coeff, out, zero, cx);
    CALL_DECISIONS.with(|m| {
        if let Some(d) = m.borrow_mut().as_mut() {
            d.placements.insert(key, r.clone());
        }
    });
    r
}

/// [`sign_place_signed`], placed.
fn sign_placed(mut coeff: Rat, mut out: Vec<Ex>, zero: Option<bool>, cx: &Cx) -> Ex {
    // A sum holding a `Const` takes a sign into its terms (`negate_term`: a Const term absorbs it
    // in the refit, every other term negates; `Law::AbsorbSumSign`), so it carries the product's
    // sign exactly, as a negation-absorbing sum does; only a bare `Const` factor or a zero
    // coefficient eats a sign.
    // Beside such a sum the coefficient and the infinities are positive and the sum holds their
    // signs (the Const sign fold, extended to the infinities).
    let const_sum = if !coeff.is_zero() && !out.iter().any(|f| matches!(f, Ex::Const(_))) {
        out.iter()
            .position(|f| matches!(f, Ex::Add(v) if v.iter().any(Ex::contains_const)))
    } else {
        None
    };
    if let Some(i) = const_sum {
        let negs = out.iter().filter(|f| matches!(f, Ex::NegInf)).count();
        if (negs % 2 == 1) != coeff.is_negative() {
            let Ex::Add(terms) = &out[i] else {
                unreachable!()
            };
            let flipped: Option<Vec<Ex>> = terms.iter().map(|t| negate_term(t, cx)).collect();
            if let Some(flipped) = flipped {
                let re = add(flipped, cx);
                if matches!(re, Ex::Add(_)) {
                    out[i] = re;
                    coeff = coeff.abs();
                    for f in out.iter_mut() {
                        if matches!(f, Ex::NegInf) {
                            *f = Ex::PosInf;
                        }
                    }
                }
            }
        } else if negs > 0 || coeff.is_negative() {
            coeff = coeff.abs();
            for f in out.iter_mut() {
                if matches!(f, Ex::NegInf) {
                    *f = Ex::PosInf;
                }
            }
        }
    }
    let assemble = |factors: Vec<Ex>, c: &Rat| -> Ex {
        let mut v = factors;
        // A unit coefficient is no member (`mul.one`, total).
        if !(c.is_one() && Law::MulOne.total()) {
            v.push(Ex::Num(c.clone()));
        }
        match v.len() {
            0 => Ex::Num(Rat::ONE),
            1 => v.pop().unwrap(),
            _ => {
                v.sort_by(|a, b| mul_factor_cmp(a, b, cx.view));
                Ex::Mul(v)
            }
        }
    };
    // The bag's SIGN CARRIER -- where a sign toggle legally lives (mirroring the
    // bare-inf, Const and negation absorption arms above, which guarantee the ENTRY spelling is
    // carrier-normalized; the owner must return only carrier-normalized spellings
    // too, or the arms and the owner mint two states for one value -- measured live
    // as 9 idem failures at 64k / 50 at 1M when `assemble` parked a raw `-1` beside
    // a bare Const (row 1419) and a bare infinity (row 7320)):
    //   * a bare Const factor: the sign VANISHES into the forall-exists refit -- every
    //     site orientation is value-equal as a family, so orientations are chosen
    //     freely;
    //   * a Const-bearing Add factor: the sign folds into that sum term by term
    //     (`negate_term`; the Const sign fold), the coefficient and the infinities positive;
    //   * a bare +-Inf factor: the infinity carries the sign itself;
    //   * an Add factor with a negation-absorbing member: the sign folds into that
    //     sum member-wise;
    //   * a ZERO coefficient (the kept-zero bag `Mul[0, ..]`): the zero eats every
    //     sign -- `0 * X` and `0 * (-X)` are value-equal in EVERY case (finite -> 0,
    //     +-inf -> nan, nan -> nan), so the whole sign dimension collapses exactly
    //     like a Const-carrier's (fuzz rows 392777/647852:
    //     the kept-zero arm assembled RAW and froze the arrival mirror, so the
    //     direct build and the parse of its own rendering shipped two states);
    //   * otherwise: the rational coefficient.
    {
        // Each site's flip is materialized ONCE (the site test computes it anyway) and
        // reused by every orientation mask below: a flip negates a sum term by term, and
        // each negated term is built through `mul`, which places its own signs the same way,
        // so re-flipping per mask multiplied the work at every nesting level. Each mask's
        // price is the sum of per-factor prices (see `complexity`'s Mul arm), each factor and
        // each flip priced once.
        // A class trades its sign only as a whole: the sites built over one sum (`site_class`)
        // are written in one orientation and flip together, so a flip moves the product's sign
        // by the parity of the class's members. At the zero of the sum every member's base is +0,
        // so the product's value there is the coefficient's infinity -- the limit wherever the
        // pole is one-signed; a single member flipping alone would turn it (`1/(cbrt(u) u)`).
        // Beside a bare `Const`, which absorbs every constant factor of the product and so the
        // orientation sign of every class, every class is a site (`refit_trade_flip`,
        // `Law::AbsorbRefitSign`).
        let refit = out.iter().any(|f| matches!(f, Ex::Const(_)));
        let site_flips: Vec<(usize, Ex)> = out
            .iter()
            .enumerate()
            .filter_map(|(i, f)| {
                let flip = if refit {
                    refit_trade_flip(f, cx)
                } else {
                    sign_trade_flip(f, cx)
                };
                flip.map(|nf| (i, nf))
            })
            .collect();
        // A class is read only where its sum meets another occurrence of itself: two sites over
        // sums of one shape, or a sum inside another factor (`classes_meet`). Elsewhere each site
        // is a class of its own, which nothing pins and no other member checks.
        let meets = classes_meet(&out, &site_flips, cx);
        // Each group: its sites (indices into `site_flips`) and, per site, whether it spells the
        // class against the class's own orientation (`site_class`). A configuration bit picks one
        // orientation for the whole group; the sites spelled against it flip.
        let mut groups: Vec<ClassGroup> = Vec::new();
        for (bit, (pos, _)) in site_flips.iter().enumerate() {
            let class = match site_multiple(&out[*pos], cx) {
                Some(c) => Some(c),
                None if meets => site_class(&out[*pos], cx),
                None => None,
            };
            let joined = class.as_ref().and_then(|(key, flipped)| {
                groups
                    .iter_mut()
                    .find(|(k, _)| k.as_ref() == Some(key))
                    .map(|g| (g, *flipped))
            });
            match joined {
                Some(((_, members), flipped)) => members.push((bit, flipped)),
                None => {
                    let (key, flipped) = class.map_or((None, false), |(k, f)| (Some(k), f));
                    groups.push((key, vec![(bit, flipped)]));
                }
            }
        }
        // Reading A at the zero of each class (`zero_limit`): where the product keeps a value
        // there, a configuration that leaves it off the limit is no candidate. A configuration
        // changes the count of members whose sign at the zero differs from their sign beside it by
        // the sites it flips whose own count changes; a member built over the sum's other
        // orientation inside a product (`sin(-(x1 (x1 - 3)))` for `sin(x1 (3 - x1))`) is counted
        // where it stands, so the product restores the limit an inner orientation moved.
        let toggles_of = |k: &Ex| -> Vec<Option<bool>> {
            site_flips
                .iter()
                .map(|(pos, nf)| {
                    let r = zero_read(&out[*pos], k, cx);
                    if r.parity == Parity::Free {
                        return Some(false);
                    }
                    Some(r.differs? != zero_read(nf, k, cx).differs?)
                })
                .collect()
        };
        let checks: Vec<(bool, Vec<Option<bool>>)> = groups
            .iter()
            .filter_map(|(k, _)| {
                let k = k.as_ref()?;
                let (_, now) = zero_limit(&out, k, cx)?;
                Some((now, toggles_of(k)))
            })
            .collect();
        let factor_price = |f: &Ex| -> u64 {
            match f {
                Ex::Num(r) if r.is_one() => 0,
                Ex::Num(r) if *r == Rat::NEG_ONE => mu_sign(),
                Ex::Num(r) => mu_rat(r),
                _ => complexity(f, cx.view),
            }
        };
        if let (false, Some(nc)) = (site_flips.is_empty(), coeff.checked_neg()) {
            // Priority: Free FIRST -- a bare Const eats EVERY sign (coefficient and
            // bare-infinity signs alike, by the forall-exists refit), so with one present
            // the whole sign dimension collapses and orientations are chosen freely.
            // Checking Inf first split the orbit into two components joined only through
            // the refit (fuzz row 39122: {PosInf, x4-x1} and {PosInf, x1-x4} both stable,
            // one value class). A zero coefficient is Free by the same collapse (see the
            // carrier list above). A Const-bearing sum comes next: it holds the signs of
            // the coefficient and the infinities (normalized above), so every toggle goes
            // into it and the orbit has one component.
            let carrier = if let (Some(negative), true) = (zero, coeff.is_zero()) {
                Carrier::Zero(negative)
            } else if coeff.is_zero() || out.iter().any(|f| matches!(f, Ex::Const(_))) {
                Carrier::Free
            } else if let Some(i) = const_sum {
                Carrier::Absorb(i)
            } else if let Some(i) = out
                .iter()
                .position(|f| matches!(f, Ex::PosInf | Ex::NegInf))
            {
                Carrier::Inf(i)
            } else if let Some(i) = out
                .iter()
                .position(|f| matches!(f, Ex::Add(ts) if ts.iter().any(term_absorbs_negation)))
            {
                Carrier::Absorb(i)
            } else {
                Carrier::Coeff
            };
            let base_price: Vec<u64> = out.iter().map(&factor_price).collect();
            let flip_price: Vec<u64> = site_flips.iter().map(|(_, nf)| factor_price(nf)).collect();
            let orbit = SignOrbit {
                out: &out,
                coeff: &coeff,
                nc: &nc,
                site_flips: &site_flips,
                groups: &groups,
                checks: &checks,
                base_price: &base_price,
                flip_price: &flip_price,
                carrier,
                assemble: &assemble,
            };
            let chosen = sign_place_groups(&orbit, cx);
            // The closed form's price is the enumeration's optimum: checked on every small orbit.
            #[cfg(debug_assertions)]
            if groups.len() <= 6 {
                debug_assert_eq!(
                    chosen.as_ref().map(|c| complexity(c, cx.view)),
                    sign_place_enumerated(&orbit, cx),
                    "the sign orbit's closed form prices off its enumeration"
                );
            }
            if let Some(chosen) = chosen {
                return chosen;
            }
        }
    }
    match zero {
        Some(negative) if coeff.is_zero() => {
            out.push(signed_zero(Some(negative), cx));
            assemble(out, &Rat::ONE)
        }
        _ => assemble(out, &coeff),
    }
}

/// The negation of a sum that carries a product's sign (`sign_place`): a sum holding a `Const`
/// is negated through `add`, which settles a negated member that holds a Const-bearing sum itself
/// (its sign folds into that sum, as `mul()` folds it); any other carrier term by term. `None`
/// where a negation overflows or the sum does not stay one.
fn flip_carrier_sum(f: &Ex, cx: &Cx) -> Option<Ex> {
    let Ex::Add(ts) = f else {
        return None;
    };
    let mut fl = ts
        .iter()
        .map(|x| negate_term(x, cx))
        .collect::<Option<Vec<Ex>>>()?;
    if ts.iter().any(Ex::contains_const) {
        return Some(add(fl, cx)).filter(|s| matches!(s, Ex::Add(_)));
    }
    fl.sort_by(|a, b| add_term_cmp(a, b, cx.view));
    Some(Ex::Add(fl))
}

/// A product's sign carrier (`sign_place`): where a sign toggle lives.
#[derive(Clone, Copy)]
enum Carrier {
    Inf(usize),
    Free,
    Absorb(usize),
    Coeff,
    /// A zero coefficient in `f64`, of the sign given: a toggle turns `+0` into `-0`.
    Zero(bool),
}

/// A product's sign orbit (`sign_place`): its factors and coefficient (and the coefficient's
/// negation), its sign-trade sites with their flips, the classes they form, the zero checks, the
/// prices of every factor and flip, and the sign carrier.
struct SignOrbit<'a> {
    out: &'a [Ex],
    coeff: &'a Rat,
    nc: &'a Rat,
    site_flips: &'a [(usize, Ex)],
    groups: &'a [ClassGroup],
    checks: &'a [(bool, Vec<Option<bool>>)],
    base_price: &'a [u64],
    flip_price: &'a [u64],
    carrier: Carrier,
    assemble: &'a dyn Fn(Vec<Ex>, &Rat) -> Ex,
}

/// One orientation of a group of sites (`SignOrbit::group_options`): the sites it flips, their
/// price, whether it moves the product's sign, and its sites as they then stand (the tie order).
#[derive(Clone)]
struct SiteOption {
    flips: Vec<usize>,
    cost: u64,
    odd: bool,
    key: Vec<Ex>,
    /// The signs of its sites as they then stand (`sign_key`, summed).
    signs: (u32, u32),
}

impl SignOrbit<'_> {
    /// The product the flipped sites give, the carrier taking the sign their count moves, and
    /// whether its coefficient is negative; `None` where the carrier cannot take it.
    fn candidate(&self, flipped: &[bool], cx: &Cx) -> Option<(Ex, bool)> {
        let mut factors = self.out.to_vec();
        let mut toggled = false;
        for (bit, (pos, nf)) in self.site_flips.iter().enumerate() {
            if flipped[bit] {
                factors[*pos] = nf.clone();
                toggled = !toggled;
            }
        }
        if let Carrier::Zero(negative) = self.carrier {
            let negative = negative ^ toggled;
            factors.push(signed_zero(Some(negative), cx));
            return Some(((self.assemble)(factors, &Rat::ONE), negative));
        }
        let c = match self.carrier {
            Carrier::Zero(_) => unreachable!("the zero carrier is assembled above"),
            Carrier::Coeff if toggled => self.nc.clone(),
            Carrier::Coeff => self.coeff.clone(),
            Carrier::Inf(i) => {
                if toggled {
                    factors[i] = match &factors[i] {
                        Ex::PosInf => Ex::NegInf,
                        _ => Ex::PosInf,
                    };
                }
                self.coeff.clone()
            }
            // The refit eats the toggle: every configuration is value-equal as a fitted family.
            // The whole sign dimension goes into the refit: coefficient magnitude only, bare
            // infinities positive.
            Carrier::Free => {
                for f in factors.iter_mut() {
                    if matches!(f, Ex::NegInf) {
                        *f = Ex::PosInf;
                    }
                }
                if self.coeff.is_negative() {
                    self.nc.clone()
                } else {
                    self.coeff.clone()
                }
            }
            Carrier::Absorb(i) => {
                if toggled {
                    factors[i] = flip_carrier_sum(&factors[i], cx)?;
                }
                self.coeff.clone()
            }
        };
        let negative = c.is_negative();
        Some(((self.assemble)(factors, &c), negative))
    }

    /// Each group's orientations: the class written along its key or against it, every site
    /// spelled the other way flipping. An orientation that flips a site onto the base of another
    /// factor is none: the bag would be an unmerged spelling that collection merges, so the
    /// product would not read back as itself (two factors the entry already holds over one base
    /// are how `mul()` emits them, and block nothing).
    fn group_options(&self, cx: &Cx) -> Vec<Vec<SiteOption>> {
        let view = cx.view;
        let mut site_of: Vec<Option<usize>> = vec![None; self.out.len()];
        let mut group_of = vec![0usize; self.site_flips.len()];
        for (bit, (pos, _)) in self.site_flips.iter().enumerate() {
            site_of[*pos] = Some(bit);
        }
        for (g, (_, members)) in self.groups.iter().enumerate() {
            for (bit, _) in members {
                group_of[*bit] = g;
            }
        }
        let state = |bit: usize, flipped: bool| -> &Ex {
            if flipped {
                &self.site_flips[bit].1
            } else {
                &self.out[self.site_flips[bit].0]
            }
        };
        self.groups
            .iter()
            .enumerate()
            .map(|(g, (_, members))| {
                [false, true]
                    .into_iter()
                    .filter_map(|against| {
                        let flips: Vec<usize> = members
                            .iter()
                            .filter(|(_, other)| *other != against)
                            .map(|(bit, _)| *bit)
                            .collect();
                        for &bit in &flips {
                            let (pos, nf) = &self.site_flips[bit];
                            for j in (0..self.out.len()).filter(|j| j != pos) {
                                let hit = match site_of[j] {
                                    Some(b) if group_of[b] == g => {
                                        same_collection_base(nf, state(b, flips.contains(&b)), cx)
                                    }
                                    Some(b) => {
                                        same_collection_base(nf, state(b, false), cx)
                                            || same_collection_base(nf, state(b, true), cx)
                                    }
                                    None => same_collection_base(nf, &self.out[j], cx),
                                };
                                if hit {
                                    return None;
                                }
                            }
                        }
                        let cost = members.iter().fold(0u64, |acc, (bit, _)| {
                            acc.saturating_add(if flips.contains(bit) {
                                self.flip_price[*bit]
                            } else {
                                self.base_price[self.site_flips[*bit].0]
                            })
                        });
                        let mut key: Vec<Ex> = members
                            .iter()
                            .map(|(bit, _)| state(*bit, flips.contains(bit)).clone())
                            .collect();
                        key.sort_by(|a, b| cmp_ex(a, b, view));
                        let signs = key
                            .iter()
                            .map(|f| sign_key(f, view))
                            .fold((0, 0), |a, k| (a.0 + k.0, a.1 + k.1));
                        Some(SiteOption {
                            odd: flips.len() % 2 == 1,
                            flips,
                            cost,
                            key,
                            signs,
                        })
                    })
                    .collect()
            })
            .collect()
    }

    /// Does a configuration (the sites it flips) keep every value the product keeps at a class's
    /// zero (`zero_limit`)? A site whose own count is not read may not flip.
    fn keeps(&self, checks: &[usize], flipped: &[bool]) -> bool {
        checks.iter().all(|&c| {
            let (now, toggles) = &self.checks[c];
            let mut off = *now;
            for (bit, t) in toggles.iter().enumerate() {
                if flipped[bit] {
                    match t {
                        Some(t) => off ^= t,
                        None => return false,
                    }
                }
            }
            !off
        })
    }
}

/// THE SIGN ORBIT'S ARGMIN, for any number of classes. A product's price is the sum of its
/// factors' prices and the carrier's (`complexity`'s Mul arm; pieces of a literal, `is_piece`,
/// price together, which a site that is one reads as its own price), so the cheapest spelling is
/// separable up to the one coupling the classes share, the PARITY of the sites they flip (the
/// sign the carrier takes). Each class flips as a whole and prices as the sum of its members; a
/// zero check (`zero_limit`) constrains the classes whose sites it counts, and the classes one
/// check couples are decided together (a component, its orientations enumerated: in practice a
/// check counts the sites of its own class only). Per component its cheapest orientation, and per
/// parity of the sign the one component whose switch to the other parity costs least: the
/// cheapest product of either sign, priced exactly, the cheaper winning, on an equal price the tie
/// order (`tie_cmp`: the sign key, then the structure). A normal-form choice: the configurations
/// are admitted by the zero checks, never by the price. Ties within a parity resolve by a fixed
/// order that is a function of the orbit, never of the entry spelling: each component's
/// orientation by its sites as they then stand, the switch by the components' smallest site.
/// `None` where no configuration keeps the product's values at its zeros, or a component
/// couples more than `COMPONENT_LIMIT` classes: the caller keeps the entry spelling.
fn sign_place_groups(o: &SignOrbit, cx: &Cx) -> Option<Ex> {
    const COMPONENT_LIMIT: usize = 12;
    let view = cx.view;
    let n = o.site_flips.len();
    let options = o.group_options(cx);
    if options.iter().any(Vec::is_empty) {
        return None;
    }
    let mut group_of = vec![0usize; n];
    for (g, (_, members)) in o.groups.iter().enumerate() {
        for (bit, _) in members {
            group_of[*bit] = g;
        }
    }
    // Components: the groups each check couples (the sites whose flip changes its count, or whose
    // count is not read), joined.
    let mut root: Vec<usize> = (0..o.groups.len()).collect();
    fn find(root: &mut [usize], g: usize) -> usize {
        let mut r = g;
        while root[r] != r {
            r = root[r];
        }
        root[g] = r;
        r
    }
    let mut check_groups: Vec<Vec<usize>> = Vec::with_capacity(o.checks.len());
    for (now, toggles) in o.checks {
        let mut gs: Vec<usize> = toggles
            .iter()
            .enumerate()
            .filter(|(_, t)| **t != Some(false))
            .map(|(bit, _)| group_of[bit])
            .collect();
        gs.sort_unstable();
        gs.dedup();
        if gs.is_empty() && *now {
            return None;
        }
        for w in gs.windows(2) {
            let (a, b) = (find(&mut root, w[0]), find(&mut root, w[1]));
            root[a.max(b)] = a.min(b);
        }
        check_groups.push(gs);
    }
    let mut components: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    for g in 0..o.groups.len() {
        let r = find(&mut root, g);
        match components
            .iter_mut()
            .find(|(gs, _)| find(&mut root, gs[0]) == r)
        {
            Some((gs, _)) => gs.push(g),
            None => components.push((vec![g], Vec::new())),
        }
    }
    for (c, gs) in check_groups.iter().enumerate() {
        if let Some(&g) = gs.first() {
            let r = find(&mut root, g);
            if let Some((_, cs)) = components
                .iter_mut()
                .find(|(groups, _)| find(&mut root, groups[0]) == r)
            {
                cs.push(c);
            }
        }
    }
    // Each component's orientations, cheapest first (then by its sites as they stand), and its
    // rank: its smallest site in either orientation.
    let mut decided: Vec<(Ex, Vec<SiteOption>)> = Vec::with_capacity(components.len());
    for (gs, cs) in &components {
        if gs.len() > COMPONENT_LIMIT {
            return None;
        }
        let mut opts: Vec<SiteOption> = vec![SiteOption {
            flips: Vec::new(),
            cost: 0,
            odd: false,
            key: Vec::new(),
            signs: (0, 0),
        }];
        for &g in gs {
            opts = opts
                .into_iter()
                .flat_map(|a| {
                    options[g].iter().map(move |b| {
                        let mut flips = a.flips.clone();
                        flips.extend(b.flips.iter().copied());
                        let mut key = a.key.clone();
                        key.extend(b.key.iter().cloned());
                        SiteOption {
                            flips,
                            cost: a.cost.saturating_add(b.cost),
                            odd: a.odd ^ b.odd,
                            key,
                            signs: (a.signs.0 + b.signs.0, a.signs.1 + b.signs.1),
                        }
                    })
                })
                .collect();
        }
        if !cs.is_empty() {
            opts.retain(|opt| {
                let mut flipped = vec![false; n];
                for &bit in &opt.flips {
                    flipped[bit] = true;
                }
                o.keeps(cs, &flipped)
            });
        }
        if opts.is_empty() {
            return None;
        }
        opts.sort_by(|a, b| {
            (a.cost, a.signs)
                .cmp(&(b.cost, b.signs))
                .then_with(|| cmp_vec(&a.key, &b.key, view))
        });
        let rank = gs
            .iter()
            .flat_map(|g| o.groups[*g].1.iter())
            .flat_map(|(bit, _)| [&o.out[o.site_flips[*bit].0], &o.site_flips[*bit].1])
            .min_by(|a, b| cmp_ex(a, b, view))
            .cloned()
            .unwrap_or(Ex::Num(Rat::ONE));
        decided.push((rank, opts));
    }
    decided.sort_by(|a, b| cmp_ex(&a.0, &b.0, view));
    // Per parity, the cheapest configuration: every component's cheapest orientation, and where
    // their parity is the other one, the cheapest switch of one component to its cheapest
    // orientation of the other parity (among switches of one cost, the one whose product is
    // structurally smallest).
    let best: Vec<&SiteOption> = decided.iter().map(|(_, opts)| &opts[0]).collect();
    let build = |pick: &[&SiteOption]| {
        let mut flipped = vec![false; n];
        for opt in pick {
            for &bit in &opt.flips {
                flipped[bit] = true;
            }
        }
        o.candidate(&flipped, cx)
    };
    // Where the sign is absorbed (a bare `Const`, which absorbs every constant factor of its
    // product, or a zero coefficient) the parity is no constraint: every class takes its own
    // cheapest orientation, ties by its sites as they then stand, whatever else the product holds.
    if matches!(o.carrier, Carrier::Free) {
        return build(&best).map(|(e, _)| e);
    }
    let parity = best.iter().fold(false, |p, opt| p ^ opt.odd);
    type Delta = (u64, i64, i64);
    let switches: Vec<(Delta, usize, &SiteOption)> = decided
        .iter()
        .enumerate()
        .filter_map(|(i, (_, opts))| {
            let other = opts.iter().find(|opt| opt.odd != opts[0].odd)?;
            let delta = (
                other.cost - opts[0].cost,
                i64::from(other.signs.0) - i64::from(opts[0].signs.0),
                i64::from(other.signs.1) - i64::from(opts[0].signs.1),
            );
            Some((delta, i, other))
        })
        .collect();
    let least = switches.iter().map(|(delta, _, _)| *delta).min();
    let mut chosen: Option<(u64, Ex)> = None;
    for target in [false, true] {
        let built = if target == parity {
            build(&best)
        } else {
            switches
                .iter()
                .filter(|(delta, _, _)| Some(*delta) == least)
                .filter_map(|(_, i, other)| {
                    let mut pick = best.clone();
                    pick[*i] = other;
                    build(&pick)
                })
                .min_by(|a, b| tie_cmp(&a.0, &b.0, view))
        };
        let Some((cand, _)) = built else {
            continue;
        };
        let mu = complexity(&cand, view);
        let better = match &chosen {
            None => true,
            Some((bmu, bex)) => {
                mu < *bmu || (mu == *bmu && tie_cmp(&cand, bex, view) == Ordering::Less)
            }
        };
        if better {
            chosen = Some((mu, cand));
        }
    }
    chosen.map(|(_, e)| e)
}

/// The debug build's cross-check of `sign_place_groups`: the cheapest price over every
/// configuration of whole classes, under the same orientations and checks.
#[cfg(debug_assertions)]
fn sign_place_enumerated(o: &SignOrbit, cx: &Cx) -> Option<u64> {
    let n = o.site_flips.len();
    let options = o.group_options(cx);
    let all: Vec<usize> = (0..o.checks.len()).collect();
    let mut best: Option<u64> = None;
    for code in 0usize..(1usize << o.groups.len()) {
        let mut flipped = vec![false; n];
        let mut allowed = true;
        for (g, (_, members)) in o.groups.iter().enumerate() {
            let against = code & (1 << g) != 0;
            let flips: Vec<usize> = members
                .iter()
                .filter(|(_, other)| *other != against)
                .map(|(bit, _)| *bit)
                .collect();
            allowed &= options[g].iter().any(|opt| opt.flips == flips);
            for bit in flips {
                flipped[bit] = true;
            }
        }
        if !allowed || !o.keeps(&all, &flipped) {
            continue;
        }
        if let Some((cand, _)) = o.candidate(&flipped, cx) {
            let mu = complexity(&cand, cx.view);
            best = Some(best.map_or(mu, |b| b.min(mu)));
        }
    }
    best
}

/// Cheap shape test for [`sign_trade_flip`] -- true iff the factor is an odd
/// carrier of a tradeable mixed-sign sum. Must stay the exact mirror of that
/// function's match (this one detects, that one materializes).
pub(super) fn is_sign_trade_site(f: &Ex, cx: &Cx) -> bool {
    // Delegates to the materializing test: the FILES-BARE gate needs the flipped
    // terms, so a cheap shape-only mirror cannot exist without drifting (the
    // mixed-sign counting attempt regressed 5 corpus rows -- see sign_trade_flip).
    sign_trade_flip(f, cx).is_some()
}

/// Is `f` a sign-trade site -- an odd carrier of a mixed-sign sum -- and if so,
/// the factor with that sum FLIPPED (`None` = not a site, or refused: Const-bearing
/// and negation-absorbing sums have absorption owners, and negation
/// overflow fails safe). The odd carriers, each a TOTAL identity f(-S) = -f(S) on the
/// extended reals: the sum itself; Pow(S, odd n >= 3); rootn(S, odd m >= 3) (the
/// SIGNED total root is an odd bijection); the eight odd functions (`odd_fun`); and
/// Pow(c, odd n >= 3) of either of the last two over a sum. A negative odd power of any of them
/// is one too, its identity failing only at the pole: there the two sides disagree, so the sign
/// of the pole's infinity is no value the answer keeps (reading A). A product flips the sites of
/// one class only together (`sign_place`), so a one-signed pole keeps its infinity.
///
/// MEMOIZED for the dynamic extent of the outermost call: the flip is a pure function of
/// `(f, cx)` and the thread's number domain, and computing it re-enters itself FOUR times per
/// nesting level -- it negates each term through `term_join`, whose site test
/// (`is_sign_trade_site`) and `mul()` assembly (`sign_place`) each ask the flip of the nested
/// sum, and the files-bare gate (`flipped_orientation_wins`) negates the flipped terms again
/// through the same two. Without the memo a sum nested `d` deep cost `4^d` flips (measured:
/// `x1*(x1 + (N + 1)/x1)` nested 9 deep took 7 s with the search off, all of it here).
pub(super) fn sign_trade_flip(f: &Ex, cx: &Cx) -> Option<Ex> {
    trade_flip(f, cx, false)
}

/// The flip of `f` as a factor of a product that holds a bare `Const` (`sign_place`). A
/// `<constant>` absorbs every constant factor of its product, and a class's orientation sign is
/// one, the constant factor -1: beside one, the orientation of EVERY class is free, so the classes
/// whose sign an absorption owner holds elsewhere (a sum holding a `Const`, one with a
/// negation-absorbing member, one with an infinity) are sites too, their sign the constant's.
fn refit_trade_flip(f: &Ex, cx: &Cx) -> Option<Ex> {
    trade_flip(f, cx, true)
}

/// [`sign_trade_flip`] and [`refit_trade_flip`], memoized; `refit`: the product's sign is a bare
/// `Const`'s.
fn trade_flip(f: &Ex, cx: &Cx, refit: bool) -> Option<Ex> {
    if !matches!(f, Ex::Add(_) | Ex::Pow(..) | Ex::Fun(..)) {
        return None;
    }
    let key = (trade_cx_key(cx), f.clone(), refit);
    let lived = call_lived(&key.0);
    if lived {
        if let Some(hit) =
            CALL_DECISIONS.with(|m| m.borrow().as_ref().and_then(|d| d.flips.get(&key).cloned()))
        {
            return hit;
        }
    }
    let outer = FLIP_MEMO.with(|m| {
        let mut m = m.borrow_mut();
        if m.is_none() {
            *m = Some(rustc_hash::FxHashMap::default());
            true
        } else {
            false
        }
    });
    let _scope = FlipScope(outer);
    if let Some(hit) = FLIP_MEMO.with(|m| m.borrow().as_ref().and_then(|t| t.get(&key).cloned())) {
        return hit;
    }
    let r = sign_trade_flip_raw(f, cx, refit);
    if lived {
        CALL_DECISIONS.with(|m| {
            if let Some(d) = m.borrow_mut().as_mut() {
                d.flips.insert(key.clone(), r.clone());
            }
        });
    }
    FLIP_MEMO.with(|m| {
        if let Some(t) = m.borrow_mut().as_mut() {
            t.insert(key, r.clone());
        }
    });
    r
}

/// THE CALL'S DECISIONS: the flips ([`trade_flip`]) and the opposite-sums decisions
/// ([`collect_opposite_sums`]) made under the simplify call's own contexts, remembered for the
/// whole call. Each is a function of its key (the factor or the product as it arrived, and the
/// context: its token view, certificate closures, mode, switches and the thread's number
/// domain), so a decision met again in a later product -- the search rebuilds every candidate
/// through the constructors, round after round, over sums whose orientation does not change --
/// is the one already made. A key is remembered beyond its decision only where every closure it
/// names lives as long as the call (`call_lived`): a context a caller builds around a closure of
/// its own (a sum's held sign, `Cx::sign_held`) is keyed by that closure's address, which a later
/// caller's closure may take, so its decisions keep the extent of the decision that made them
/// ([`FlipScope`]).
pub(crate) struct CallDecisions(Option<CallMemo>);

/// The call's decisions and the identities of its own contexts' closures and token view.
pub(crate) struct CallMemo {
    lived: Vec<usize>,
    flips: FlipMemo,
    pub(super) opposite: rustc_hash::FxHashMap<OppositeKey, Ex>,
    placements: rustc_hash::FxHashMap<(DecisionCxKey, Rat, Vec<Ex>, Option<bool>), Ex>,
    pub(super) orientations: rustc_hash::FxHashMap<(DecisionCxKey, Vec<Ex>, Wrapper, Tie), bool>,
}

/// The context part of a placement's or an orientation's key: the flip's
/// ([`trade_cx_key`]) and the enclosure certificate's identity, which the constructors a
/// decision builds through read too.
type DecisionCxKey = (FlipCxKey, usize);

/// The context key of a decision under one of the call's own contexts (`call_lived`), `None`
/// under any other.
pub(super) fn call_decision_key(cx: &Cx) -> Option<DecisionCxKey> {
    let key = trade_cx_key(cx);
    let encl = cx.cert_encl.map_or(0, |f| {
        f as *const dyn Fn(&Ex) -> Option<(f64, f64)> as *const () as usize
    });
    let lived = call_lived(&key)
        && (encl == 0
            || CALL_DECISIONS
                .with(|m| m.borrow().as_ref().is_some_and(|d| d.lived.contains(&encl))));
    lived.then_some((key, encl))
}

thread_local! {
    pub(super) static CALL_DECISIONS: std::cell::RefCell<Option<CallMemo>> =
        const { std::cell::RefCell::new(None) };
}

impl CallDecisions {
    /// Opens the call's memo for the contexts of `cx` (its token view and certificate closures),
    /// restoring the caller's when it drops.
    pub(crate) fn open(cx: &Cx) -> Self {
        let (view, certs, _) = flip_cx_key(cx);
        let mut lived: Vec<usize> = certs.iter().copied().filter(|&c| c != 0).collect();
        lived.push(view);
        if let Some(f) = cx.cert_encl {
            lived.push(f as *const dyn Fn(&Ex) -> Option<(f64, f64)> as *const () as usize);
        }
        CallDecisions(CALL_DECISIONS.with(|m| {
            m.borrow_mut().replace(CallMemo {
                lived,
                flips: FlipMemo::default(),
                opposite: rustc_hash::FxHashMap::default(),
                placements: rustc_hash::FxHashMap::default(),
                orientations: rustc_hash::FxHashMap::default(),
            })
        }))
    }
}

impl Drop for CallDecisions {
    fn drop(&mut self) {
        let outer = self.0.take();
        CALL_DECISIONS.with(|m| *m.borrow_mut() = outer);
    }
}

/// Does every closure the context key names (and its token view) live as long as the call? Not
/// while a fire log records the folds (`rules::FireLog`): a decision taken from the memo would not
/// record the folds of its constructions again.
pub(super) fn call_lived(key: &FlipCxKey) -> bool {
    if crate::ac::rules::folds_logged() {
        return false;
    }
    CALL_DECISIONS.with(|m| {
        m.borrow().as_ref().is_some_and(|d| {
            d.lived.contains(&key.0) && key.1.iter().all(|&c| c == 0 || d.lived.contains(&c))
        })
    })
}

/// The memo key's context part: everything `sign_trade_flip` reads besides the factor -- the
/// token view, the four certificate closures (by identity), the mode, the two fold switches and
/// the thread's number domain. All of them outlive the outermost call, so no identity is reused
/// within the memo's extent.
pub(super) type FlipCxKey = (usize, [usize; 6], u8);

type FlipMemo = rustc_hash::FxHashMap<(FlipCxKey, Ex, bool), Option<Ex>>;

/// The opposite-sums decision of a product (`collect_opposite_sums`), keyed like a flip: by the
/// context, the sign host and the factors as they arrived. It lives exactly as long as the flip
/// memo, so a parse or a re-walk decides each distinct product once however often it occurs.
type OppositeKey = (FlipCxKey, HostKey, Vec<Ex>);

thread_local! {
    pub(super) static FLIP_MEMO: std::cell::RefCell<Option<FlipMemo>> = const { std::cell::RefCell::new(None) };
    pub(super) static OPPOSITE_MEMO: std::cell::RefCell<rustc_hash::FxHashMap<OppositeKey, Ex>> =
        std::cell::RefCell::new(rustc_hash::FxHashMap::default());
}

/// Drops the memos when the outermost `sign_trade_flip`, opposite-sums decision, parse or
/// re-walk returns (or unwinds).
pub(crate) struct FlipScope(bool);

impl FlipScope {
    /// Holds the memos open until the guard drops (unless an enclosing scope already does), so
    /// the flips of one decision are computed once however many products it prices, and a parse
    /// or re-walk decides each distinct product once.
    pub(crate) fn open() -> Self {
        let outer = FLIP_MEMO.with(|m| {
            let mut m = m.borrow_mut();
            if m.is_none() {
                *m = Some(rustc_hash::FxHashMap::default());
                true
            } else {
                false
            }
        });
        FlipScope(outer)
    }
}

impl Drop for FlipScope {
    fn drop(&mut self) {
        if self.0 {
            FLIP_MEMO.with(|m| *m.borrow_mut() = None);
            OPPOSITE_MEMO.with(|m| m.borrow_mut().clear());
        }
    }
}

pub(super) fn flip_cx_key(cx: &Cx) -> FlipCxKey {
    fn id(c: Option<&dyn Fn(&Ex) -> bool>) -> usize {
        c.map_or(0, |f| f as *const dyn Fn(&Ex) -> bool as *const () as usize)
    }
    let mode = match cx.mode {
        RuleMode::Default => 0u8,
        RuleMode::Permissive => 2,
    };
    let bits = mode
        | (u8::from(cx.fold_f64) << 2)
        | (u8::from(crate::ac::rat::f64_numbers()) << 4)
        | (u8::from(cx.flat_unread) << 6);
    (
        cx.view as *const TokenView as usize,
        [
            id(cx.cert_fin),
            id(cx.cert_finnz),
            id(cx.cert_nzae),
            id(cx.cert_nce),
            id(cx.cert_region),
            id(cx.sign_held),
        ],
        bits,
    )
}

/// [`flip_cx_key`] with whether the context reads a sum's zero (`Cx::zero_sign_unread`): a sign
/// trade and a product's opposite sums depend on it (`sum_holds_its_sign`), the region test does
/// not.
pub(super) fn trade_cx_key(cx: &Cx) -> FlipCxKey {
    let (view, certs, bits) = flip_cx_key(cx);
    (view, certs, bits | (u8::from(cx.zero_sign_unread) << 5))
}

fn sign_trade_flip_raw(f: &Ex, cx: &Cx, refit: bool) -> Option<Ex> {
    let flip_sum = |ts: &[Ex]| -> Option<Vec<Ex>> {
        // Exclusions = the absorption owners' classes: Const-bearing,
        // absorbing-member, and BARE-infinity terms (the inf carries the
        // sign itself -- `term_absorbs_negation` only matches inf-carrying
        // PRODUCTS, so the bare member needs its own test; missing it let an
        // odd-fun trade flip an inf-bearing sum and inflate mu: fuzz row 131227,
        // the one violation in 200k). Beside a bare `Const` (`refit`) the
        // constant owns every class's sign, theirs too: such a sum negates through
        // `add`, which files it bare (`primitive_sum` never wraps one).
        if ts.iter().any(Ex::contains_const)
            || ts.iter().any(term_absorbs_negation)
            || ts.iter().any(|x| matches!(x, Ex::PosInf | Ex::NegInf))
        {
            if !refit {
                return None;
            }
            let negated = ts
                .iter()
                .map(|t| negate_term(t, cx))
                .collect::<Option<Vec<Ex>>>()?;
            return match add(negated, cx) {
                Ex::Add(v) => Some(v),
                _ => None,
            };
        }
        let mut out = ts
            .iter()
            .map(|t| negate_term(t, cx))
            .collect::<Option<Vec<Ex>>>()?;
        // FILES-BARE gate: the flipped multiset must be a spelling the sum's own orientation
        // (`flipped_orientation_wins`) files bare. Where it would wrap it (`-a - b` rests as
        // `-1 x (a + b)`) the site is none: trading into it would mint the wrapper again, and a
        // power over it would freeze a one-signed sum its constructors do not file. Counting signs
        // is not the same test: an odd literal's sign sits inside a function's argument, where no
        // coefficient shows it.
        if flipped_orientation_wins(&out, Wrapper::Flipped, Tie::Keep, cx) {
            return None;
        }
        // A sign enters a sum only where it holds none (`Law::ClassSignTrade`, as `add.content` in
        // `primitive_sum`): float64's sum is `+0` wherever it cancels, so `-(a - b)` is `-0` there
        // and `b - a` is `+0`, a sign the expression around it may read (`2*(-a - b)` is not
        // `-2*(a + b)` where `a + b` cancels on a region). Asked last.
        if !Law::ClassSignTrade.holds(&Ex::Add(ts.to_vec()), cx) {
            return None;
        }
        out.sort_by(|a, b| add_term_cmp(a, b, cx.view));
        Some(out)
    };
    let odd_int_ge3 = |e: &Ex| -> bool {
        matches!(e, Ex::Num(r) if r.is_odd_integer() && r.cmp_int(3) != Ordering::Less)
    };
    // A negative odd power's pole is two-signed, so its orientation is not pinned there
    // (`-1/(x1 - 3)` is `1/(3 - x1)`).
    let odd_power = |e: &Ex| -> bool {
        odd_int_ge3(e) || matches!(e, Ex::Num(r) if r.is_odd_integer() && r.is_negative())
    };
    match f {
        Ex::Add(ts) => flip_sum(ts).map(Ex::Add),
        Ex::Pow(b, e) if odd_power(e) => {
            let flipped = match &**b {
                Ex::Add(ts) => flip_sum(ts).map(|s| Ex::Pow(Box::new(Ex::Add(s)), e.clone())),
                // An odd power of an odd head of a negatable argument (`class_base`) is one too:
                // `f(-u)^n = -f(u)^n`, for a negative `n` as for a positive one. A constant
                // argument keeps its sign on its literal under a power (`1/tanh(-2)`), as the
                // sign key files it.
                Ex::Fun(op, args)
                    if odd_head(cx.view, *op, args)
                        .map_or(args.first(), |(p, _)| args.get(p))
                        .is_some_and(|a| {
                            a.has_measure_space(cx.view) || a.contains_wildcard(cx.view)
                        }) =>
                {
                    trade_flip(b, cx, refit).map(|nb| Ex::Pow(Box::new(nb), e.clone()))
                }
                _ => None,
            }?;
            // A negative power's pole pins the orientation where the base's zero set may hold a
            // region: there the inverse is infinite on the region, a value the answer keeps. Asked
            // last, of a base whose flip exists.
            (cx.permits(Permissive::Agreement)
                || !matches!(&**e, Ex::Num(r) if r.is_negative())
                || zero_set_null(b, cx))
            .then_some(flipped)
        }
        Ex::Fun(op, args)
            if args.len() == 2 && cx.view.tok_is(*op, "rootn") && odd_int_ge3(&args[1]) =>
        {
            let Ex::Add(ts) = &args[0] else {
                return literal_multiple_flip(f, cx);
            };
            flip_sum(ts).map(|s| Ex::Fun(*op, vec![Ex::Add(s), args[1].clone()]))
        }
        Ex::Fun(op, args)
            if (args.len() == 1 && odd_fun(cx.view, *op))
                || (args.len() == 2
                    && cx.view.tok_is(*op, "besselj")
                    && odd_head(cx.view, *op, args).is_some()) =>
        {
            let pos = args.len() - 1;
            let with = |a: Ex| {
                let mut v = args.clone();
                v[pos] = a;
                v
            };
            match &args[pos] {
                Ex::Add(ts) => flip_sum(ts).map(|s| Ex::Fun(*op, with(Ex::Add(s)))),
                // An odd function of a LITERAL is a trade site too -- f(-r) = -f(r)
                // exactly, so the sign relocates between the coefficient slot and the
                // literal, and the orbit argmin (mu, then the NON-NEGATIVE-coefficient tie,
                // then cmp_ex) lands every route on the ruled direction: the literal
                // owns the sign (`-5 * sin(2)` files as `5 * sin(-2)`; `-1 * sin(2)` as
                // `sin(-2)`, the Mul node dissolving). This is the hoist of `term_join`'s
                // former private fusion arm into the shared owner: `mul()`'s direct
                // assembly and the collector's rebuild now price the SAME orbit. Rebuilt
                // through `fun()`, NEVER raw, so a boundary ground keeps folding
                // (`atanh(-1)` must reach `-inf`); a rebuild that FOLDS to a non-Fun is
                // refused as a site (fail-safe: the orbit stays type-stable, and such
                // grounds fold on their own paths).
                //
                // In `f64` an odd function of the literal ZERO is no site: `-sin(0)` is -0 in
                // float64 and `sin(-0)` would need the written negative zero, so the sign stays on
                // the coefficient, where the ground fold of a ground product writes float64's -0
                // and the zero of a product with a variable is the one zero (`-x1*sin(0)` is
                // `0`). Read as the zero again, the flip would drop the sign (`-sin(0)` as
                // `sin(0)`, a +0). `permissive` reads one zero.
                Ex::Num(r) if r.is_zero() && cx.f64_semantics() && cx.f64_numbers() => None,
                Ex::Num(r) => {
                    let flipped = fun(*op, with(Ex::Num(r.checked_neg()?)), cx);
                    matches!(flipped, Ex::Fun(..)).then_some(flipped)
                }
                _ => literal_multiple_flip(f, cx),
            }
        }
        Ex::Fun(..) => literal_multiple_flip(f, cx),
        _ => None,
    }
}

/// The other spelling of an odd function or odd root of a literal multiple
/// (`odd_of_literal_multiple`): the argument negated through `mul()`, rebuilt through `fun()`;
/// `None` where the rebuild leaves the shape (a fold, or a negation the argument's own sites took
/// instead of the literal).
fn literal_multiple_flip(f: &Ex, cx: &Cx) -> Option<Ex> {
    let c = odd_of_literal_multiple(f, cx.view)?;
    let Ex::Fun(op, args) = f else { return None };
    let (pos, _) = odd_head(cx.view, *op, args)?;
    let mut items = vec![Ex::Num(c.checked_neg()?)];
    match &args[pos] {
        Ex::Num(_) => {}
        Ex::Mul(v) if matches!(v.first(), Some(Ex::Num(_))) => items.extend(v[1..].iter().cloned()),
        u => items.push(u.clone()),
    }
    let mut flipped_args = args.clone();
    flipped_args[pos] = mul(items, cx);
    let flipped = fun(*op, flipped_args, cx);
    matches!(
        class_base(&flipped, cx.view),
        Some(ClassBase { head: Some(_), .. })
    )
    .then_some(flipped)
}
