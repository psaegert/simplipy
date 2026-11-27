//! The canonicalization context [`Cx`]: the mode, the number domain, the certificates, and the list
//! of permissive's own simplifications ([`Permissive`]).

use super::*;

/// The enclosure certificate's type ([`Cx::cert_encl`]): a ground composite's closed float64 box.
pub type EnclosureCert<'a> = &'a dyn Fn(&Ex) -> Option<(f64, f64)>;

/// The parse's ground evaluator (`Cx::ground_first`): a fully numeric part's token span to the
/// atom it is, or `None` to build it as written.
pub type GroundFirst<'a> = &'a dyn Fn(&[Tok]) -> Option<Ex>;

/// The canonicalization context: the token view (leaf classification, string order) plus the two
/// OPTIONAL soundness certificates. With a certificate absent the corresponding gated collection
/// simply does not happen -- fail-closed, exactly like the old matcher's certifier plumbing.
pub struct Cx<'a> {
    pub view: &'a TokenView<'a>,
    /// The `!`-certificate: is this expression defined and finite almost everywhere? Gates
    /// additive SIGN-CANCELLING collection (`2t - t -> t` moves a `t = +-inf` point from NaN to
    /// `+-inf`, sound only when that set is null).
    pub cert_fin: Option<&'a dyn Fn(&Ex) -> bool>,
    /// The `$`-certificate: defined, finite AND nonzero a.e.? Gates multiplicative
    /// exponent-CANCELLING collection (`t^2 * t^-1 -> t` fills the NaN at `t = 0`).
    pub cert_finnz: Option<&'a dyn Fn(&Ex) -> bool>,
    /// The zero-set licence: NONZERO almost everywhere? Consumed by the odd-negative
    /// power distribution (see `pow`), whose identity breaks ONLY on fat zero sets --
    /// finiteness and NaN domains are irrelevant there (both spellings agree at every
    /// infinite/undefined configuration), so this is deliberately WEAKER than the
    /// `$`-certificate (`interval::zero_set_null` alone, not `finite_nonzero_ae`).
    pub cert_nzae: Option<&'a dyn Fn(&Ex) -> bool>,
    /// The nonconstant-entire certificate (`interval::nonconstant_entire`): licenses the
    /// SYMBOLIC-exponent occurrence merge (see `mul`'s like-base collection) on
    /// sign-indefinite bases by proving the exponent's level sets are null.
    pub cert_nce: Option<&'a dyn Fn(&Ex) -> bool>,
    /// The REGION certificate: the zero-set certificate, consulted only where a sum's zero sign is
    /// decided (`keeps_zero_sign`). A parse holds no other certificate, yet where it inverts a sum
    /// whose zero set may hold a region it distributes the inverse (`inverts_a_signed_zero`), a
    /// decision its canon cannot undo, so in `f64` it reads that region as the canon does while
    /// every licence of the parse stays certificate-free. `None`: reading A's test
    /// (`zero_set_null`).
    pub cert_region: Option<&'a dyn Fn(&Ex) -> bool>,
    /// The ENCLOSURE certificate: a closed float64 box `[lo, hi]` holding the value of a ground
    /// composite that is certainly a finite real (the interval layer's value set, sound for the
    /// real function and for the deployed kernel), `None` for anything else. Read outside `f64`
    /// alone, where a lattice bag orders its ground members by it (`max(sin 1, 1/2) = sin 1`).
    pub cert_encl: Option<EnclosureCert<'a>>,
    /// WHICH SOUNDNESS this construction preserves: the mode, from which every reading below
    /// is derived (`f64_semantics`, `permits`, `ieee_zeros`, `table_values`), so the mode stays
    /// the single representation.
    pub mode: RuleMode,
    /// Whether the walk folds a ground node to the float the deployed evaluator computes for
    /// it (`Engine::ground_fold`), in `f64`. DERIVED from the mode by `Cx::folds_for`; the
    /// miner sets it explicitly (it canonicalises unfolded). No constructor folds a
    /// transcendental: folding is bottom-up there, so `exp(19.1...)` would become a 9-digit
    /// literal before the root over it is read, and `tanh 30` would become `1` before `atanh`
    /// sees it. The fold is a step of the pass, tried after the rules at each node and kept
    /// only where it descends; `permissive` always folds so (its lossy reading).
    pub fold_f64: bool,
    /// GROUND FIRST, at the parse: the ground evaluator's verdict on a fully numeric part as
    /// written (its token span), before any constructor builds it. `Some(atom)` where the part's
    /// value is only that atom (NaN or an infinity: a part the reals give no value, or one whose
    /// float64 value is the only one admitted), which the parse takes in place of the part, so no
    /// law ever reaches it; `None` keeps the part, which is then built as written: exact
    /// arithmetic between its numbers keeps its value in the reals, and the fold reads the built
    /// node's value. Absent (no evaluator): the parse builds every part (conversions, patterns,
    /// the measure's canon).
    pub ground_first: Option<GroundFirst<'a>>,
    /// The construction stands where nothing reads the sign of its zero (`zero_read_below`): a
    /// sum's sign moves there whatever its zero set (no sum holds its sign,
    /// `sum_holds_its_sign`). Set by [`release_unread`] alone, for the nodes it rebuilds.
    pub zero_sign_unread: bool,
    /// The construction stands where nothing reads the flat point of `log o exp` or `acosh o cosh`
    /// (`flat_read_below`): float64's `log(exp(t))` is `+0` wherever 0 < |t| < 1.1e-16 and
    /// `acosh(cosh(t))` is 0 wherever |t| < 1.5e-8, where `t` is not, so in `f64` the pair
    /// collapses only where nothing around reads that zero (`fun.log_exp`, `fun.acosh_cosh`); and
    /// the log of an inverse, which collapses at the root's walk in both modes, wherever it stands
    /// (`fun.log_inverse`).
    /// Set by [`release_flat`] alone, for the nodes it rebuilds.
    pub flat_unread: bool,
    /// A PRINTER'S held sums: the sums of the state it spells that hold their sign
    /// ([`held_sums`]), whose negations it writes as the negations they are. `None` for a
    /// construction, which reads a sum's form (`sum_holds_its_sign`).
    pub sign_held: Option<&'a dyn Fn(&Ex) -> bool>,
    /// THE CONFINED BASES (`confine.sibling`): the bases `t` this construction stands confined to
    /// `t >= 0` by, an enclosing node of the state being NaN wherever `t < 0` whatever this
    /// construction computes ([`release_confined`]), so the laws of powers read `t >= 0` of each
    /// (`Cx::confined_nonneg`). Set by [`release_confined`] alone, for the nodes it rebuilds;
    /// empty everywhere else.
    pub confined: &'a [Ex],
}

/// PERMISSIVE'S OWN SIMPLIFICATIONS (the specification's 4.9), the one list, named here and
/// applied in `permissive` only. Permissive makes an expression prettier at the cost of some
/// equivalence, the way one cancels by hand: naively, keeping the spirit of the expression. Every
/// place the engine does something in `permissive` that the sound modes refuse names its entry
/// (`Cx::permits`), and nothing else differs. `abs`, `max`, `floor`, `min` and `sign`-like
/// constructions are no entry: their shape is the point of the expression, and permissive leaves
/// them as the sound modes do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Permissive {
    /// An operation cancels its own inverse, in both orders, ignoring domain and branch
    /// ([`INVERSE_PAIRS`]; `sqrt`/`pow 2` and `rootn n`/`pow n` are powers, which compose,
    /// [`Permissive::PowersCompose`]): `asin(sin(x0))`, `exp(log(x0))` and `acosh(cosh(x0))` are
    /// `x0`.
    InversePairs,
    /// Powers compose naively, ignoring the sign: `(x^a)^b` is `x^(a*b)`, so `sqrt(x0^2)` and
    /// `(x0^3)^(1/3)` are `x0` and `(x0^2)^(1/4)` is `x0^(1/2)` (an even root of a power or a power
    /// of a root alike).
    PowersCompose,
    /// Like terms cancel without a definedness check: `log(x0) - log(x0)` is 0 and
    /// `asin(x0)/asin(x0)` is 1. A part with a variable beside an infinity is one term to them,
    /// and they run before the infinity absorbs it ([`Law::InfinityAbsorbs`]):
    /// `(x0 + inf)/(x0 + inf)` is 1 and `(x0 + inf) - (x0 + inf)` is 0. A fully numeric
    /// `inf/inf` has no value: NaN.
    LikeTerms,
    /// A zero factor absorbs the product without a definedness check (`Law::MulZero`):
    /// `0*log(x0)`, `0*sqrt(x0)`, `0*x0^x0`, `0*asin(x0)`, `0/x0^(1/2)`, `0*acosh(x0)` and
    /// `0*(1/x0)` are 0, as `0*x0` and `0*sin(x0)` are in every mode. Wherever the product has a
    /// value it is 0, so the step never changes a value; it gives one where the factor has none
    /// or is infinite, the kind of step like-term cancellation is. A fully numeric factor is read
    /// for its value first: `0*tan(asin(1))` is NaN.
    ZeroAbsorbs,
    /// A constant collapses without evidence: `(-1)^C` is `C` (`interval::collapse_licence`).
    ConstantCollapse,
    /// A literal becomes its float where the float is cheaper: the quotient of two 16-digit
    /// decimals is `0.46867105279529636` (`land`, and the engine's ground fold).
    LiteralToFloat,
    /// Every slot matches any subexpression, without its certificate: `$0/$0 -> 1` also matches
    /// `inf/inf` (the matcher's `wildcard_all`).
    AnySlot,
    /// The rules that agree with the reals' values up to float resolution (8 units in the last
    /// place) are served (the permissive rule set), judged at the `_` sort this mode's matcher
    /// binds (`AnySlot`), so with no isolated point excused: `asin(1e-8) -> 1e-8`, off by
    /// 1.667e-25.
    FloatResolution,
    /// No simplification of permissive's own: a step that HOLDS in permissive by its criterion
    /// -- the input's value is kept wherever the input has one, an infinity included -- and
    /// which the sound modes refuse because it gives a value where the input has
    /// none (`x0^(1/2) * x0^(1/2)` is `x0`) or because of a zero's sign or a pole's limit.
    Agreement,
}

impl Permissive {
    /// The whole list, in the specification's order.
    pub const ALL: [Permissive; 9] = [
        Permissive::InversePairs,
        Permissive::PowersCompose,
        Permissive::LikeTerms,
        Permissive::ZeroAbsorbs,
        Permissive::ConstantCollapse,
        Permissive::LiteralToFloat,
        Permissive::AnySlot,
        Permissive::FloatResolution,
        Permissive::Agreement,
    ];

    /// Does the entry apply in `mode`? In `permissive` only: the one place that says so.
    #[inline]
    pub fn applies_in(self, mode: RuleMode) -> bool {
        matches!(mode, RuleMode::Permissive)
    }

    /// The entry's name, as the documentation and the Python side spell it.
    pub fn name(self) -> &'static str {
        match self {
            Permissive::InversePairs => "inverse pairs cancel",
            Permissive::PowersCompose => "powers compose",
            Permissive::LikeTerms => "like terms cancel",
            Permissive::ZeroAbsorbs => "a zero factor absorbs the product",
            Permissive::ConstantCollapse => "constant collapse without evidence",
            Permissive::LiteralToFloat => "literal to float",
            Permissive::AnySlot => "any slot matches",
            Permissive::FloatResolution => "float-resolution rules served",
            Permissive::Agreement => "holds by agreement",
        }
    }
}

/// The inverse pairs of [`Permissive::InversePairs`] that are functions (the powers compose,
/// [`Permissive::PowersCompose`]).
pub const INVERSE_PAIRS: [(&str, &str); 7] = [
    ("exp", "log"),
    ("sin", "asin"),
    ("cos", "acos"),
    ("tan", "atan"),
    ("sinh", "asinh"),
    ("cosh", "acosh"),
    ("tanh", "atanh"),
];

/// Does `outer(inner(t))` cancel to `t` by [`Permissive::InversePairs`], in either order?
pub fn cancels_its_inverse(outer: &str, inner: &str) -> bool {
    INVERSE_PAIRS
        .iter()
        .any(|&(f, g)| (outer == f && inner == g) || (outer == g && inner == f))
}

impl<'a> Cx<'a> {
    /// May this construction take the step [`Permissive`] names? In `permissive` only.
    #[inline]
    pub fn permits(&self, entry: Permissive) -> bool {
        entry.applies_in(self.mode)
    }

    /// May this construction take the step [`Permissive`] names on `operand`? A simplification
    /// of permissive's own acts on a part with a variable or a constant (or a pattern's
    /// wildcard); a fully numeric operand is read for its value first, and no simplification
    /// changes that value: `asin(sin(x0))` is `x0`, `asin(sin(4))` keeps pi - 4.
    #[inline]
    pub fn permits_on(&self, entry: Permissive, operand: &Ex) -> bool {
        self.permits(entry)
            && (operand.has_measure_space(self.view) || operand.contains_wildcard(self.view))
    }
}

impl<'a> Cx<'a> {
    /// THE DERIVATION of `fold_f64` from a mode: `f64` folds a ground node to its float in the
    /// pass, and `permissive` folds in its pass by its lossy reading.
    #[inline]
    pub fn folds_for(mode: RuleMode) -> bool {
        matches!(mode, RuleMode::Strict)
    }

    /// The F64 reading: is this construction answerable only in what the deployed f64
    /// evaluator computes? True for `f64` alone. `permissive` reasons over the reals, where
    /// saturation does not exist -- so a guard that exists BECAUSE of f64 saturation must ask
    /// this and not `permits`.
    #[inline]
    pub fn f64_semantics(&self) -> bool {
        matches!(self.mode, RuleMode::Strict)
    }

    /// The number domain this construction runs in, the thread's (`rat::number_domain`): f64 in
    /// every mode -- `permissive` is evaluated in float64 too -- and exact whenever the enclosing
    /// run is (an exact display or reading).
    #[inline]
    pub fn f64_numbers(&self) -> bool {
        crate::ac::rat::f64_numbers()
    }

    /// The SIGNED-ZERO reading: does this construction answer in float64's signed zeros (IEEE
    /// 754)? `f64` is evaluated in float64, where an exact zero carries the sign of what formed
    /// it (`x*0` is -0 wherever x < 0) and its inverse is the infinity of that sign.
    /// `permissive`, whose values are the reals', has one zero, and `1/0` has no value there.
    #[inline]
    pub fn ieee_zeros(&self) -> bool {
        matches!(self.mode, RuleMode::Strict)
    }

    /// Are this mode's values THE TABLE's real values, so that an operation on literals (a
    /// rational, an infinity, NaN) is the table's exact value (`table_literal_fold`)?
    /// `permissive`, whose values are the reals': `1/0` has no value there.
    #[inline]
    pub fn table_values(&self) -> bool {
        !matches!(self.mode, RuleMode::Strict)
    }

    /// In f64 mode, a numeral leaf the evaluator reads as +-inf or (underflowing) as 0:
    /// literals beyond float64's range, which stay leaves there.
    fn f64_reads_degenerate(&self, t: Tok, zero_too: bool) -> bool {
        self.f64_numbers()
            && self.view.with_str(t, |s| {
                crate::utils::looks_numeric(s)
                    && crate::numeric::leaf_value(s)
                        .is_some_and(|v| v.is_infinite() || (zero_too && v == 0.0))
            })
    }

    /// A certificate-free context (conversions, pattern handling, tests).
    pub fn bare(view: &'a TokenView<'a>) -> Self {
        Cx {
            view,
            cert_fin: None,
            cert_finnz: None,
            cert_nzae: None,
            cert_nce: None,
            cert_region: None,
            cert_encl: None,
            mode: RuleMode::Strict,
            fold_f64: false,
            ground_first: None,
            zero_sign_unread: false,
            flat_unread: false,
            sign_held: None,
            confined: &[],
        }
    }

    /// Is `t` one of the bases this construction stands confined to `t >= 0` by
    /// (`Cx::confined`, `confine.sibling`)? The laws of powers whose two sides differ only where
    /// their base is negative read it beside `certainly_nonneg`: `mul.collect`,
    /// `mul.collect_symbolic`, `pow.compose` over a fractional power, `pow.root_inverse` and
    /// `pow.odd_root_compose` in a product.
    #[inline]
    pub(crate) fn confined_nonneg(&self, t: &Ex) -> bool {
        self.confined.iter().any(|b| b == t)
    }

    /// Symbolic-exponent merge licence (see `mul`'s like-base collection): may n
    /// occurrences of `base^y` merge into `base^(n*y)`? The disagreement set is
    /// {base < 0} x {y in (1/n)Z \ Z} -- NaN on the unmerged side, defined on the
    /// merged one wherever n*y lands on an integer that y misses (the branch-cut
    /// question lives in the EXPONENT's level sets, not in the integer occurrence
    /// counts). Sound when the base never goes negative, or when y's level sets are
    /// certifiably NULL (nonconstant entire analyticity, identity theorem): a fat
    /// exponent pinned to 1/2 by an abs-construction, or an identically-1/2 unfolded
    /// analytic identity, is refused. Fail-closed. (The sibling of the
    /// pow-distribution bug, with the fat set in the exponent.)
    pub(super) fn sym_merge_licensed(&self, base: &Ex, y: &Ex) -> bool {
        Law::MulCollectSymbolic.holds_powers(Powers::Symbolic { base, y }, self)
    }

    /// The nonconstant-entire certificate (`cert_nce`): are the level sets of `y` null? Fail
    /// closed without the certificate.
    pub(crate) fn nonconstant_entire(&self, y: &Ex) -> bool {
        self.cert_nce.is_some_and(|f| f(y))
    }

    /// Is `e` CERTAINLY defined-and-finite (a.e.) everywhere, syntactically? Variables range
    /// over finite reals by domain; `Pi`/`E`/`Num` are finite literals; `Const` is a fitted REAL
    /// (finite). The certainty is CLOSED under finite-preserving structure: sums and products of
    /// finite reals are finite; non-negative integer powers preserve finiteness; the
    /// everywhere-finite unary functions preserve it. Anything else (wildcards, `log`, negative
    /// or fractional powers, ...) is never certain here -- it goes through `cert_fin`, the real
    /// interval machinery. A conservative subset by construction: syntactic certainty may say
    /// "no" where the certificate would say "yes", never the reverse.
    pub(crate) fn certainly_finite(&self, e: &Ex) -> bool {
        match e {
            Ex::Num(_) | Ex::Pi | Ex::E | Ex::Const(_) => true,
            // A numeral leaf beyond float64's range denotes a finite real, but in f64
            // mode the deployed evaluator reads it as inf: there it
            // is not certainly finite (`0 * 1e400` is NaN to the evaluator).
            Ex::Leaf(t) => self.view.sigil(*t) == 0 && !self.f64_reads_degenerate(*t, false),
            Ex::Add(v) | Ex::Mul(v) => v.iter().all(|x| self.certainly_finite(x)),
            Ex::Pow(b, ex) => {
                self.certainly_finite(b)
                    && matches!(&**ex, Ex::Num(r) if r.is_integer() && !r.is_negative())
            }
            // `besselj` of an integer literal order is finite at every finite argument (|J_n| <= 1;
            // the kernel is NaN-free there up to order 1e3, `interval::BESSELJ_NU_MEASURED`).
            Ex::Fun(f, v) if v.len() == 2 && self.view.tok_is(*f, "besselj") => {
                matches!(&v[0], Ex::Num(n) if n.is_integer() && n.abs().cmp_int(1000) != Ordering::Greater)
                    && self.certainly_finite(&v[1])
            }
            Ex::Fun(f, v) => {
                let s = self.view.resolve_owned(*f);
                // (the retired `pow1_3`/`pow1_5` names are not in this list --
                // `from_prefix` maps those spellings to `rootn` unconditionally, so no
                // `Fun` ever carries them. `rootn` itself stays OUT: an even literal
                // index on a negative base is NaN, so it is not finiteness-preserving.)
                // `max` and `min` pick a member (a bag of any size, `lattice`).
                matches!(
                    s.as_str(),
                    "sin"
                        | "cos"
                        | "tanh"
                        | "atan"
                        | "abs"
                        | "exp"
                        | "sinh"
                        | "cosh"
                        | "asinh"
                        | "max"
                        | "min"
                ) && v.iter().all(|x| self.certainly_finite(x))
            }
            _ => false,
        }
    }

    /// `certainly_finite` for every element of a term key (a key is the non-coefficient factor
    /// list of an addend). Also the value-preservation licence of the exploration
    /// moves (`ac::search::distribute_product`), which is why it is crate-visible.
    pub(crate) fn fin_licensed(&self, e: &Ex) -> bool {
        self.permits(Permissive::LikeTerms) || self.fin_certified(e)
    }

    /// The certificate body of [`fin_licensed`] WITHOUT the lossy blanket: the answer the
    /// SOUND licence gives (the `nz_ae_certified` pattern). Lossy-mode callers use it where
    /// the blanket's premise cannot hold (`ac::search::distribute_product`, a piece that
    /// spells an infinity).
    pub(crate) fn fin_certified(&self, e: &Ex) -> bool {
        if self.certainly_finite(e) {
            return true;
        }
        // Ground expressions have no measure space: a.e. tolerance degenerates to exactness,
        // and only `certainly_finite` (exact) may license them (see `has_measure_space`).
        if !e.has_measure_space(self.view) {
            return false;
        }
        match self.cert_fin {
            Some(f) => f(e),
            None => false,
        }
    }

    /// Is `e` INFINITE ONLY ON A NULL SET of the variables' space? It may have no value on a
    /// region (`sqrt(x0)` for x0 < 0, `log(x0)` there too), but an infinity it takes is a pole,
    /// never a region (`1/x0` at x0 = 0, `log(x0)` at 0). The premise the distributive law asks
    /// of a piece in `f64` (`search.distribute`, `search.factor`), weaker than finite almost
    /// everywhere ([`Cx::fin_certified`], which implies it). Read by the form: a variable, a
    /// constant and a finite literal; sums and products of such pieces (each infinite only where
    /// a member is); a function bounded on the reals (`sin`, `tanh`, `asin`, `erf`, ...) of
    /// anything; a function that maps finite to finite or to no value (`exp`, `abs`, `acosh`,
    /// `floor`, ...) of such a piece; a power of one with a non-negative literal exponent, or with
    /// another exponent and a base whose zero set is null (`Cx::nz_ae_certified`), and a
    /// logarithm or a root of either sign of index likewise. A float64 overflow is a range event,
    /// not an infinity (as for `certainly_finite`). Anything else asks the finite a.e.
    /// certificate. Fail closed.
    pub(crate) fn inf_null(&self, e: &Ex) -> bool {
        match e {
            Ex::PosInf | Ex::NegInf => false,
            Ex::Num(_) | Ex::Pi | Ex::E | Ex::Const(_) | Ex::NaN => true,
            Ex::Leaf(_) => self.certainly_finite(e),
            Ex::Add(v) | Ex::Mul(v) => v.iter().all(|x| self.inf_null(x)),
            Ex::Pow(b, x) => {
                self.inf_null(b)
                    && match &**x {
                        Ex::Num(r) => !r.is_negative() || self.nz_ae_certified(b),
                        y => self.inf_null(y) && self.nz_ae_certified(b),
                    }
            }
            Ex::Fun(f, v) => {
                let is = |s: &str| self.view.tok_is(*f, s);
                let bounded = v.len() == 1
                    && ["sin", "cos", "tanh", "atan", "asin", "acos", "erf", "erfc"]
                        .iter()
                        .any(|s| is(s));
                let finite_to_finite = [
                    "exp", "sinh", "cosh", "asinh", "acosh", "abs", "floor", "ceil", "max", "min",
                ]
                .iter()
                .any(|s| is(s));
                if bounded {
                    true
                } else if finite_to_finite {
                    v.iter().all(|x| self.inf_null(x))
                } else if v.len() == 1 && is("log") {
                    self.inf_null(&v[0]) && self.nz_ae_certified(&v[0])
                } else if v.len() == 2 && is("rootn") {
                    match &v[1] {
                        Ex::Num(k) if k.is_integer() && !k.is_zero() => {
                            self.inf_null(&v[0])
                                && (!k.is_negative() || self.nz_ae_certified(&v[0]))
                        }
                        _ => self.fin_certified(e),
                    }
                } else {
                    self.fin_certified(e)
                }
            }
        }
    }

    /// Is `e` CERTAINLY finite and nonzero almost everywhere (the `$`-certificate), the `f64`
    /// premise of `mul.cancel` ([`Law::MulCancel`])? Variables qualify (a variable is zero only
    /// on a null set); `Pi`/`E` are nonzero literals; nonzero `Num` likewise; `Const` does NOT
    /// qualify syntactically (`c = 0` is a reachable fitted value, and the certificate's job is
    /// exactly to protect `c^1 * c^-1` at `c = 0`)... but the certificate may still pass it (the
    /// null set `{0}` in c-space is what `forall c` quantifies over -- we stay conservative and
    /// leave `Const` to the certificate).
    pub(crate) fn finnz_certified(&self, e: &Ex) -> bool {
        match e {
            Ex::Pi | Ex::E => true,
            Ex::Num(r) => !r.is_zero(),
            // f64 mode: `1e400/1e400` and `1e-400/1e-400` are NaN to the evaluator
            Ex::Leaf(t) => self.view.sigil(*t) == 0 && !self.f64_reads_degenerate(*t, true),
            _ => {
                if !e.has_measure_space(self.view) {
                    return false; // ground: no a.e. tolerance (see `has_measure_space`)
                }
                match self.cert_finnz {
                    Some(f) => f(e),
                    None => false,
                }
            }
        }
    }

    /// Zero-set-null licence for the odd-negative power distribution (see `pow`).
    /// `Const` QUALIFIES here ({c = 0} is a null set of the mask's own parameter space --
    /// unlike the SELFCANCEL licence, which deliberately protects the c = 0 atom);
    /// variables are zero on a null hyperplane; everything else goes to the structural
    /// zero-set certificate, fail-closed.
    pub(crate) fn nz_ae_licensed(&self, e: &Ex) -> bool {
        if self.permits(Permissive::Agreement) {
            return true;
        }
        self.nz_ae_certified(e)
    }

    /// The certificate body of [`nz_ae_licensed`] WITHOUT the lossy blanket: the answer
    /// the SOUND licence would give. Lossy-mode callers use it to ask "would sound-mode
    /// `pow` re-distribute this joined reciprocal?" (the `mul` rejoin arm) --
    /// inside lossy mode `nz_ae_licensed` is constant true and cannot pose that question.
    pub(crate) fn nz_ae_certified(&self, e: &Ex) -> bool {
        match e {
            Ex::Pi | Ex::E | Ex::Const(_) => true,
            Ex::Num(r) => !r.is_zero(),
            Ex::Leaf(t) => self.view.sigil(*t) == 0,
            _ => match self.cert_nzae {
                Some(f) => f(e),
                None => false,
            },
        }
    }

    /// Is `e` CERTAINLY non-negative wherever it is defined, syntactically? Every clause is a
    /// theorem of the form "wherever this expression takes a value, that value is >= 0" --
    /// undefined (NaN-domain) points are outside the claim, so consumers must pair it with
    /// identities that agree on the undefined set. Conservative by construction: `false` means
    /// "not certain", never "certainly negative".
    ///
    /// Consumers: the branch-cut licence for merging NON-INTEGER rational exponents
    /// (`x^(1/2) * x^(1/2) -> x` is WRONG on `x < 0` -- NaN vs x, full measure -- but sound on
    /// a certainly-non-negative base), the pow licences, and `fun`'s constructor-level abs
    /// elimination (`abs t -> t`, a `_`-grade pointwise identity on such an operand).
    ///
    /// A useful induction the abs fold leans on: no clause below can EVALUATE to IEEE -0.0
    /// (even powers, exp/cosh, sums/products of such, ... all land on +0.0 when they vanish),
    /// so `abs t -> t` is exact even under signed-zero observation -- though the one-zero
    /// contract (SIMPLIFICATION_CONTRACT_v2 s9.2) already makes the zero sign non-normative.
    /// Can this subtree NEVER evaluate to an infinity? FAIL-CLOSED: leaves only.
    ///
    /// §9.1 quantifies variables, wildcard bindings and constants over the REALS, and §10.8's
    /// free-constant convention makes an infinity a LIMIT rather than an attained value, so no
    /// LEAF ever takes one as a value. Every COMPOUND is refused, and that refusal is the whole
    /// content of the predicate: `inv(x0)` is `+inf` at `x0 = 0` and `neg(inv(x0))` is `-inf`
    /// there, which is the counterexample the predicate exists for. `Ex::PosInf` / `Ex::NegInf` SPELL an
    /// infinity and are excluded for that reason; `Ex::NaN` is not a real value at all.
    ///
    /// Used by the power-of-power composition, whose odd-negative-exponent arm is sound exactly
    /// when the base cannot reach an infinity (see the licence comment in `pow`).
    pub(crate) fn never_infinite(&self, e: &Ex) -> bool {
        matches!(e, Ex::Leaf(_) | Ex::Const(_) | Ex::Num(_) | Ex::Pi | Ex::E)
    }

    pub(crate) fn certainly_nonneg(&self, e: &Ex) -> bool {
        match e {
            Ex::Num(r) => !r.is_negative(),
            Ex::Pi | Ex::E | Ex::PosInf => true,
            // Sums and products of certainly-non-negative parts: with no negative part
            // reachable there is no cancellation or sign flip; an infinite part drives the
            // bag to +inf or NaN, both within the claim.
            Ex::Add(v) | Ex::Mul(v) => v.iter().all(|t| self.certainly_nonneg(t)),
            // Even integer exponents force b^n >= 0 for ANY base; a certainly-non-negative
            // base stays non-negative under EVERY real exponent (0^negative = +inf under the
            // one-zero contract -- still non-negative). A power of a non-integer exponent is the
            // principal power, `>= 0` wherever it has a value: NaN on b < 0, `+0` or `+inf` at
            // b = -0 and `+inf` or `+0` at b = -inf (the square root numpy keeps `-0` in is the
            // spelled call, no `Pow`).
            Ex::Pow(b, ex) => {
                matches!(&**ex, Ex::Num(r) if r.is_even_integer() || !r.is_integer())
                    || self.certainly_nonneg(b)
            }
            Ex::Fun(f, v) => {
                let s = self.view.resolve_owned(*f);
                match s.as_str() {
                    // Ranges within [0, +inf] for every argument: |.|, exp, cosh; acos
                    // ([0, pi]), acosh ([0, inf)).
                    "abs" | "exp" | "cosh" | "acos" | "acosh" => true,
                    // `rootn` was MISSING here, and the four `pow1_*` spellings that stood
                    // in its place were DEAD: `from_prefix` maps them to `rootn`
                    // unconditionally, so no `Fun` ever carries one. The gap is invisible
                    // on the shipped engine because the ruleset masks it for whatever
                    // shapes the mine happened to cover -- `abs(rootn(x,2))` folds by RULE,
                    // and a `rules=[]` engine keeps the `abs`.
                    "rootn" => {
                        v.len() == 2
                            && match &v[1] {
                                // EVEN index: defined only on [0, inf) and its
                                // principal value is non-negative there -- a range
                                // fact, exactly like `acos`. Off-domain it is NaN,
                                // which no licence reads as a sign. (A negative even
                                // index is the reciprocal of one, still non-negative;
                                // index 0 is excluded, it is not a root.)
                                Ex::Num(r) if r.is_even_integer() && !r.is_zero() => true,
                                // ODD index: an increasing odd bijection on the whole
                                // line, so it maps [0, inf) into [0, inf) -- the same
                                // clause as `sinh`/`asinh` below.
                                Ex::Num(r) if r.is_integer() => self.certainly_nonneg(&v[0]),
                                _ => false,
                            }
                    }
                    // GLOBALLY increasing odd functions map [0, inf) into [0, inf); asin is
                    // increasing on its whole domain. `tan` is deliberately absent: odd but
                    // periodic (tan 2 < 0).
                    "sinh" | "tanh" | "asinh" | "atan" | "asin" => {
                        v.len() == 1 && self.certainly_nonneg(&v[0])
                    }
                    // cos of a [-1, 1]-bounded argument: cos restricted to [-1, 1] is
                    // >= cos(1) > 0 (1 < pi/2).
                    "cos" => v.len() == 1 && self.certainly_in_unit(&v[0]),
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Is `f` EVEN about the origin -- `f(-t) = f(t)` at every value, including the
    /// extension's? A CERTIFIED table, deliberately keyed by operator NAME and owned by the
    /// engine rather than declared in a config: this is a soundness licence, exactly like
    /// `certainly_nonneg` above, and a config author writing `parity: even` on an operator
    /// whose realization is not even would silently corrupt the canon.
    ///
    /// `abs`, `cos`, `cosh` -- and the extension agrees at both ends (`cos(-inf)` and
    /// `cos(inf)` are both NaN; `abs`/`cosh` both +inf).
    pub(super) fn is_even_fun(&self, f: Tok) -> bool {
        self.view.tok_is(f, "abs") || self.view.tok_is(f, "cos") || self.view.tok_is(f, "cosh")
    }

    /// Is `f` ODD about the origin -- `f(-t) = -f(t)` at every value? The eight globally
    /// odd members of the vocabulary. Each holds on the extension: `sin`/`tan` propagate the
    /// sign and are NaN at both infinities; `sinh`/`tanh`/`asinh`/`atan` are odd bijections
    /// with matching signed limits; `asin`/`atanh` are odd on their domain and NaN off it on
    /// both sides.
    ///
    /// `inv` is DELIBERATELY ABSENT and its absence is load-bearing. It is odd on the reals
    /// minus the origin and NOT odd here: with one zero (contract §9.2) `neg(0) = 0`, so
    /// `inv(neg 0) = +inf` while `neg(inv 0) = -inf` (measured, not argued). That is §9.8.4's
    /// pole trilemma, which is a THEOREM: no total semantics with `+inf != -inf` keeps both
    /// pole symmetry and absorption, and this contract keeps absorption. Nothing is lost --
    /// the `Pow` arm of `drop_sign` takes POSITIVE odd exponents only, where `(-x)^n =
    /// -(x^n)` holds at 0 and at both infinities as well.
    pub(crate) fn is_odd_fun(&self, f: Tok) -> bool {
        odd_fun(self.view, f)
    }

    /// The representative of `e` in a SIGN-BLIND context -- a position whose consumer cares
    /// only about `|value|` -- or `None` when `e` is already it.
    ///
    /// Sign-blindness is a property of the CONTEXT, not of a node, and it PROPAGATES. An even
    /// function plants it (the arm in `fun` is the only place that does); from there:
    ///
    ///   * a CARRIED sign is dropped: a negative literal, `-inf`, a bag's negative
    ///     coefficient. `neg 3`, `-3` and `(-1)*x` are one case here and three token patterns
    ///     to a rule. That is the whole point -- this reads the canonical SIGN, never a `neg`
    ///     NODE, so the constructor's own fold of `neg(3)` into `-3` cannot blind it, which is
    ///     exactly how all thirty mined rules went blind at every numeric binding.
    ///   * an ODD function propagates sign-blindness INWARD: `f(-t) = -f(t)`, and the context
    ///     discards that sign anyway, so the recursion continues through it.
    ///   * an `abs` is a NO-OP and is stripped: its whole job was erasing a sign this context
    ///     erases too. `|tanh(|x|)| = |tanh(x)|` falls out of this line. The miner minted
    ///     exactly that rule while the arm stripped `abs` only at the TOP -- the gap was
    ///     found by the mine, not by inspection, and the fix was to make propagation the rule
    ///     rather than a special case.
    ///   * an INTEGER power propagates: an EVEN exponent erases the base's sign
    ///     (`(-u)^2 = u^2`), a POSITIVE ODD one carries it out. The negative odd half is
    ///     refused -- see `is_odd_fun` on the pole trilemma.
    ///
    /// Structural recursion, so it terminates; a projection, so it is idempotent.
    /// `-e` when `e` SYNTACTICALLY carries a negative sign, `None` otherwise. The one place
    /// that decides what "carries a sign" means, so the two consumers cannot drift: a
    /// sign-blind context drops it (`sign_blind_rep`) and a literal power base absorbs it
    /// (`pow`). The claim is about the SPELLING, not the value -- `Mul[-1, x0]` is positive
    /// at x0 < 0 -- which is exactly what both consumers need.
    pub(super) fn carried_negative(&self, e: &Ex) -> Option<Ex> {
        match e {
            Ex::Num(r) if r.is_negative() => r.checked_neg().map(Ex::Num),
            Ex::NegInf => Some(Ex::PosInf),
            Ex::Mul(v) if v.iter().any(|m| matches!(m, Ex::Num(r) if r.is_negative())) => {
                let mut out = Vec::with_capacity(v.len());
                for m in v {
                    match m {
                        Ex::Num(r) if r.is_negative() => out.push(Ex::Num(r.checked_neg()?)),
                        other => out.push(other.clone()),
                    }
                }
                Some(mul(out, self))
            }
            _ => None,
        }
    }

    pub(super) fn sign_blind_rep(&self, e: &Ex) -> Option<Ex> {
        if let Some(flipped) = self.carried_negative(e) {
            return Some(flipped);
        }
        match e {
            Ex::Fun(f, a) if a.len() == 1 && self.view.tok_is(*f, "abs") => {
                Some(self.sign_blind_rep(&a[0]).unwrap_or_else(|| a[0].clone()))
            }
            Ex::Fun(f, a) if a.len() == 1 && self.is_odd_fun(*f) => {
                self.sign_blind_rep(&a[0]).map(|s| fun(*f, vec![s], self))
            }
            Ex::Fun(f, a) if a.len() == 2 && self.view.tok_is(*f, "besselj") => {
                let (pos, _) = odd_head(self.view, *f, a)?;
                self.sign_blind_rep(&a[pos])
                    .map(|s| fun(*f, vec![a[0].clone(), s], self))
            }
            Ex::Pow(b, ex) => {
                let ok = matches!(&**ex, Ex::Num(r)
                    if r.is_even_integer() || (r.is_integer() && r.signum() > 0));
                if !ok {
                    return None;
                }
                self.sign_blind_rep(b).map(|s| pow(s, (**ex).clone(), self))
            }
            // A product's sign is erased here too, so each of its sign-trade sites takes its
            // cheaper spelling whatever sign that leaves, and on an equal price the tie order
            // (`tie_cmp`), a normal-form choice: `|333 atan(u - 0.66)^5|` is
            // `|333 atan(0.66 - u)^5|`, a sign bit cheaper, where the product alone keeps its
            // coefficient positive on the tie (a sign bit for nothing), and
            // `cos(x1*rootn(-x2*x3, 3))` is `cos(x1*rootn(x2*x3, 3))`.
            Ex::Mul(v) => {
                let mut out = v.clone();
                let mut moved = false;
                for f in out.iter_mut() {
                    if let Some(nf) = sign_trade_flip(f, self) {
                        let (mn, mf) = (complexity(&nf, self.view), complexity(f, self.view));
                        if mn < mf || (mn == mf && tie_cmp(&nf, f, self.view) == Ordering::Less) {
                            *f = nf;
                            moved = true;
                        }
                    }
                }
                if !moved {
                    return None;
                }
                let m = mul(out, self);
                Some(self.carried_negative(&m).unwrap_or(m))
            }
            _ => None,
        }
    }

    /// A proven CEILING on `|e|`, in exact integers, or `None` when none is known.
    ///
    /// CONSERVATIVE BY CONSTRUCTION: a bound is returned only where it is provable and
    /// anything unrecognised yields `None`. Read on the INTERNAL form, never on the token
    /// spelling, so the canon unifying two spellings cannot change the answer (the
    /// spelling-key hazard) -- the discipline `certainly_nonneg` already follows. Integer arithmetic
    /// throughout: this guard must not acquire an f64 reading of a `Rat`.
    ///
    /// EXISTS for `fun.atanh_tanh`'s band in `f64`: float64's `tanh` is `+-1` from |t| = 19.0615
    /// on, where `atanh` gives an infinity against `t` (`inverse_pair_collapse`).
    fn magnitude_ceiling(&self, e: &Ex) -> Option<i128> {
        match e {
            Ex::Num(r) => r.ceil_abs(),
            Ex::Pi => Some(4),
            Ex::E => Some(3),
            Ex::Fun(f, a) => {
                let s = self.view.resolve_owned(*f);
                match s.as_str() {
                    // bounded RANGES, whatever the argument
                    "sin" | "cos" | "tanh" => Some(1),
                    "atan" | "asin" => Some(2),
                    "acos" => Some(4),
                    // magnitude-preserving, and `floor` and `ceil` of a part within an integer
                    // bound stay within it
                    "abs" | "neg" | "floor" | "ceil" => {
                        a.first().and_then(|t| self.magnitude_ceiling(t))
                    }
                    _ => None,
                }
            }
            // |sum| <= sum of ceilings, |product| <= product of ceilings -- each needs
            // EVERY part bounded, and overflow abstains rather than wrapping.
            Ex::Add(v) => v.iter().try_fold(0i128, |acc, t| {
                self.magnitude_ceiling(t).and_then(|m| acc.checked_add(m))
            }),
            Ex::Mul(v) => v.iter().try_fold(1i128, |acc, t| {
                self.magnitude_ceiling(t).and_then(|m| acc.checked_mul(m))
            }),
            _ => None,
        }
    }

    /// Is `|e| <= bound` PROVEN? `false` whenever unknown -- refusal is the safe direction.
    pub(crate) fn certainly_in_band(&self, e: &Ex, bound: i128) -> bool {
        self.magnitude_ceiling(e).is_some_and(|m| m <= bound)
    }

    /// Is `e` certainly in [-1, 1] wherever it has a value: `sin`, `cos` or `tanh` of something?
    pub(crate) fn certainly_in_unit(&self, e: &Ex) -> bool {
        match e {
            Ex::Fun(f, _) => {
                let s = self.view.resolve_owned(*f);
                matches!(s.as_str(), "sin" | "cos" | "tanh")
            }
            _ => false,
        }
    }

    /// Is `m` a pi member of an Add bag whose coefficient is EXACTLY +-1?
    ///
    /// The coefficient bound is the whole scope of the pi-shift arm. In a canonical bag like
    /// terms merge, so `t + pi + pi` is a COEFFICIENT-2 pi member, and folding that is
    /// 2pi-PERIODICITY -- genuine argument reduction modulo the period, a concept the engine
    /// does not have. Keyed on +-1 the arm cannot reach it. There is no corpus pressure for
    /// the wider fact either: `sin + x0 * 2 np.pi` is six tokens, past the mine's source cap.
    fn is_unit_pi(m: &Ex) -> bool {
        match m {
            Ex::Pi => true,
            // `-pi` is a two-factor product and nothing else: a third factor (`-pi*x0`) is a
            // different member and must not strip.
            Ex::Mul(v) => {
                v.len() == 2
                    && v.iter().any(|x| matches!(x, Ex::Pi))
                    && v.iter()
                        .any(|x| matches!(x, Ex::Num(r) if *r == Rat::NEG_ONE))
            }
            _ => false,
        }
    }

    /// `u` with one coefficient-+-1 `pi` member removed, or `None`. Reads the CANONICAL bag,
    /// which is the entire point: at a negative literal the constructor factors the bag's
    /// common sign, so `(-2) - pi` is stored as `neg(2 + pi)` and the `-pi` member a pattern
    /// rule needs no longer exists. That is why `sin - _0 np.pi -> neg sin _0` fires at a
    /// variable and at `2` but goes blind at `(-2)` -- the same defect, one family over.
    ///
    /// The negated-bag case needs no parity argument: `u = -(v' + pi) = (-v') - pi`, so
    /// stripping under the negation and RE-NEGATING the remainder lands on the same shift law
    /// as the direct case. One rule for both.
    pub(super) fn strip_unit_pi(&self, u: &Ex) -> Option<Ex> {
        let strip = |ms: &Vec<Ex>| -> Option<Ex> {
            let i = ms.iter().position(Self::is_unit_pi)?;
            let rest: Vec<Ex> = ms
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, m)| m.clone())
                .collect();
            Some(add(rest, self))
        };
        if let Ex::Add(ms) = u {
            return strip(ms);
        }
        if let Some(Ex::Add(ms)) = self.carried_negative(u) {
            let stripped = strip(&ms)?;
            return Some(mul(vec![Ex::Num(Rat::NEG_ONE), stripped], self));
        }
        None
    }

    /// `outer(inner(t))` collapsed, when the pair is an INVERSE pair whose law holds of `t` --
    /// `None` otherwise. Each pair is a law of the catalogue with its premise per mode, the
    /// judge's. Float64's round trips drift by many units in the last place (`atanh o tanh` past 8
    /// from |t| ~ 2.7, `log o exp` and `acosh o cosh` near 0) while both sides stay finite: the
    /// rounding keeps the kind, so no pair carries a rounding band.
    ///
    ///   TOTAL -- the bijections of the extended line: `asinh o sinh`, `sinh o asinh` and
    ///   `acosh o cosh` (`fun.asinh_sinh`, `fun.sinh_asinh`, `fun.acosh_cosh`). `acosh o cosh` is
    ///   NOT the identity: `cosh` is even, so the collapse is `|t|`. Where the inner function
    ///   overflows (`sinh` and `cosh` past 710.48) a rewrite may give the true value.
    ///
    ///   A BAND OR A ZERO'S SIGN in `f64` -- `atanh o tanh` where `|t|` is at most 19, short of
    ///   float64's `tanh(19.0615) = 1` (`fun.atanh_tanh`); `log o exp` where `t` is not `-0` on a
    ///   region, `log(exp(-0))` being `+0` (`fun.log_exp`; `exp`'s overflow and underflow are
    ///   range events).
    ///
    ///   A FLAT POINT in `f64` -- `log o exp` and `acosh o cosh` are flat at 0: float64's
    ///   `log(exp(t))` is `+0` wherever 0 < |t| < 1.1e-16 and `acosh(cosh(t))` is 0 wherever
    ///   |t| < 1.5e-8, so a head that reads a zero above them (an inverse, `log`, ...) turns the
    ///   rounding into a change of kind. A construction does not see what is around it, so it
    ///   holds both pairs, and the root's walk collapses each where nothing reads its flat point
    ///   (`release_flat`, `Cx::flat_unread`).
    ///
    ///   A DOMAIN -- `exp o log` where `t >= 0` (and in `f64` not `-0` on a region,
    ///   `fun.exp_log`), and the three arcsin-side pairs where `t` is in [-1, 1]
    ///   (`fun.unit_inverses`): elsewhere the left side has no value and the right side has.
    ///
    ///   REFUSED ENTIRELY. `asin o sin`, `acos o cos` and `atan o tan` are NOT identities in
    ///   any direction -- the judge convicts them under clause (a), a REAL CHANGE, because
    ///   they fold the argument into the inverse's range (`asin(sin 4) = pi - 4`). `tan o atan`
    ///   is left out too: it holds but at the null-set pole `atan(inf) = pi/2`, a rule-level
    ///   allowance under a finiteness certificate.
    ///
    /// In `permissive` every pair cancels in both orders, ignoring domain and branch
    /// (`perm.inverse_pairs`, one of its own simplifications).
    ///
    /// The laws applied here (`ac::laws::Law`): `fun.atanh_tanh`, `fun.log_exp`,
    /// `fun.asinh_sinh`, `fun.sinh_asinh`, `fun.acosh_cosh`, `fun.exp_log`, `fun.unit_inverses`,
    /// `perm.inverse_pairs`.
    ///
    /// Every arm drops two operator nodes, so the descent is unconditional.
    pub(super) fn inverse_pair_collapse(&self, outer: Tok, inner: Tok, t: &Ex) -> Option<Ex> {
        let o = self.view.resolve_owned(outer);
        let i = self.view.resolve_owned(inner);
        if Law::PermInversePairs.holds(t, self) && cancels_its_inverse(&o, &i) {
            return Some(t.clone());
        }
        // In `f64` a fully numeric operand is read for its value first (`num.ground_first`): a
        // pair's law speaks of parts with a variable or a constant, whose rounding keeps the kind,
        // while a ground part's kind is its whole value's (`rootn(6, atanh(tanh(10)))` is NaN in
        // float64, its index 10.000000000000002).
        if self.f64_semantics()
            && !t.has_measure_space(self.view)
            && !t.contains_wildcard(self.view)
        {
            return None;
        }
        let law = match (o.as_str(), i.as_str()) {
            ("atanh", "tanh") => Law::FunAtanhTanh,
            ("log", "exp") => Law::FunLogExp,
            ("asinh", "sinh") => Law::FunAsinhSinh,
            ("sinh", "asinh") => Law::FunSinhAsinh,
            ("acosh", "cosh") => Law::FunAcoshCosh,
            ("exp", "log") => Law::FunExpLog,
            ("cos", "acos") | ("sin", "asin") | ("tanh", "atanh") => Law::FunUnitInverses,
            _ => return None,
        };
        if !law.holds(t, self) {
            return None;
        }
        Some(if matches!(law, Law::FunAcoshCosh) {
            fun(self.view.intern("abs"), vec![t.clone()], self)
        } else {
            t.clone()
        })
    }
}
