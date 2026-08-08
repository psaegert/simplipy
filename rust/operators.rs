//! Operator metadata, mirroring `simplipy/operators.py` + the operator block of an engine
//! `config.yaml` (arity, inverse, precedence, commutativity, realization, aliases).
//!
//! Built from the parsed `config.yaml`. This is the Rust analogue of the tables
//! `SimpliPyEngine.__init__` constructs (operator_arity, operator_inverses, commutative_operators,
//! operator_precedence_compat, inverse_base/unary/binary, connection_classes, ...). v1 only needs
//! the subset the deployed skeleton path touches; the rest are stubs to fill as ported.

use rustc_hash::FxHashMap;
use serde::Deserialize;

/// The AC CORE SERIALIZATION LANGUAGE: `(token, arity, infix precedence)` for every sugar
/// operator the serializers emit REGARDLESS of the config vocabulary (`ac::convert`'s
/// explicit/tagged emitters and the pretty-infix renderer). These are engine BUILT-INS,
/// not config operators: every engine reads them unconditionally so it can always re-read
/// its own output (emit alphabet <= parse alphabet) under ANY config -- without this, a
/// config missing e.g. `neg` made `simplify(simplify(e))` raise on the engine's own
/// output and made the mine's ordering acceptance silently refuse correct rules. A config
/// MAY declare these tokens (supplying realizations/precedences for the legacy paths) but
/// never with a conflicting arity, and never as an alias of another operator -- both are
/// build-time config errors (`Engine::from_strs`).
pub const CORE_SERIALIZATION_OPS: [(&str, u8, f64, Option<&str>); 8] = [
    ("+", 2, 1.0, None),
    ("-", 2, 1.0, None),
    ("*", 2, 2.0, None),
    ("/", 2, 2.0, Some("simplipy.operators.div")),
    ("neg", 1, 2.5, Some("simplipy.operators.neg")),
    ("inv", 1, 4.0, Some("simplipy.operators.inv")),
    ("pow", 2, 3.0, Some("simplipy.operators.pow")),
    ("rootn", 2, 3.0, Some("simplipy.operators.rootn")),
];

/// The built-in arity of a core serialization token (`None` for everything else).
#[inline]
pub fn core_arity(token: &str) -> Option<u8> {
    CORE_SERIALIZATION_OPS
        .iter()
        .find(|(t, ..)| *t == token)
        .map(|(_, a, ..)| *a)
}

/// The built-in PRECEDENCE of a core serialization token (`None` for everything else).
///
/// Load-bearing for the same reason as [`core_arity`]: the realization rendering is a
/// string handed to PYTHON's `eval`, and Python's precedence is fixed. A config that
/// declares `/` at precedence 0.5 renders `(x0+x1)/x2` as `x0 + x1 / x2`, which this
/// engine re-parses correctly (its parser shares the same table) but Python evaluates as
/// `x0 + (x1/x2)` -- 2.5 where the value is 2.0. A silent wrong answer at the eval
/// boundary, which is exactly the closure the core-token guard exists to hold.
#[inline]
pub fn core_precedence(token: &str) -> Option<f64> {
    CORE_SERIALIZATION_OPS
        .iter()
        .find(|(t, ..)| *t == token)
        .map(|(_, _, p, _)| *p)
}

#[derive(Debug, Clone, Deserialize)]
pub struct OperatorSpec {
    #[serde(default)]
    pub realization: String,
    #[serde(default)]
    pub alias: Vec<String>,
    /// PARSE-ONLY at 0.12.0 (D3): shipped configs declare `inverse:` entries, so the
    /// field stays accepted for deserialization compatibility, but nothing derives from
    /// it any more -- the `operator_inverses` map and its one (dead) intern site were
    /// removed with the binary kernel's relic layer. The shipped acj configs' historic
    /// typos (`acosh: cos`, `atanh: tan` -- missing the `h`) were FIXED 2026-08-05;
    /// legacy configs in the wild still carry them (the legacy fixture reproduces one
    /// verbatim), which is why `worker.rs::inverse_unary_name` keeps its own map
    /// rather than trusting the field.
    #[serde(default)]
    pub inverse: Option<String>,
    pub arity: u8,
    #[serde(default)]
    pub precedence: Option<f64>,
    #[serde(default)]
    pub commutative: bool,
}

/// The engine's operator universe. Insertion order is preserved (config.yaml order) because some
/// faithful behaviours (default precedence fallback = enumeration index) depend on it.
#[derive(Debug, Default)]
pub struct Operators {
    pub arity: FxHashMap<String, u8>,
    pub commutative: Vec<String>,
    /// `max_power`: `max(int(op[3:]) for op in operator_tokens if op matches
    /// `^pow\d+` NOT followed by `_`)`, else 0. The factor ceiling for the LEGACY
    /// conversion paths' power-chain factorization (`convert.rs`; the other historical
    /// consumer, `cancel_terms`, is deleted).
    pub max_power: i64,
    /// `max_fractional_power`: `max(int(op[5:]) for op in operator_tokens if op
    /// matches `^pow1_\d+`)`, else 0. The factor ceiling for `convert_expression`'s fractional-power
    /// chain combining (dev_7-3: pow1_2..pow1_5 -> 5).
    pub max_fractional_power: i64,
    /// `operator_precedence_compat`: `{name: spec.precedence if present else
    /// ENUMERATION-INDEX i}` in config order, THEN `['**'] = 3` and `['sqrt'] = 3` overlaid. Consumed
    /// by `prefix_to_infix` (paren decisions) and `infix_to_prefix` (shunting-yard precedence pops).
    /// f64 because `neg` declares the float precedence 2.5. The enum-index default never fires for
    /// dev_7-3 (all 38 ops declare a precedence) but is ported faithfully for config-robustness.
    pub operator_precedence_compat: FxHashMap<String, f64>,
    /// `operator_arity_compat`: `deepcopy(operator_arity)` + `'**' -> 2` (the Python-style
    /// power token). Consumed by the infix conversion paths; the historical consumer
    /// (`sort_operands`, the deleted operand-sort pass) is gone.
    pub operator_arity_compat: FxHashMap<String, u8>,
    /// `operator_aliases`: `{alias: operator}` over every operator's `alias` list.
    /// The conversion paths resolve an input token to its canonical operator through it.
    pub operator_aliases: FxHashMap<String, String>,
    /// `operator_realizations`: `{name: realization}` (e.g. `sin` ->
    /// `simplipy.operators.sin`, `+` -> `+`). Used by `operators_to_realizations`.
    pub operator_realizations: FxHashMap<String, String>,
    /// `realization_to_operator`: the inverse, built in config order so a (here
    /// absent) realization collision would resolve last-wins exactly as Python's dict comprehension.
    pub realization_to_operator: FxHashMap<String, String>,
}

impl Operators {
    pub fn from_specs(order: Vec<String>, specs: FxHashMap<String, OperatorSpec>) -> Self {
        let arity: FxHashMap<String, u8> =
            specs.iter().map(|(k, v)| (k.clone(), v.arity)).collect();
        let commutative = order
            .iter()
            .filter(|k| specs.get(*k).map(|s| s.commutative).unwrap_or(false))
            .cloned()
            .collect();
        let max_power = order.iter().filter_map(|t| pow_power(t)).max().unwrap_or(0);
        let max_fractional_power = order
            .iter()
            .filter_map(|t| pow1_power(t))
            .max()
            .unwrap_or(0);
        // operator_precedence_compat: declared precedence, else the config enumeration index, then
        // overlay '**' = 3 and 'sqrt' = 3.
        let mut operator_precedence_compat: FxHashMap<String, f64> = FxHashMap::default();
        for (i, name) in order.iter().enumerate() {
            let prec = specs
                .get(name)
                .and_then(|s| s.precedence)
                .unwrap_or(i as f64);
            operator_precedence_compat.insert(name.clone(), prec);
        }
        operator_precedence_compat.insert("**".to_string(), 3.0);
        operator_precedence_compat.insert("sqrt".to_string(), 3.0);
        let operator_aliases = order
            .iter()
            .flat_map(|name| {
                let name = name.clone();
                specs
                    .get(&name)
                    .map(|s| s.alias.clone())
                    .unwrap_or_default()
                    .into_iter()
                    .map(move |a| (a, name.clone()))
            })
            .collect();
        let mut operator_arity_compat: FxHashMap<String, u8> = arity.clone();
        operator_arity_compat.insert("**".to_string(), 2);
        // The core serialization language joins the infix-compat maps too (the tables
        // `infix_to_prefix`/`convert_expression` consult), insert-if-absent so a config's
        // own declaration wins -- same mechanism as the '**'/'sqrt' overlays. Without
        // this, a config missing '/' parsed the renderer's own 'x0 + 1/x0' as
        // '(x0+1)/x0' (unknown operator -> precedence 0), silently changing the value.
        for (tok, ar, prec, _) in CORE_SERIALIZATION_OPS {
            operator_arity_compat.entry(tok.to_string()).or_insert(ar);
            operator_precedence_compat
                .entry(tok.to_string())
                .or_insert(prec);
        }
        // Build the realization maps in config (`order`) order so the inverse resolves last-wins.
        let mut operator_realizations: FxHashMap<String, String> = FxHashMap::default();
        let mut realization_to_operator: FxHashMap<String, String> = FxHashMap::default();
        for name in &order {
            let realization = specs
                .get(name)
                .map(|s| s.realization.clone())
                .unwrap_or_default();
            operator_realizations.insert(name.clone(), realization.clone());
            realization_to_operator.insert(realization, name.clone());
        }
        // The core serialization language again, for the SAME reason as the arity and
        // precedence fallbacks above: the projections can emit a core operator the config
        // never declared (`x0 * x0` serializes as `pow x0 2` under a config whose only
        // operator is `*`), and a realization map that lacks it renders the BARE name --
        // which Python's builtins then read with the wrong semantics (`pow(-2.0, 1.5)` is
        // a COMPLEX number where the engine gives NaN; `rootn` is a NameError; a bare `/`
        // raises ZeroDivisionError where the engine gives inf). `or_insert` keeps a
        // config's own declaration winning, exactly like the two loops above. `+`, `-`
        // and `*` carry `None`: they render as Python infix operators whose semantics
        // already match, so they need no realization.
        for (tok, _, _, realization) in CORE_SERIALIZATION_OPS {
            if let Some(r) = realization {
                operator_realizations
                    .entry(tok.to_string())
                    .or_insert_with(|| r.to_string());
                realization_to_operator
                    .entry(r.to_string())
                    .or_insert_with(|| tok.to_string());
            }
        }
        Self {
            arity,
            commutative,
            max_power,
            max_fractional_power,
            operator_precedence_compat,
            operator_arity_compat,
            operator_aliases,
            operator_realizations,
            realization_to_operator,
        }
    }

    #[inline]
    pub fn arity_of(&self, token: &str) -> Option<u8> {
        // The core serialization language (see `CORE_SERIALIZATION_OPS`): engine
        // built-ins, recognized before -- and identically regardless of -- the config
        // vocabulary, so every engine re-reads its own output. The build-time guard in
        // `Engine::from_strs` keeps a config from declaring one with another arity.
        if let Some(a) = core_arity(token) {
            return Some(a);
        }
        self.arity.get(token).copied()
    }

    #[inline]
    pub fn is_operator(&self, token: &str) -> bool {
        core_arity(token).is_some() || self.arity.contains_key(token)
    }

    /// `operator in self.commutative_operators`. dev_7-3: `{'+', '*'}`.
    #[inline]
    pub fn is_commutative(&self, token: &str) -> bool {
        self.commutative.iter().any(|c| c == token)
    }

    /// `operator_precedence_compat.get(token)`. `None` if the token is unknown;
    /// call sites apply their own default (`.get(t, 0)` in `infix_to_prefix`, the `pow`/inf fallback
    /// in `prefix_to_infix`).
    #[inline]
    pub fn precedence_get(&self, token: &str) -> Option<f64> {
        self.operator_precedence_compat.get(token).copied()
    }

    /// `token in self.operator_tokens` (the 38 config operator names). Equivalent to
    /// `arity.contains_key` since `arity` is built from exactly those keys (excludes the synthetic
    /// `**` that only `operator_arity_compat` carries). Used by `prefix_to_infix`'s operator test.
    #[inline]
    pub fn is_operator_token(&self, token: &str) -> bool {
        self.arity.contains_key(token)
    }
}

/// Match the Python regex `^pow\d+(?!_)` and return `int(token[3:])` if it matches, else `None`.
/// `pow2` -> Some(2); `pow10` -> Some(10); `pow2_3` -> None (the `_` fails the negative lookahead);
/// `power` / `pow` (no digits) -> None. Greedy `\d+` consumes ALL trailing digits, so the lookahead
/// only inspects the single char after the last digit (end-of-string passes).
pub(crate) fn pow_power(token: &str) -> Option<i64> {
    let rest = token.strip_prefix("pow")?;
    // one-or-more leading digits
    let digit_end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    if digit_end == 0 {
        return None; // `\d+` requires at least one digit
    }
    // negative lookahead `(?!_)`: the char immediately after the digit run must not be `_`.
    if rest[digit_end..].starts_with('_') {
        return None;
    }
    // `int(op[3:])`: faithful only when the entire post-`pow` slice is digits (true for dev_7-3
    // `pow2..pow5`). If a non-digit/non-`_` tail ever appears, Python's `int(op[3:])` would raise;
    // we mirror "no contribution" rather than panic, since such tokens are absent from the grammar.
    rest[..digit_end].parse::<i64>().ok()
}

/// Match the Python regex `^pow1_\d+` and return `int(token[5:])` if it matches, else `None`.
/// `pow1_2` -> Some(2); `pow1_5` -> Some(5); `pow2` -> None (no `1_` infix); `pow1_` -> None
/// (`\d+` needs at least one digit). `int(op[5:])` reads ALL chars after `pow1_`; for dev_7-3
/// (`pow1_2..pow1_5`) that is exactly the digit run.
pub(crate) fn pow1_power(token: &str) -> Option<i64> {
    let rest = token.strip_prefix("pow1_")?;
    // `\d+`: require at least one leading digit, then `int(op[5:])` parses the whole tail.
    if !rest.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        return None;
    }
    rest.parse::<i64>().ok()
}
