//! Operator metadata, mirroring `simplipy/operators.py` + the operator block of an engine
//! `config.yaml` (arity, inverse, precedence, commutativity, realization, aliases).
//!
//! Built from the parsed `config.yaml`. This is the Rust analogue of the tables
//! `SimpliPyEngine.__init__` constructs (operator_arity, operator_inverses, commutative_operators,
//! operator_precedence_compat, inverse_base/unary/binary, connection_classes, ...). v1 only needs
//! the subset the deployed skeleton path touches; the rest are stubs to fill as ported.

use rustc_hash::FxHashMap;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct OperatorSpec {
    #[serde(default)]
    pub realization: String,
    #[serde(default)]
    pub alias: Vec<String>,
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
    /// `operator_inverses` (engine.py:145): `{k: v["inverse"]}` for operators with a declared
    /// inverse. `cancel_terms` reads `operator_inverses["+"]` / `["*"]` to decide parity flips.
    pub operator_inverses: FxHashMap<String, String>,
    /// `max_power` (engine.py:166): `max(int(op[3:]) for op in operator_tokens if op matches
    /// `^pow\d+` NOT followed by `_`)`, else 0. The factor ceiling for `cancel_terms`' factorization.
    pub max_power: i64,
    /// `max_fractional_power` (engine.py:167): `max(int(op[5:]) for op in operator_tokens if op
    /// matches `^pow1_\d+`)`, else 0. The factor ceiling for `convert_expression`'s fractional-power
    /// chain combining (dev_7-3: pow1_2..pow1_5 -> 5).
    pub max_fractional_power: i64,
    /// `operator_precedence_compat` (engine.py:157-159): `{name: spec.precedence if present else
    /// ENUMERATION-INDEX i}` in config order, THEN `['**'] = 3` and `['sqrt'] = 3` overlaid. Consumed
    /// by `prefix_to_infix` (paren decisions) and `infix_to_prefix` (shunting-yard precedence pops).
    /// f64 because `neg` declares the float precedence 2.5. The enum-index default never fires for
    /// dev_7-3 (all 38 ops declare a precedence) but is ported faithfully for config-robustness.
    pub operator_precedence_compat: FxHashMap<String, f64>,
    /// `operator_arity_compat` (engine.py:162-163): `deepcopy(operator_arity)` + `'**' -> 2`. The
    /// arity table `sort_operands` consults (it adds the Python-style `**` power token).
    pub operator_arity_compat: FxHashMap<String, u8>,
    /// `operator_aliases` (engine.py:144): `{alias: operator}` over every operator's `alias` list.
    /// `sort_operands` resolves an input token to its canonical operator through this map.
    pub operator_aliases: FxHashMap<String, String>,
    /// `operator_realizations` (engine.py:154): `{name: realization}` (e.g. `sin` ->
    /// `simplipy.operators.sin`, `+` -> `+`). Used by `operators_to_realizations`.
    pub operator_realizations: FxHashMap<String, String>,
    /// `realization_to_operator` (engine.py:155): the inverse, built in config order so a (here
    /// absent) realization collision would resolve last-wins exactly as Python's dict comprehension.
    pub realization_to_operator: FxHashMap<String, String>,
    // TODO(port): inverse_base / inverse_unary / inverse_binary, connection_classes,
    // operator_to_class, binary_connectable_operators, max_fractional_power.
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
        let operator_inverses = specs
            .iter()
            .filter_map(|(k, v)| v.inverse.clone().map(|inv| (k.clone(), inv)))
            .collect();
        let max_power = order.iter().filter_map(|t| pow_power(t)).max().unwrap_or(0);
        let max_fractional_power = order
            .iter()
            .filter_map(|t| pow1_power(t))
            .max()
            .unwrap_or(0);
        // operator_precedence_compat: declared precedence, else the config enumeration index, then
        // overlay '**' = 3 and 'sqrt' = 3 (engine.py:157-159).
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
        operator_arity_compat.insert("**".to_string(), 2); // engine.py:163
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
        Self {
            arity,
            commutative,
            operator_inverses,
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
        self.arity.get(token).copied()
    }

    #[inline]
    pub fn is_operator(&self, token: &str) -> bool {
        self.arity.contains_key(token)
    }

    /// `operator_inverses.get(token)` (engine.py:145). `None` if the operator declares no inverse.
    #[inline]
    pub fn operator_inverse(&self, token: &str) -> Option<&str> {
        self.operator_inverses.get(token).map(|s| s.as_str())
    }

    /// `operator in self.commutative_operators` (engine.py:152). dev_7-3: `{'+', '*'}`.
    #[inline]
    pub fn is_commutative(&self, token: &str) -> bool {
        self.commutative.iter().any(|c| c == token)
    }

    /// `operator_precedence_compat.get(token)` (engine.py:157-159). `None` if the token is unknown;
    /// call sites apply their own default (`.get(t, 0)` in `infix_to_prefix`, the `pow`/inf fallback
    /// in `prefix_to_infix`).
    #[inline]
    pub fn precedence_get(&self, token: &str) -> Option<f64> {
        self.operator_precedence_compat.get(token).copied()
    }

    /// `token in self.operator_tokens` (the 38 config operator names; engine.py:143). Equivalent to
    /// `arity.contains_key` since `arity` is built from exactly those keys (excludes the synthetic
    /// `**` that only `operator_arity_compat` carries). Used by `prefix_to_infix`'s operator test.
    #[inline]
    pub fn is_operator_token(&self, token: &str) -> bool {
        self.arity.contains_key(token)
    }

    /// The `sort_operands` operator test (engine.py:1660-1662): `token in operator_arity_compat or
    /// token in operator_aliases`. Returns the resolved `(canonical_operator, arity)`, else `None`
    /// (a leaf). `operator = operator_aliases.get(token, token)`; `arity = operator_arity_compat[operator]`.
    pub fn sort_resolve(&self, token: &str) -> Option<(String, usize)> {
        let in_compat = self.operator_arity_compat.contains_key(token);
        let in_alias = self.operator_aliases.contains_key(token);
        if !in_compat && !in_alias {
            return None;
        }
        let operator = self
            .operator_aliases
            .get(token)
            .cloned()
            .unwrap_or_else(|| token.to_string());
        let arity = *self.operator_arity_compat.get(&operator)? as usize;
        Some((operator, arity))
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
/// (`pow1_2..pow1_5`) that is exactly the digit run. Mirrors engine.py:167.
pub(crate) fn pow1_power(token: &str) -> Option<i64> {
    let rest = token.strip_prefix("pow1_")?;
    // `\d+`: require at least one leading digit, then `int(op[5:])` parses the whole tail.
    if !rest.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        return None;
    }
    rest.parse::<i64>().ok()
}
