//! The "drop-in engine" conversion surface: `prefix_to_infix`, `infix_to_prefix`,
//! `convert_expression`, `parse`. These let flash-ansr swap the whole `simplipy_engine` object for
//! the Rust port (it calls `is_valid` / `prefix_to_infix` / `infix_to_prefix` / `parse` on the engine
//! OBJECT), not just route `.simplify`.

use crate::operators::{pow1_power, pow_power, Operators};
use crate::utils::is_numeric_string;

/// How `prefix_to_infix` renders power operators (the `power` parameter).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Power {
    /// `'func'`: keep engine names (`pow(a, b)`, `pow2(x)`). The DEFAULT and dominant deployment mode.
    Func,
    /// `'**'`: Python-style exponentiation (`a ** b`, `x**2`, `x**(1/2)`).
    StarStar,
}

/// `op_associativity`: a LOCAL hardcoded map, NOT from config. Absent operators
/// default to `'left'`. Only `+,-,*,/` (left) and `**,pow` (right) appear.
fn associativity(op: &str) -> &'static str {
    match op {
        "**" | "pow" => "right",
        _ => "left", // +,-,*,/ and the default
    }
}

/// A render-stack element: `(rendered_str, precedence, root_operator)`.
type Item = (String, f64, Option<String>);

/// `prefix_to_infix`: renders a prefix token list to an infix
/// string with minimal parentheses. Returns `Err` where Python raises `ValueError` (malformed prefix:
/// too few / too many operands) -- the FFI maps that to a Python `ValueError`.
///
/// The contract-critical points:
///  * precedence is f64 (`neg`=2.5 sits strictly between `*`=2 and `pow`=3).
///  * THREE distinct unary comparison operators: `neg` strict `<`, `inv` `<=`, `pow{N}`-under-`**`
///    `<=`. Do not unify them.
///  * `inv` PUSHES `op_precedence['/']` (=2) as its node precedence, not its own 4.
///  * under `realization=True`, a realization containing `'.'` renders pure func-form and PREEMPTS the
///    neg/inv/pow special-cases (so only `+,-,*` are ever infix); checked BEFORE `power`.
///  * an equal-precedence right operand always keeps its parens (no flattening), coordinated with
///    the associativity-respecting `infix_to_prefix` so prefix<->infix round-trips.
///  * spacing is load-bearing: `a + b` (spaces) vs `x**2` / `-x` / `1/x` (no spaces).
pub fn prefix_to_infix(
    tokens: &[String],
    ops: &Operators,
    power: Power,
    realization: bool,
) -> Result<String, String> {
    if tokens.is_empty() {
        return Ok(String::new());
    }

    const INF: f64 = f64::INFINITY; // FUNC_PRECEDENCE = TERMINAL_PRECEDENCE
    let mut stack: Vec<Item> = Vec::with_capacity(tokens.len());

    for token in tokens.iter().rev() {
        // operator = realization_to_operator.get(token, token): an already-realized input token
        // (e.g. 'simplipy.operators.sin') maps back to canonical first.
        let operator: &str = ops
            .realization_to_operator
            .get(token)
            .map(|s| s.as_str())
            .unwrap_or(token);
        // canonical_operator = operator_aliases.get(operator, operator).
        let canonical: &str = ops
            .operator_aliases
            .get(operator)
            .map(|s| s.as_str())
            .unwrap_or(operator);

        // membership: 3-way OR.
        let is_op = ops.is_operator_token(canonical)
            || ops.operator_aliases.contains_key(operator)
            || ops.operator_arity_compat.contains_key(canonical);

        if !is_op {
            stack.push((token.clone(), INF, None)); // terminal
            continue;
        }

        // arity = operator_arity_compat.get(canonical, 1).
        let arity = ops
            .operator_arity_compat
            .get(canonical)
            .copied()
            .unwrap_or(1) as usize;
        if stack.len() < arity {
            // ValueError with the RESOLVED operator var (pre-alias).
            return Err(format!(
                "Invalid prefix expression: Not enough operands for operator '{operator}'"
            ));
        }
        // operands_data = [stack.pop() for _ in range(arity)] -> [0]=left, [1]=right.
        let operands_data: Vec<Item> = (0..arity).map(|_| stack.pop().unwrap()).collect();

        // write_operator: realization ? realization_map[canonical] : canonical.
        let write_operator: String = if realization {
            ops.operator_realizations
                .get(canonical)
                .cloned()
                .unwrap_or_else(|| canonical.to_string())
        } else {
            canonical.to_string()
        };

        // realization '.'-in-name OR arity>2 -> pure func-form, PREEMPTS all special-cases.
        if realization && (write_operator.contains('.') || arity > 2) {
            let joined = join_operands(&operands_data);
            stack.push((
                format!("{write_operator}({joined})"),
                INF,
                Some(canonical.to_string()),
            ));
            continue;
        }

        // current_precedence = op_precedence.get(canonical, op_precedence.get('pow', INF)).
        let mut current_precedence = ops
            .precedence_get(canonical)
            .unwrap_or_else(|| ops.precedence_get("pow").unwrap_or(INF));
        // current_assoc = op_associativity.get(canonical, 'left').
        let mut current_assoc = associativity(canonical);

        if arity == 2 {
            let (mut left_str, left_prec, _left_root) = operands_data[0].clone();
            let (mut right_str, right_prec, _right_root) = operands_data[1].clone();
            let mut write_operator = write_operator;

            // pow under power='func' -> func-form.
            if canonical == "pow" && power == Power::Func {
                stack.push((
                    format!("{write_operator}({left_str}, {right_str})"),
                    INF,
                    Some(canonical.to_string()),
                ));
                continue;
            }
            // pow under power='**' -> switch to infix '**', right-assoc.
            if canonical == "pow" && power == Power::StarStar {
                write_operator = "**".to_string();
                current_precedence = ops.precedence_get("**").unwrap_or(current_precedence);
                current_assoc = "right";
            }

            // left paren: left_prec < cur OR (== AND assoc right).
            if left_prec < current_precedence
                || (left_prec == current_precedence && current_assoc == "right")
            {
                left_str = format!("({left_str})");
            }
            // right paren: right_prec < cur OR (== AND assoc left). An equal-precedence right
            // operand always keeps its parens (no flattening) -- paired with the left-assoc parse
            // in `infix_to_prefix`, this preserves prefix<->infix round-trip identity.
            if right_prec < current_precedence
                || (right_prec == current_precedence && current_assoc == "left")
            {
                right_str = format!("({right_str})");
            }
            stack.push((
                format!("{left_str} {write_operator} {right_str}"),
                current_precedence,
                Some(canonical.to_string()),
            ));
            continue;
        }

        if arity == 1 {
            let (mut operand_str, operand_prec, _operand_root) = operands_data[0].clone();
            let is_pow_op = pow_power(canonical).is_some(); // r'pow\d+(?!_)'
            let is_frac_pow_op = pow1_power(canonical).is_some(); // r'pow1_\d+'

            if canonical == "neg" {
                // parens iff operand_prec STRICT-< current.
                if operand_prec < current_precedence {
                    operand_str = format!("({operand_str})");
                }
                stack.push((
                    format!("-{operand_str}"),
                    current_precedence,
                    Some(canonical.to_string()),
                ));
                continue;
            }
            if canonical == "inv" {
                // parens iff operand_prec <= current; PUSH op_precedence['/'] (=2), not its own.
                if operand_prec <= current_precedence {
                    operand_str = format!("({operand_str})");
                }
                let inv_precedence = ops.precedence_get("/").unwrap_or(current_precedence);
                stack.push((
                    format!("1/{operand_str}"),
                    inv_precedence,
                    Some(canonical.to_string()),
                ));
                continue;
            }
            if power == Power::StarStar && (is_pow_op || is_frac_pow_op) {
                // x**N / x**(1/N); operand parens iff operand_prec <= power_prec.
                let power_precedence = ops.precedence_get("**").unwrap_or(current_precedence);
                if operand_prec <= power_precedence {
                    operand_str = format!("({operand_str})");
                }
                let rendered = if is_pow_op {
                    let exponent = pow_power(canonical).unwrap(); // int(op[3:])
                    format!("{operand_str}**{exponent}")
                } else {
                    let denominator = pow1_power(canonical).unwrap(); // int(op[5:])
                    format!("{operand_str}**(1/{denominator})")
                };
                stack.push((rendered, power_precedence, Some(canonical.to_string())));
                continue;
            }
            // func fallback.
            stack.push((
                format!("{write_operator}({operand_str})"),
                INF,
                Some(canonical.to_string()),
            ));
            continue;
        }

        // nullary / arity>2 fallback (DEAD for the shipped asset; max arity 2, no nullary ops).
        let joined = join_operands(&operands_data);
        stack.push((
            format!("{write_operator}({joined})"),
            INF,
            Some(canonical.to_string()),
        ));
    }

    if stack.len() != 1 {
        // Too many operands. The Python message embeds a list-repr of the leftover rendered
        // parts (stack order, reversed vs input); we surface failure-parity, not the exact repr.
        let parts: Vec<String> = stack.iter().map(|(s, _, _)| format!("'{s}'")).collect();
        return Err(format!(
            "Malformed prefix expression: too many operands remain after processing. Stack: [{}]",
            parts.join(", ")
        ));
    }
    Ok(stack.into_iter().next().unwrap().0)
}

/// `', '.join(op_str for op_str, _, _ in operands_data)`: join the rendered
/// strings of the popped operands in pop order ([0]=left, ..).
fn join_operands(operands_data: &[Item]) -> String {
    operands_data
        .iter()
        .map(|(s, _, _)| s.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

// ---- infix_to_prefix ----------------------------------------------------------------------------

/// Tokenize the (space-stripped) infix string, per the Python regex
/// `float_special|<constant>|number|[A-Za-z_][\w.]*|\*\*|[-+*/^()]` under `re.findall` semantics:
/// scan left to right, at each position take the FIRST alternative (in pattern order) that matches,
/// and SILENTLY DROP any char that matches no alternative. Numbers/identifiers are emitted as
/// verbatim source substrings. (`\w` is treated as ASCII `[A-Za-z0-9_]`; the deployment corpus is
/// ASCII -- a non-ASCII identifier is the documented out-of-domain boundary.)
fn tokenize_infix(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().filter(|&c| c != ' ').collect(); // `.replace(' ', '')`
    let n = chars.len();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < n {
        // The numeric folder's inf/nan tokens stay ATOMIC -- mirrors the Python `float_special`
        // alternation -- else they split on the '(' / '"'. Leads the scan; the token is then
        // classified as a leaf by `is_ident_start` in `infix_to_prefix`.
        if let Some(j) = match_float_special(&chars, i) {
            tokens.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }
        if let Some(j) = match_constant(&chars, i) {
            tokens.push(chars[i..j].iter().collect());
            i = j;
        } else if let Some(j) = match_number(&chars, i) {
            tokens.push(chars[i..j].iter().collect());
            i = j;
        } else if let Some(j) = match_ident(&chars, i) {
            tokens.push(chars[i..j].iter().collect());
            i = j;
        } else if i + 1 < n && chars[i] == '*' && chars[i + 1] == '*' {
            tokens.push("**".to_string());
            i += 2;
        } else if matches!(chars[i], '-' | '+' | '*' | '/' | '^' | '(' | ')') {
            tokens.push(chars[i].to_string());
            i += 1;
        } else {
            i += 1; // unmatched -> drop (no token, no error)
        }
    }
    tokens
}

fn match_constant(s: &[char], i: usize) -> Option<usize> {
    const C: &[char] = &['<', 'c', 'o', 'n', 's', 't', 'a', 'n', 't', '>'];
    if s.len() - i >= C.len() && s[i..i + C.len()] == *C {
        Some(i + C.len())
    } else {
        None
    }
}

/// The numeric folder's inf/nan result tokens, kept ATOMIC (mirrors the Python `float_special`
/// alternation `float\("(?:-?inf|nan)"\)`): `float("inf")` / `float("-inf")` / `float("nan")`.
/// Returns the end index of the literal at `i`, or `None`.
fn match_float_special(s: &[char], i: usize) -> Option<usize> {
    for lit in ["float(\"-inf\")", "float(\"inf\")", "float(\"nan\")"] {
        let l: Vec<char> = lit.chars().collect();
        if s.len() - i >= l.len() && s[i..i + l.len()] == l[..] {
            return Some(i + l.len());
        }
    }
    None
}

/// `(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?` matched at `i`; returns the end index, or `None` if no
/// number starts here. (The `e`/`E` exponent is optional and only consumed if a digit follows.)
fn match_number(s: &[char], i: usize) -> Option<usize> {
    let n = s.len();
    let mut j = i;
    if j < n && s[j].is_ascii_digit() {
        while j < n && s[j].is_ascii_digit() {
            j += 1;
        }
        if j < n && s[j] == '.' {
            j += 1;
            while j < n && s[j].is_ascii_digit() {
                j += 1;
            }
        }
    } else if j < n && s[j] == '.' {
        // `\.\d+` needs at least one digit after the dot
        if !(j + 1 < n && s[j + 1].is_ascii_digit()) {
            return None;
        }
        j += 1;
        while j < n && s[j].is_ascii_digit() {
            j += 1;
        }
    } else {
        return None;
    }
    // optional `[eE][+-]?\d+`
    if j < n && (s[j] == 'e' || s[j] == 'E') {
        let mut k = j + 1;
        if k < n && (s[k] == '+' || s[k] == '-') {
            k += 1;
        }
        if k < n && s[k].is_ascii_digit() {
            while k < n && s[k].is_ascii_digit() {
                k += 1;
            }
            j = k;
        }
    }
    Some(j)
}

/// `[A-Za-z_][\w.]*` matched at `i` (`\w` as ASCII; '.' allowed in the tail). `None` if no identifier
/// starts here.
fn match_ident(s: &[char], i: usize) -> Option<usize> {
    let n = s.len();
    if !(i < n && (s[i].is_ascii_alphabetic() || s[i] == '_')) {
        return None;
    }
    let mut j = i + 1;
    while j < n && (s[j].is_ascii_alphanumeric() || s[j] == '_' || s[j] == '.') {
        j += 1;
    }
    Some(j)
}

/// `re.fullmatch(number_pattern, token)`: does the WHOLE token parse as a number?
fn is_number_fullmatch(token: &str) -> bool {
    let chars: Vec<char> = token.chars().collect();
    !chars.is_empty() && match_number(&chars, 0) == Some(chars.len())
}

/// `re.match(r'[A-Za-z_][\w.]*', token)` (unanchored): the token STARTS with an
/// identifier char. (Since the classifier only ever sees tokenizer outputs, "starts with" suffices.)
fn is_ident_start(token: &str) -> bool {
    token
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
}

/// `infix_to_prefix`: a RIGHT-to-LEFT shunting-yard. Never
/// raises (degenerate/malformed inputs produce structurally-degenerate prefix lists, matching Python).
pub fn infix_to_prefix(infix_expression: &str, ops: &Operators) -> Vec<String> {
    let mut tokens = tokenize_infix(infix_expression);
    tokens.reverse(); // right-to-left parse

    let mut stack: Vec<String> = Vec::new();
    let mut prefix_expr: Vec<String> = Vec::new();
    let prec = |t: &str| ops.precedence_get(t).unwrap_or(0.0); // `.get(t, 0)`

    let mut i = 0;
    while i < tokens.len() {
        // current token, with '^' normalized to '**'.
        let mut token = tokens[i].clone();
        if token == "^" {
            token = "**".to_string();
        }

        if is_number_fullmatch(&token) {
            prefix_expr.push(token);
        } else if is_ident_start(&token) || token == "<constant>" {
            prefix_expr.push(token);
        } else if token == ")" {
            stack.push(token);
        } else if token == "(" {
            while stack.last().is_some_and(|t| t != ")") {
                prefix_expr.push(stack.pop().unwrap());
            }
            if stack.last().is_some_and(|t| t == ")") {
                stack.pop();
            }
        } else {
            // operator. Unary-minus detection: on the REVERSED stream, tokens[i+1] is the
            // original LEFT neighbor; membership is the FULL precedence_compat keyset.
            let next_raw = tokens.get(i + 1).map(|s| s.as_str());
            // '^' is normalized to '**' for the lookahead too, so `x ^ -y` parses the '-' as
            // unary exactly like `x ** -y`.
            let next_norm = match next_raw {
                Some("^") => Some("**"),
                other => other,
            };
            if token == "-"
                && (next_raw.is_none()
                    || next_norm == Some("(")
                    || next_norm
                        .map(|t| ops.precedence_get(t).is_some())
                        .unwrap_or(false))
            {
                token = "neg".to_string();
            }
            // `token != ')'` is always true here. The ELSE block is provably a plain
            // push (`stack.insert(-1)` only runs on an empty stack, where it == push).
            // The pop rule respects associativity -- pop on strict `>` for left-assoc, `>=` only
            // for right-assoc ('**'/'pow'). Coordinated with the render half (no equal-precedence
            // right-operand flattening) to preserve prefix<->infix round-trip identity.
            if stack.last().is_some_and(|t| t != ")") {
                let cur = prec(&token);
                let right_assoc = token == "**" || token == "pow";
                while let Some(top) = stack.last() {
                    if top == ")" {
                        break;
                    }
                    let tp = prec(top);
                    let pop = tp > cur || (tp == cur && right_assoc);
                    if pop {
                        prefix_expr.push(stack.pop().unwrap());
                    } else {
                        break;
                    }
                }
                stack.push(token);
            } else {
                stack.push(token);
            }
        }
        i += 1;
    }

    while let Some(op) = stack.pop() {
        prefix_expr.push(op);
    }
    prefix_expr.reverse(); // `[::-1]`
    prefix_expr
}

// ---- convert_expression -------------------------------------------------------------------------

/// The nested-list intermediate representation `convert_expression` builds: an arbitrarily-nested
/// list of strings, exactly as Python (`[token]` leaves, `[op, [children]]` nodes, plus the quirky
/// `[base]` extra-nesting the float branch produces). `flatten` linearizes it to a prefix token list.
#[derive(Debug, Clone)]
enum Ir {
    S(String),
    L(Vec<Ir>),
}

/// `flatten_nested_list(list_of_items)[::-1]`:
/// a LIFO reverse-DFS over the items, then reversed -> a prefix token list. (Nesting depth is
/// irrelevant: any list is linearized, so the `[base]` quirk flattens away harmlessly.)
fn flatten_list(items: &[Ir]) -> Vec<String> {
    let mut flat = Vec::new();
    let mut work: Vec<&Ir> = items.iter().collect();
    while let Some(cur) = work.pop() {
        match cur {
            Ir::L(v) => work.extend(v.iter()),
            Ir::S(s) => flat.push(s.clone()),
        }
    }
    flat.reverse();
    flat
}

/// `node[0]` as a string, if it is one (Python `isinstance(x[0], str)`). `None` if the node is a
/// bare string or its first element is itself a list.
fn first_str(ir: &Ir) -> Option<&str> {
    match ir {
        Ir::L(v) => match v.first() {
            Some(Ir::S(s)) => Some(s.as_str()),
            _ => None,
        },
        Ir::S(_) => None,
    }
}

/// Replace `node[0]` (a string leaf) in place: `stack[-1][0] = new`.
fn set_first(ir: &mut Ir, new: String) {
    if let Ir::L(v) = ir {
        if let Some(Ir::S(s)) = v.first_mut() {
            *s = new;
        }
    }
}

/// `re.match(r'-?\d+$', s)`: optional leading `-`, then >=1 digits, whole string.
fn is_int_string(s: &str) -> bool {
    let t = s.strip_prefix('-').unwrap_or(s);
    !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit())
}

/// `re.match(r'pow1_\d+', s)`: starts with `pow1_` + >=1 digit.
fn matches_frac_pow(s: &str) -> bool {
    s.strip_prefix("pow1_")
        .is_some_and(|r| r.bytes().next().is_some_and(|b| b.is_ascii_digit()))
}

/// `int(re.match(r'pow(\d+)', op).group(1))`: the leading digit-run right after `pow`.
fn int_chain_exp(op: &str) -> Option<i128> {
    let rest = op.strip_prefix("pow")?;
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    // i128 (not i64): a `pow<N>` exponent can exceed i64 (e.g. `x ** 2^63`); Python's int() is
    // arbitrary-precision, so parsing as i64 would drop a real exponent (-> wrong decomposition).
    rest[..end].parse::<i128>().ok()
}

/// `Fraction(x).as_integer_ratio()` reduced (`Fraction(abs(float(s)))`): the EXACT
/// dyadic ratio of the f64 (NOT a decimal parse of the source string). `None` if the exact ratio
/// exceeds the i128 domain (pathological subnormals / huge magnitudes -- documented out-of-domain;
/// they never occur on the deployment distribution, which doesn't reach this branch at all).
fn exact_ratio(x: f64) -> Option<(i128, i128)> {
    if x == 0.0 {
        return Some((0, 1));
    }
    if !x.is_finite() {
        return None;
    }
    let bits = x.to_bits();
    let exp_field = ((bits >> 52) & 0x7ff) as i64;
    let mant = (bits & 0x000f_ffff_ffff_ffff) as i128;
    let (num0, e) = if exp_field == 0 {
        (mant, -1074i64) // subnormal: value = mant * 2^-1074
    } else {
        (mant + (1i128 << 52), exp_field - 1075) // normal: (1.mant) * 2^(exp-1023-52)
    };
    if e >= 0 {
        if e >= 127 {
            return None;
        }
        Some((num0 << e, 1))
    } else {
        let shift = (-e) as u32;
        if shift >= 127 {
            return None;
        }
        // reduce by the common power of two
        let tz = num0.trailing_zeros().min(shift);
        Some((num0 >> tz, 1i128 << (shift - tz)))
    }
}

/// `Fraction(num, den).limit_denominator(1_000_000)` (CPython Lib/fractions.py). Returns the closest
/// fraction with denominator <= 1e6, in lowest terms.
fn limit_denominator(num: i128, den: i128) -> (i128, i128) {
    const M: i128 = 1_000_000;
    if den <= M {
        return reduce(num, den);
    }
    let (mut p0, mut q0, mut p1, mut q1) = (0i128, 1i128, 1i128, 0i128);
    let (mut n, mut d) = (num, den);
    loop {
        let a = n / d;
        let q2 = q0 + a * q1;
        if q2 > M {
            break;
        }
        (p0, q0, p1, q1) = (p1, q1, p0 + a * p1, q2);
        (n, d) = (d, n - a * d);
    }
    let k = (M - q0) / q1;
    // bound2 = p1/q1 ; bound1 = (p0+k*p1)/(q0+k*q1). Pick bound2 iff |bound2-self| <= |bound1-self|,
    // compared without floats: |p1*den - num*q1|*(q0+k*q1)  <=  |(p0+k*p1)*den - num*(q0+k*q1)|*q1.
    let qa = q0 + k * q1;
    let pa = p0 + k * p1;
    let diff2 = (p1 * den - num * q1).abs();
    let diff1 = (pa * den - num * qa).abs();
    if diff2 * qa <= diff1 * q1 {
        reduce(p1, q1)
    } else {
        reduce(pa, qa)
    }
}

fn reduce(num: i128, den: i128) -> (i128, i128) {
    let g = gcd_i128(num.abs(), den.abs());
    if g == 0 {
        (num, den)
    } else {
        (num / g, den / g)
    }
}

fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `Fraction(abs(float(s))).limit_denominator()` reduced. `None` only on the
/// documented out-of-i128-domain pathological inputs (never on the deployment distribution).
fn fraction_limit_denominator(x: f64) -> Option<(i128, i128)> {
    let (num, den) = exact_ratio(x)?;
    Some(limit_denominator(num, den))
}

/// `['pow', [base, exponent]]` -- the KEEP fallback (gate-fail / non-numeric exponent).
fn pow_keep(base: Ir, exponent: Ir) -> Ir {
    Ir::L(vec![Ir::S("pow".into()), Ir::L(vec![base, exponent])])
}

/// `**` handling: integer / float / integer-fraction exponent. `Err` mirrors the
/// dead len==2 float-division branch's `int()` `ValueError` (failure-parity).
fn handle_pow(base: Ir, exponent: Ir, ops: &Operators) -> Result<Ir, String> {
    let ev = match &exponent {
        Ir::L(v) => v,
        Ir::S(_) => return Ok(pow_keep(base, exponent)),
    };
    if ev.len() == 1 {
        let tok = match &ev[0] {
            Ir::S(s) => s.clone(),
            _ => return Ok(pow_keep(base, exponent)),
        };
        if is_int_string(&tok) {
            let v: i128 = tok.parse().map_err(|_| "int overflow".to_string())?;
            if v == 0 {
                return Ok(Ir::L(vec![Ir::S("1".into())])); // x**0 -> 1 (not 'pow0')
            }
            // Decomposability gate (phantom powN): only exponents factorizable into the
            // unary pow2..pow{max_power} vocabulary may become powN tokens; non-smooth
            // exponents (7, 11, 14, ...) previously emitted operators like `pow7` with no
            // realization, corrupting arity downstream. Keep binary pow instead.
            if crate::utils::factorize_to_at_most(v.unsigned_abs() as i128, ops.max_power, 1000)
                .is_err()
            {
                return Ok(pow_keep(base, exponent));
            }
            let pow_op = format!("pow{}", v.unsigned_abs());
            if v < 0 {
                // ['inv', [[pow_op, [base]]]]
                Ok(Ir::L(vec![
                    Ir::S("inv".into()),
                    Ir::L(vec![Ir::L(vec![Ir::S(pow_op), Ir::L(vec![base])])]),
                ]))
            } else {
                Ok(Ir::L(vec![Ir::S(pow_op), Ir::L(vec![base])])) // [pow_op, [base]]
            }
        } else if is_numeric_string(&tok) {
            let fv: f64 = match tok.parse() {
                Ok(v) => v,
                Err(_) => return Err("could not convert to float".into()), // is_numeric_string but float() raises
            };
            match fraction_limit_denominator(fv.abs()) {
                Some((num, den)) if num <= 5 && den <= 5 => {
                    if num == 0 {
                        return Ok(Ir::L(vec![Ir::S("1".into())])); // x**0.0 -> 1
                    }
                    let mut new_expr = Ir::L(vec![base]); // [base]
                    if num != 1 {
                        new_expr = Ir::L(vec![Ir::S(format!("pow{num}")), new_expr]);
                    }
                    if den != 1 {
                        new_expr = Ir::L(vec![Ir::S(format!("pow1_{den}")), new_expr]);
                    }
                    if fv < 0.0 {
                        new_expr = Ir::L(vec![Ir::S("inv".into()), new_expr]);
                    }
                    Ok(new_expr)
                }
                _ => Ok(pow_keep(base, exponent)), // gate-fail / out-of-domain -> KEEP
            }
        } else {
            Ok(pow_keep(base, exponent)) // non-numeric exponent -> KEEP
        }
    } else if ev.len() == 2 {
        // exponent[0][0] == '/' and both operands numeric strings.
        let op0_is_div = matches!(&ev[0], Ir::S(s) if s.starts_with('/'));
        let (num_tok, den_tok) = match &ev[1] {
            Ir::L(operands) if operands.len() == 2 => {
                (first_str(&operands[0]), first_str(&operands[1]))
            }
            _ => (None, None),
        };
        match (op0_is_div, num_tok, den_tok) {
            (true, Some(nt), Some(dt)) if is_numeric_string(nt) && is_numeric_string(dt) => {
                if is_int_string(nt) && is_int_string(dt) {
                    let numerator: i128 = nt.parse().map_err(|_| "int overflow".to_string())?;
                    let denominator: i128 = dt.parse().map_err(|_| "int overflow".to_string())?;
                    if numerator == 0 {
                        return Ok(Ir::L(vec![Ir::S("1".into())])); // x**(0/N) -> 1
                    }
                    // Same decomposability gate for the power (pow{num}) and root (pow1_{den}).
                    if crate::utils::factorize_to_at_most(
                        numerator.unsigned_abs() as i128,
                        ops.max_power,
                        1000,
                    )
                    .is_err()
                        || crate::utils::factorize_to_at_most(
                            denominator.unsigned_abs() as i128,
                            ops.max_fractional_power,
                            1000,
                        )
                        .is_err()
                    {
                        return Ok(pow_keep(base, exponent));
                    }
                    let num_power = format!("pow{}", numerator.unsigned_abs());
                    let den_power = format!("pow1_{}", denominator.unsigned_abs());
                    // [den_power, [[num_power, [base]]]]
                    let inner = Ir::L(vec![Ir::S(num_power), Ir::L(vec![base])]);
                    let dnode = Ir::L(vec![Ir::S(den_power), Ir::L(vec![inner])]);
                    if numerator * denominator < 0 {
                        Ok(Ir::L(vec![Ir::S("inv".into()), Ir::L(vec![dnode])]))
                    } else {
                        Ok(dnode)
                    }
                } else {
                    // dead float-division branch: int('2.0') raises BEFORE limit_denominator.
                    Err("invalid literal for int()".into())
                }
            }
            _ => Ok(pow_keep(base, exponent)), // else -> KEEP
        }
    } else {
        Ok(pow_keep(base, exponent))
    }
}

/// `convert_expression`: normalize a prefix expression into the
/// engine's internal form (`**` -> `pow{N}`, chained powers combined, unary negation folded into
/// numeric literals). `Err` mirrors a Python raise (the dead float-division `int()` ValueError).
pub fn convert_expression(prefix_expr: &[String], ops: &Operators) -> Result<Vec<String>, String> {
    // ---- PASS 1: build the nested-list IR (right-to-left) ----
    let mut stack: Vec<Ir> = Vec::new();
    for token in prefix_expr.iter().rev() {
        let is_op = ops.operator_arity_compat.contains_key(token)
            || ops.operator_aliases.contains_key(token)
            || pow_power(token).is_some()
            || pow1_power(token).is_some();
        if !is_op {
            stack.push(Ir::L(vec![Ir::S(token.clone())])); // [token]
            continue;
        }
        let operator = ops
            .operator_aliases
            .get(token)
            .map(|s| s.as_str())
            .unwrap_or(token);
        // arity: graceful default 1 for a raw unconfigured `powN` token, like pass-2's `.get`.
        let arity = ops
            .operator_arity_compat
            .get(operator)
            .map(|a| *a as usize)
            .unwrap_or(1);
        if operator == "neg" {
            match stack.last().and_then(first_str) {
                Some(s) if is_numeric_string(s) => {
                    let s = s.to_string();
                    let mut top = stack.pop().ok_or("neg: empty stack")?;
                    // Toggle ONE leading '-' (strip if already negative, else prepend).
                    let new = if s.starts_with('-') {
                        s[1..].to_string()
                    } else {
                        format!("-{s}")
                    };
                    set_first(&mut top, new);
                    stack.push(top);
                }
                _ => {
                    let operands = pop_operands(&mut stack, arity)?;
                    stack.push(Ir::L(vec![Ir::S("neg".into()), Ir::L(operands)]));
                }
            }
        } else if operator == "**" {
            let base = stack.pop().ok_or("**: missing base")?;
            let exponent = stack.pop().ok_or("**: missing exponent")?;
            stack.push(handle_pow(base, exponent, ops)?);
        } else {
            let operands = pop_operands(&mut stack, arity)?;
            stack.push(Ir::L(vec![Ir::S(operator.into()), Ir::L(operands)]));
        }
    }
    let need_to_convert = flatten_list(&stack); // flatten_nested_list(stack)[::-1]

    // ---- PASS 2: combine pow-chains (right-to-left) ----
    let mut stack2: Vec<Ir> = Vec::new();
    for token in need_to_convert.iter().rev() {
        let is_pow_token = pow_power(token).is_some() || pow1_power(token).is_some(); // r'pow\d+(?!_)' | r'pow1_\d+'
        if is_pow_token {
            let operator = ops
                .operator_aliases
                .get(token)
                .map(|s| s.as_str())
                .unwrap_or(token);
            let arity = ops
                .operator_arity_compat
                .get(operator)
                .map(|a| *a as usize)
                .unwrap_or(1);
            let operands = take_reversed_tail(&stack2, arity)?;
            let is_frac = matches_frac_pow(operator); // pow1_ pattern first; else int

            // chain detection: descend operands[0] while it is a 2-elem pow node of the same family.
            let mut operator_chain: Vec<String> = vec![operator.to_string()];
            let mut current_operand = operands[0].clone();
            loop {
                let next = match &current_operand {
                    Ir::L(v) if v.len() == 2 => match &v[0] {
                        // The integer chain uses `pow_power` (the `(?!_)` lookahead), so a child
                        // `pow1_M` is NOT absorbed into an integer-power chain.
                        Ir::S(op0)
                            if (is_frac && matches_frac_pow(op0))
                                || (!is_frac && pow_power(op0).is_some()) =>
                        {
                            Some((op0.clone(), v[1].clone()))
                        }
                        _ => None,
                    },
                    _ => None,
                };
                match next {
                    Some((op0, descend)) => {
                        operator_chain.push(op0);
                        current_operand = descend;
                    }
                    None => break,
                }
            }

            // p = product of the chain's exponents.
            // i128: Python's `prod` is arbitrary-precision; i128 pushes the divergence boundary past
            // any reachable exponent (the frac family `pow1_M` stays tiny).
            let mut p: i128 = 1;
            for op in &operator_chain {
                let e = if is_frac {
                    pow1_power(op).map(|x| x as i128)
                } else {
                    int_chain_exp(op)
                }
                .unwrap_or(0);
                p = p.saturating_mul(e);
            }
            let max_p = if is_frac {
                ops.max_fractional_power
            } else {
                ops.max_power
            };
            let base_str = if is_frac { "pow1_" } else { "pow" };

            let new_chain = match crate::utils::factorize_to_at_most(p, max_p, 1000) {
                Ok(factors) => {
                    let new_operators: Vec<String> =
                        factors.iter().map(|f| format!("{base_str}{f}")).collect();
                    build_chain(&new_operators, current_operand)
                }
                Err(()) => build_chain(&operator_chain, current_operand), // VE fallback: original chain
            };
            for _ in 0..arity {
                stack2.pop();
            }
            stack2.push(new_chain);
        } else if ops.operator_arity_compat.contains_key(token)
            || ops.operator_aliases.contains_key(token)
        {
            let operator = ops
                .operator_aliases
                .get(token)
                .map(|s| s.as_str())
                .unwrap_or(token);
            let arity = *ops.operator_arity_compat.get(operator).unwrap() as usize;
            let operands = take_reversed_tail(&stack2, arity)?;
            for _ in 0..arity {
                stack2.pop();
            }
            stack2.push(Ir::L(vec![Ir::S(operator.into()), Ir::L(operands)]));
        } else {
            stack2.push(Ir::L(vec![Ir::S(token.clone())])); // [token]
        }
    }
    Ok(flatten_list(&stack2))
}

/// `[stack.pop() for _ in range(arity)]` (pop order, [0] = top).
fn pop_operands(stack: &mut Vec<Ir>, arity: usize) -> Result<Vec<Ir>, String> {
    let mut out = Vec::with_capacity(arity);
    for _ in 0..arity {
        out.push(stack.pop().ok_or("missing operand")?);
    }
    Ok(out)
}

/// `list(reversed(stack[-arity:]))`: the last `arity` items, reversed (NOT popped).
fn take_reversed_tail(stack: &[Ir], arity: usize) -> Result<Vec<Ir>, String> {
    if stack.len() < arity {
        return Err("pass-2: not enough operands".into());
    }
    Ok(stack[stack.len() - arity..].iter().rev().cloned().collect())
}

/// Build the nested pow chain from a list of operator names around `current_operand`:
/// `[ops[-1], [current]]` innermost, wrapped outward by `ops[-2::-1]`. Empty
/// ops -> `current_operand` itself (the pow1-vanishes case).
fn build_chain(ops_list: &[String], current_operand: Ir) -> Ir {
    if ops_list.is_empty() {
        return current_operand;
    }
    let mut nc = Ir::L(vec![
        Ir::S(ops_list[ops_list.len() - 1].clone()),
        Ir::L(vec![current_operand]),
    ]);
    for op in ops_list[..ops_list.len() - 1].iter().rev() {
        nc = Ir::L(vec![Ir::S(op.clone()), Ir::L(vec![nc])]);
    }
    nc
}

// ---- parse --------------------------------------------------------------------------------------

/// `parse`: `infix_to_prefix` -> (if `convert`) `convert_expression`
/// -> (if `mask_numbers`) `numbers_to_constant` -> ALWAYS `remove_pow1`. The high-level entry that
/// closes `simplify(str)` and the flash-ansr canonicalization path. `Err` propagates a
/// `convert_expression` raise.
pub fn parse(
    infix_expression: &str,
    ops: &Operators,
    convert: bool,
    mask_numbers: bool,
) -> Result<Vec<String>, String> {
    let parsed = infix_to_prefix(infix_expression, ops);
    let parsed = if convert {
        convert_expression(&parsed, ops)?
    } else {
        parsed
    };
    let parsed = if mask_numbers {
        crate::utils::numbers_to_constant(&parsed)
    } else {
        parsed
    };
    Ok(crate::utils::remove_pow1(&parsed))
}

#[cfg(test)]
mod tests {
    use super::Power;
    use crate::Engine;

    fn engine() -> Option<Engine> {
        crate::test_engine()
    }

    fn p2i(e: &Engine, toks: &[&str], power: Power, realization: bool) -> Result<String, String> {
        let t: Vec<String> = toks.iter().map(|s| s.to_string()).collect();
        e.prefix_to_infix(&t, power, realization)
    }

    /// Branch-discriminating cases pinning the render contract in CI.
    #[test]
    fn prefix_to_infix_traps() {
        let Some(e) = engine() else { return };
        let f = Power::Func;
        let s = Power::StarStar;
        // neg strict-< (equal-prec NO parens) vs inv <= ; realization preempt; paren-keeping.
        assert_eq!(p2i(&e, &["neg", "neg", "x1"], f, false).unwrap(), "--x1");
        assert_eq!(
            p2i(&e, &["inv", "inv", "x1"], f, false).unwrap(),
            "1/(1/x1)"
        );
        assert_eq!(
            p2i(&e, &["*", "x2", "inv", "x1"], f, false).unwrap(),
            "x2 * (1/x1)"
        );
        assert_eq!(
            p2i(&e, &["-", "x1", "+", "x2", "x3"], f, false).unwrap(),
            "x1 - (x2 + x3)"
        );
        assert_eq!(
            p2i(&e, &["/", "/", "x1", "x2", "x3"], f, false).unwrap(),
            "x1 / x2 / x3"
        );
        // An equal-precedence RIGHT operand keeps its parens (no flattening) -- the render half
        // of the round-trip identity.
        assert_eq!(
            p2i(&e, &["*", "/", "x1", "x2", "/", "x3", "x4"], f, false).unwrap(),
            "x1 / x2 * (x3 / x4)"
        );
        assert_eq!(
            p2i(&e, &["+", "a", "+", "b", "c"], f, false).unwrap(),
            "a + (b + c)"
        );
        // pow rendering + spacing.
        assert_eq!(
            p2i(&e, &["pow", "x1", "x2"], f, false).unwrap(),
            "pow(x1, x2)"
        );
        assert_eq!(p2i(&e, &["**", "x1", "x2"], f, false).unwrap(), "x1 ** x2");
        assert_eq!(
            p2i(&e, &["pow", "pow", "x1", "x2", "x3"], s, false).unwrap(),
            "(x1 ** x2) ** x3"
        );
        assert_eq!(
            p2i(&e, &["*", "pow", "x1", "x2", "pow2", "x3"], s, false).unwrap(),
            "x1 ** x2 * x3**2"
        );
        assert_eq!(
            p2i(&e, &["pow2", "neg", "x1"], s, false).unwrap(),
            "(-x1)**2"
        );
        assert_eq!(p2i(&e, &["sqrt", "x1"], s, false).unwrap(), "x1**(1/2)");
        // realization=True: only +,-,* infix; neg/div func-form; power ignored.
        assert_eq!(
            p2i(&e, &["neg", "x1"], f, true).unwrap(),
            "simplipy.operators.neg(x1)"
        );
        assert_eq!(
            p2i(&e, &["pow", "x1", "x2"], s, true).unwrap(),
            "simplipy.operators.pow(x1, x2)"
        );
        assert_eq!(
            p2i(&e, &["+", "*", "x1", "x2", "sin", "x3"], f, true).unwrap(),
            "x1 * x2 + simplipy.operators.sin(x3)"
        );
        // empty -> "" ; malformed -> Err (failure-parity).
        assert_eq!(p2i(&e, &[], f, false).unwrap(), "");
        assert!(p2i(&e, &["+", "x1"], f, false).is_err());
        assert!(p2i(&e, &["+", "x1", "x2", "x3"], f, false).is_err());
    }

    fn i2p(e: &Engine, s: &str) -> Vec<String> {
        e.infix_to_prefix(s)
    }

    /// infix_to_prefix traps pinning the parse contract in CI.
    #[test]
    fn infix_to_prefix_traps() {
        let Some(e) = engine() else { return };
        let v = |xs: &[&str]| -> Vec<String> { xs.iter().map(|s| s.to_string()).collect() };
        // leading unary (only reachable insert(-1)==push path); standalone '-'.
        assert_eq!(i2p(&e, "-x1"), v(&["neg", "x1"]));
        assert_eq!(i2p(&e, "-"), v(&["neg"]));
        // '^' normalizes to '**' in the unary-minus lookahead too.
        assert_eq!(i2p(&e, "x1 ** - x2"), v(&["neg", "**", "x1", "x2"]));
        assert_eq!(i2p(&e, "x1 ^ - x2"), v(&["neg", "**", "x1", "x2"]));
        // function-name left neighbor -> unary; neg float precedence both ways.
        assert_eq!(i2p(&e, "sin - x1"), v(&["neg", "sin", "x1"]));
        assert_eq!(i2p(&e, "-x1 ** 2"), v(&["neg", "**", "x1", "2"]));
        assert_eq!(i2p(&e, "-x1 * x2"), v(&["*", "neg", "x1", "x2"]));
        // associativity-respecting pop: left-assoc chains parse left-assoc; '**' right-assoc.
        assert_eq!(
            i2p(&e, "a - b - c - d"),
            v(&["-", "-", "-", "a", "b", "c", "d"])
        );
        assert_eq!(
            i2p(&e, "1/2 * m * v ** 2"),
            v(&["*", "*", "/", "1", "2", "m", "**", "v", "2"])
        );
        assert_eq!(i2p(&e, "a ** b ** c"), v(&["**", "a", "**", "b", "c"]));
        assert_eq!(i2p(&e, "1/x1"), v(&["/", "1", "x1"]));
        // tokenizer: '**' before '*', drop unmatched, empty parens.
        assert_eq!(i2p(&e, "x1***x2"), v(&["*", "**", "x1", "x2"]));
        assert_eq!(i2p(&e, "x1 $ x2"), v(&["x1", "x2"]));
        assert_eq!(i2p(&e, "()"), Vec::<String>::new());
        assert_eq!(i2p(&e, ""), Vec::<String>::new());
        // scientific notation single token.
        assert_eq!(i2p(&e, "1.5e-2 * x1"), v(&["*", "1.5e-2", "x1"]));
        assert_eq!(i2p(&e, "<constant> * x1"), v(&["*", "<constant>", "x1"]));
        // round-trip identity (parse half; paired with the paren-keeping render).
        let pre = v(&["*", "a", "*", "b", "c"]);
        let inf = e.prefix_to_infix(&pre, Power::StarStar, false).unwrap();
        assert_eq!(e.parse(&inf, false, false).unwrap(), pre);
    }

    fn conv(e: &Engine, toks: &[&str]) -> Result<Vec<String>, String> {
        let t: Vec<String> = toks.iter().map(|s| s.to_string()).collect();
        e.convert_expression(&t)
    }

    /// convert_expression traps pinning the normalization contract in CI.
    #[test]
    fn convert_expression_traps() {
        let Some(e) = engine() else { return };
        let v = |xs: &[&str]| -> Vec<String> { xs.iter().map(|s| s.to_string()).collect() };
        // integer exponent (live) + negative + pow1-vanish.
        assert_eq!(conv(&e, &["**", "x1", "2"]).unwrap(), v(&["pow2", "x1"]));
        assert_eq!(
            conv(&e, &["**", "x1", "-2"]).unwrap(),
            v(&["inv", "pow2", "x1"])
        );
        assert_eq!(conv(&e, &["**", "x1", "1"]).unwrap(), v(&["x1"]));
        // x**0 -> 1 (never the invalid 'pow0' token).
        assert_eq!(conv(&e, &["**", "x1", "0"]).unwrap(), v(&["1"]));
        // chain factorize order + VE fallback.
        assert_eq!(
            conv(&e, &["**", "x1", "6"]).unwrap(),
            v(&["pow2", "pow3", "x1"])
        );
        // phantom-pow gate: 7 is non-5-smooth -> binary pow kept (a bare pow7 would corrupt arity).
        assert_eq!(
            conv(&e, &["**", "x1", "7"]).unwrap(),
            v(&["pow", "x1", "7"])
        );
        assert_eq!(
            conv(&e, &["**", "x1", "30"]).unwrap(),
            v(&["pow2", "pow3", "pow5", "x1"])
        );
        assert_eq!(
            conv(&e, &["pow2", "pow2", "pow2", "x1"]).unwrap(),
            v(&["pow4", "pow2", "x1"])
        );
        // A fractional power is NOT absorbed into an integer chain (and vice versa).
        assert_eq!(
            conv(&e, &["pow2", "pow1_3", "x1"]).unwrap(),
            v(&["pow2", "pow1_3", "x1"])
        );
        assert_eq!(
            conv(&e, &["pow1_3", "pow2", "x1"]).unwrap(),
            v(&["pow1_3", "pow2", "x1"])
        );
        // Genuine same-family chains still combine.
        assert_eq!(
            conv(&e, &["pow1_2", "pow1_2", "x1"]).unwrap(),
            v(&["pow1_4", "x1"])
        );
        // inner non-smooth exponent stays binary; smooth outer still absorbs to pow2.
        assert_eq!(
            conv(&e, &["**", "**", "x1", "7", "2"]).unwrap(),
            v(&["pow2", "pow", "x1", "7"])
        );
        // neg-of-literal toggles ONE minus.
        assert_eq!(conv(&e, &["neg", "5"]).unwrap(), v(&["-5"]));
        assert_eq!(conv(&e, &["neg", "-5"]).unwrap(), v(&["5"]));
        assert_eq!(
            conv(&e, &["+", "neg", "2", "x1"]).unwrap(),
            v(&["+", "-2", "x1"])
        );
        // float branch + integer-fraction (live: v^(3/2)).
        assert_eq!(
            conv(&e, &["**", "x1", "0.5"]).unwrap(),
            v(&["pow1_2", "x1"])
        );
        assert_eq!(
            conv(&e, &["**", "x1", "2.5"]).unwrap(),
            v(&["pow1_2", "pow5", "x1"])
        );
        assert_eq!(
            conv(&e, &["**", "x1", "0.2"]).unwrap(),
            v(&["pow1_5", "x1"])
        );
        assert_eq!(
            conv(&e, &["**", "x1", "0.1"]).unwrap(),
            v(&["pow", "x1", "0.1"])
        ); // gate-fail KEEP
        assert_eq!(conv(&e, &["**", "x1", "0.0"]).unwrap(), v(&["1"]));
        assert_eq!(
            conv(&e, &["**", "x1", "/", "3", "2"]).unwrap(),
            v(&["pow1_2", "pow3", "x1"])
        );
        assert_eq!(
            conv(&e, &["**", "x1", "/", "-2", "3"]).unwrap(),
            v(&["inv", "pow1_3", "pow2", "x1"])
        );
        // Raw unconfigured powN tokens no longer KeyError: kept (pow7) / combined away (pow1).
        assert_eq!(conv(&e, &["pow7", "x1"]).unwrap(), v(&["pow7", "x1"]));
        assert_eq!(conv(&e, &["pow1", "x1"]).unwrap(), v(&["x1"]));
        // crash-parity: the dead float-division branch still raises.
        assert!(conv(&e, &["**", "x1", "/", "2.0", "3.0"]).is_err());
    }

    /// parse traps pinning the high-level parse contract in CI.
    #[test]
    fn parse_traps() {
        let Some(e) = engine() else { return };
        let v = |xs: &[&str]| -> Vec<String> { xs.iter().map(|s| s.to_string()).collect() };
        let p = |s: &str, c: bool, m: bool| e.parse(s, c, m).unwrap();
        // default (convert=True, mask=False): integer power + negative exponent -> neg (NOT inv).
        assert_eq!(p("x1 ^ 2", true, false), v(&["pow2", "x1"]));
        assert_eq!(p("x1 ** -2", true, false), v(&["neg", "pow2", "x1"]));
        assert_eq!(p("x1 ** -1", true, false), v(&["neg", "x1"]));
        // mask_numbers=True -> numbers_to_constant (float()-based).
        assert_eq!(
            p("3.14 * x1 + 2", true, true),
            v(&["+", "*", "<constant>", "x1", "<constant>"])
        );
        // convert=False -> raw infix_to_prefix + remove_pow1 (no ** conversion).
        assert_eq!(p("x1 + x2", false, false), v(&["+", "x1", "x2"]));
    }
}
