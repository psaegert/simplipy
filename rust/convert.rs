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

        // write_operator: realization ? realization_map[canonical] : canonical. The map is
        // CLOSED over the core serialization language (`Operators::from_specs` fills any the
        // config omitted from `CORE_SERIALIZATION_OPS`), so a projection can never emit an
        // operator this lookup misses.
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

            // `rootn` is a FUNCTION in every dialect: the infix parser reads
            // `rootn(x, n)` as a call, so the generic binary-infix rendering it used to
            // fall into (`x0 rootn 3`) is not merely ugly -- it does not round-trip.
            // Now that even indices survive canonicalization instead of turning into
            // `pow(x, 1/n)`, this path is reached constantly.
            if canonical == "rootn" {
                stack.push((
                    format!("{write_operator}({left_str}, {right_str})"),
                    INF,
                    Some(canonical.to_string()),
                ));
                continue;
            }
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
            // X10 (2026-08-15): a LEADING UNARY MINUS must parenthesize under '**'
            // regardless of precedence bookkeeping -- a bare negative literal is one
            // atomic-precedence token, so the generic rule above never fires, yet the
            // re-parse (this engine, Python, and SymPy 1.14 alike) binds '**' TIGHTER
            // than the minus: '-2 ** x0' reads as -(2 ** x0), a value change. Measured
            // as an exact bijection: 59/5,494 canonical outputs broke, every one a
            // `pow <negative literal>` base (verify_X10). Composites are already
            // wrapped by the generic rule and never start with '-'.
            if write_operator == "**" && left_str.starts_with('-') {
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
                if operand_prec <= power_precedence || operand_str.starts_with('-') {
                    // the leading-minus arm is X10's render site 2: same defect,
                    // same re-parse inversion -- see the binary '**' site's comment.
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
        } else if matches!(chars[i], '-' | '+' | '*' | '/' | '^' | '(' | ')' | ',') {
            // ',' is the 2-ary call-syntax argument separator the pretty renderer
            // emits (`rootn(x0, 3)`, `pow(a, b)`). It used to fall into the silent
            // drop below, which FUSED the two arguments and let operators bind across
            // the boundary: `rootn(x0, x1 - 1)` parsed as rootn(x0 - x1, 1)
            // (hardening H-011, 2026-08-03 -- a round-trip break of the parser's own
            // output language; 97/2000 fuzz rows).
            tokens.push(chars[i].to_string());
            i += 1;
        } else {
            i += 1; // unmatched -> drop (no token, no error; pinned legacy parity)
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
            // RESERVED CONSTANT NAMES of the infix language: the pretty renderer spells
            // the canonical constants as bare `pi`/`e`/`inf`/`nan` (`ac::convert::render_prec`),
            // so the parser reads exactly these identifiers back as the constants --
            // emit-parse closure for the leaf alphabet. Everything else (incl. the dotted
            // `np.pi` spelling, one atomic ident) passes through unchanged.
            prefix_expr.push(match token.as_str() {
                "pi" => "np.pi".to_string(),
                "e" => "np.e".to_string(),
                "inf" => "float(\"inf\")".to_string(),
                "nan" => "float(\"nan\")".to_string(),
                _ => token,
            });
        } else if token == ")" {
            stack.push(token);
        } else if token == "(" {
            while stack.last().is_some_and(|t| t != ")") {
                prefix_expr.push(stack.pop().unwrap());
            }
            if stack.last().is_some_and(|t| t == ")") {
                stack.pop();
            }
        } else if token == "," {
            // Argument separator: flush the just-finished argument's pending operators
            // down to the enclosing ')' (the paren boundary WITHOUT consuming it), so
            // each argument of a 2-ary call parses independently (H-011).
            while stack.last().is_some_and(|t| t != ")") {
                prefix_expr.push(stack.pop().unwrap());
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
                    || next_norm == Some(",")  // `pow(x0, -2)`: unary after a separator (H-011)
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

// The legacy float-approximation chain (`exact_ratio` -> `limit_denominator` ->
// `fraction_limit_denominator`, the CPython `Fraction(float(s)).limit_denominator(1e6)`
// port that H-044 patched) was RETIRED by H-047 (2026-08-05): the `**`-exponent fold
// gate is now spelling-exact (`Rat::parse_decimal`), so nothing approximates. See the
// register rows H-044/H-047 for the mechanisms; git history holds the port.

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
            // H-049 (2026-08-05): a beyond-i128 integer spelling KEEPS the binary pow.
            // The legacy CPython `int(tok)` was arbitrary-precision and NEVER failed on
            // a digit string (the huge power then died at the factorize gate and kept
            // binary pow anyway) -- the i128 parse error was PORT-MINTED (H-044's
            // sibling), and it raised on the engine's OWN rendered infix
            // (`^(170141183460469231731687303715884105728)`, extreme-lane P9 rows).
            let Ok(v) = tok.parse::<i128>() else {
                return Ok(pow_keep(base, exponent));
            };
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
            // H-047 (2026-08-05): the fold gate is SPELLING-EXACT. The port (and the
            // legacy CPython original) gated on `limit_denominator(1e6)` of the f64
            // VALUE -- an APPROXIMATION: any exponent within ~5e-7 of 1 collapsed the
            // pow entirely (`x**(1.0000000000000002) -> x`, `x**(0.9999995) -> x`),
            // near-0 exponents folded to 1 (`x**(1e-9) -> 1`), near -1 to `inv`, and
            // under a powN-declaring vocabulary any near-miss of a <=5/<=5 fraction
            // snapped to the wrong function (`x**(0.5000000000000001) -> sqrt(x)`).
            // The (1,1) case emitted NO token, so no vocabulary gate protected it in
            // ANY config -- a silent value change on the DEFAULT parse path, found by
            // the extreme-literal fuzz lane (P7-infix, 146/2k smoke rows). The exact
            // decimal denotation now decides: every legacy-intended fold survives
            // (nice spellings ARE exact -- "0.5", "0.2", "1.0", "2.5"), every
            // approximation keeps the verbatim binary pow (the AC core reads the
            // decimal exponent exactly). Beyond-i128 spellings refuse -> keep.
            let Some(r) = crate::ac::rat::Rat::parse_decimal(&tok) else {
                return Ok(pow_keep(base, exponent));
            };
            if r.is_zero() {
                return Ok(Ir::L(vec![Ir::S("1".into())])); // x**0.0 -> 1 (exact zero)
            }
            let (num, den) = (r.num().unsigned_abs(), r.den().unsigned_abs());
            if num <= 5 && den <= 5 {
                let (num, den) = (num as i128, den as i128);
                // VOCABULARY GATE (the p/q branch below has the same one): the
                // `pow{num}`/`pow1_{den}` spellings exist only in configs that declare
                // the hyper-op family; a clean-vocab config keeps the binary `pow`
                // (the AC core reads the decimal exponent exactly). Gated per factor
                // actually EMITTED (num==1/den==1 need no token), so legacy configs
                // keep byte-identical conversions.
                if (num != 1
                    && crate::utils::factorize_to_at_most(num, ops.max_power, 1000).is_err())
                    || (den != 1
                        && crate::utils::factorize_to_at_most(den, ops.max_fractional_power, 1000)
                            .is_err())
                {
                    return Ok(pow_keep(base, exponent));
                }
                let mut new_expr = Ir::L(vec![base]); // [base]
                if num != 1 {
                    new_expr = Ir::L(vec![Ir::S(format!("pow{num}")), new_expr]);
                }
                if den != 1 {
                    new_expr = Ir::L(vec![Ir::S(format!("pow1_{den}")), new_expr]);
                }
                if r.is_negative() {
                    new_expr = Ir::L(vec![Ir::S("inv".into()), new_expr]);
                }
                Ok(new_expr)
            } else {
                Ok(pow_keep(base, exponent)) // gate-fail / out-of-domain -> KEEP
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
                    // H-049: beyond-i128 operands keep binary pow (see the unary arm).
                    let (Ok(numerator), Ok(denominator)) = (nt.parse::<i128>(), dt.parse::<i128>())
                    else {
                        return Ok(pow_keep(base, exponent));
                    };
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
                    // Sign via comparison, not the product: `numerator * denominator`
                    // can overflow i128 (H-049 collateral; wrapping would flip the inv).
                    if (numerator < 0) != (denominator < 0) {
                        Ok(Ir::L(vec![Ir::S("inv".into()), Ir::L(vec![dnode])]))
                    } else {
                        Ok(dnode)
                    }
                } else {
                    // H-049: the legacy dead branch RAISED here (`int('2.0')`
                    // failure-parity) -- but the ENGINE'S OWN infix can render an
                    // exponent as a float division (divisor-side literal spelling,
                    // extreme-lane P9 row 355), and the engine's own serialization
                    // must always re-parse. KEEP binary pow; the AC core reads the
                    // division exactly.
                    Ok(pow_keep(base, exponent))
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
                    let new = match s.strip_prefix('-') {
                        Some(rest) => rest.to_string(),
                        None => format!("-{s}"),
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
    // H-043/D4: the tokenize + shunting stages above are iterative, but
    // `convert_expression` below recurses (its IR flatten) -- cap the TOKENIZED length
    // exactly like the token-list boundary caps its input, or a deep infix string
    // aborts the process where the equivalent token list would raise.
    if parsed.len() > crate::MAX_TOKENS {
        return Err(format!(
            "prefix expression too long ({} tokens > {}); refusing to risk a \
             deep-recursion stack overflow",
            parsed.len(),
            crate::MAX_TOKENS
        ));
    }
    // H-007: the tokenizer reads `inf`/`nan`/`1_000` as NAME tokens; refuse them HERE,
    // before `numbers_to_constant` (whose `float()` semantics WOULD read them) can
    // silently absorb a reserved spelling into `<constant>`. Same ruling as the token-list
    // boundary (`ensure_tokens_are_tokens`).
    if let Some(t) = parsed
        .iter()
        .find(|t| crate::utils::reserved_numeric_spelling(t))
    {
        return Err(format!(
            "invalid token {t:?}: reserved numeric spelling -- numeric to Python but not a \
             simplipy numeric literal (H-007); use the canonical spelling ('5', '0.5', \
             '1e-05', '1/3', float(\"inf\"), float(\"-inf\"), float(\"nan\"))"
        ));
    }
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

    /// The conversion layer's subject is the INPUT LANGUAGE of a table that declares the
    /// legacy sugar (unary hyper-operators, `**`, chain combining) -- exactly what the
    /// in-repo legacy-vocabulary fixture provides (audit Tier-1 #3: no asset dependence;
    /// the historical expectations below are unchanged, dev_7-3's table verbatim).
    fn engine() -> Option<Engine> {
        Some(crate::legacy_sugar_engine())
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
        // 2-ary CALL SYNTAX (H-011): the comma is a real argument separator -- it
        // used to be silently dropped, fusing the arguments (`rootn(x0, x1 - 1)`
        // parsed as rootn(x0 - x1, 1)). Composite args on either side, nesting,
        // and unary minus directly after the separator.
        assert_eq!(i2p(&e, "rootn(x0, 3)"), v(&["rootn", "x0", "3"]));
        assert_eq!(
            i2p(&e, "rootn(x0, x1 - 1)"),
            v(&["rootn", "x0", "-", "x1", "1"])
        );
        assert_eq!(
            i2p(&e, "pow(x0 + 1, x1 - 2)"),
            v(&["pow", "+", "x0", "1", "-", "x1", "2"])
        );
        assert_eq!(
            i2p(&e, "rootn(pow(x0, 2), x1 + 1)"),
            v(&["rootn", "pow", "x0", "2", "+", "x1", "1"])
        );
        assert_eq!(i2p(&e, "pow(x0, -2)"), v(&["pow", "x0", "neg", "2"]));
        assert_eq!(i2p(&e, "<constant> * x1"), v(&["*", "<constant>", "x1"]));
        // round-trip identity (parse half; paired with the paren-keeping render).
        let pre = v(&["*", "a", "*", "b", "c"]);
        let inf = e.prefix_to_infix(&pre, Power::StarStar, false).unwrap();
        assert_eq!(e.parse(&inf, false, false).unwrap(), pre);
    }

    /// RESERVED CONSTANT NAMES of the infix language (emit-parse closure): the pretty
    /// renderer spells the canonical constants as bare `pi`/`e`/`inf`/`nan`, so the infix
    /// parser reads exactly those identifiers back as the constants -- previously they
    /// re-parsed as free VARIABLES and the value silently changed on feed-back.
    #[test]
    fn infix_reserved_constant_names() {
        let Some(e) = engine() else { return };
        let v = |xs: &[&str]| -> Vec<String> { xs.iter().map(|s| s.to_string()).collect() };
        assert_eq!(i2p(&e, "x0 + pi"), v(&["+", "x0", "np.pi"]));
        assert_eq!(i2p(&e, "e*x0"), v(&["*", "np.e", "x0"]));
        assert_eq!(i2p(&e, "inf"), v(&["float(\"inf\")"]));
        assert_eq!(i2p(&e, "nan"), v(&["float(\"nan\")"]));
        assert_eq!(i2p(&e, "-inf"), v(&["neg", "float(\"inf\")"]));
        // The dotted spellings stay atomic idents and pass through unmapped.
        assert_eq!(i2p(&e, "np.pi + x0"), v(&["+", "np.pi", "x0"]));
        // Prefixed/suffixed identifiers are NOT reserved (`pie`, `inf2` stay variables).
        assert_eq!(i2p(&e, "pie + inf2"), v(&["+", "pie", "inf2"]));
    }

    /// `**` conversion under a CLEAN config (no `pow{N}`/`pow1_{N}` in the vocabulary):
    /// every exponent keeps the core binary `pow` -- the legacy hyper-op spellings exist
    /// only in configs that declare them (the dev_7-3 pins above stay byte-identical).
    /// Previously the float-fraction branch emitted `pow1_2` unconditionally, so
    /// `x0^0.5` produced a token the clean engine then rejected as malformed.
    #[test]
    fn power_conversion_keeps_binary_pow_without_the_hyper_vocabulary() {
        use crate::operators::{OperatorSpec, Operators};
        let mut specs: rustc_hash::FxHashMap<String, OperatorSpec> = Default::default();
        let mut order = Vec::new();
        for (name, arity, prec) in [("+", 2, 1.0), ("*", 2, 2.0), ("pow", 2, 3.0)] {
            order.push(name.to_string());
            specs.insert(
                name.to_string(),
                OperatorSpec {
                    realization: name.to_string(),
                    alias: vec![],
                    inverse: None,
                    arity,
                    precedence: Some(prec),
                    commutative: arity == 2 && name != "pow",
                },
            );
        }
        let ops = Operators::from_specs(order, specs);
        let p = |s: &str| super::parse(s, &ops, true, false).unwrap();
        let v = |xs: &[&str]| -> Vec<String> { xs.iter().map(|s| s.to_string()).collect() };
        assert_eq!(p("x0^2"), v(&["pow", "x0", "2"]));
        assert_eq!(p("x0^0.5"), v(&["pow", "x0", "0.5"]));
        assert_eq!(p("x0^(-0.5)"), v(&["pow", "x0", "-0.5"]));
        assert_eq!(p("x0^2.0"), v(&["pow", "x0", "2.0"]));
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
        // H-044 pin: a float exponent whose exact ratio exceeds i128 KEEPS binary pow.
        // 2^128 / 2^129 wrapped `num0 << e` to numerator 0 and folded x**3.4e38 to `1`;
        // 3*2^127 wrapped to i128::MIN (debug: abs() overflow panic in `reduce`).
        for huge in [
            "3.402823669209385e38",
            "6.80564733841877e38",
            "5.104235503814077e38",
        ] {
            assert_eq!(
                conv(&e, &["**", "x1", huge]).unwrap(),
                v(&["pow", "x1", huge]),
                "out-of-i128-domain exponent {huge} must KEEP"
            );
        }
        // Raw unconfigured powN tokens no longer KeyError: kept (pow7) / combined away (pow1).
        assert_eq!(conv(&e, &["pow7", "x1"]).unwrap(), v(&["pow7", "x1"]));
        assert_eq!(conv(&e, &["pow1", "x1"]).unwrap(), v(&["x1"]));
        // H-049 (2026-08-05): the dead float-division branch KEEPS binary pow now --
        // the legacy raise (int('2.0') failure-parity) fired on the engine's OWN
        // divisor-side rendered infix, and the engine's own output must always
        // re-parse.
        assert_eq!(
            conv(&e, &["**", "x1", "/", "2.0", "3.0"]).unwrap(),
            v(&["pow", "x1", "/", "2.0", "3.0"])
        );
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
