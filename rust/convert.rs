//! The M2 "drop-in engine" conversion surface: `prefix_to_infix`, `infix_to_prefix`,
//! `convert_expression`, `parse`. These let flash-ansr swap the whole `simplipy_engine` object for
//! the Rust port (it calls `is_valid` / `prefix_to_infix` / `infix_to_prefix` / `parse` on the engine
//! OBJECT), not just route `.simplify`.
//!
//! Faithful target: dev_7-3 @ simplipy 0.2.15 / tag c84741f. Every function is byte-identical
//! tag<->HEAD (it sits outside the P0 operand-index regions), so the tag IS the parity reference.
//!
//! Characterization provenance: the trap maps + ~111 Python-validated adversarial inputs come from
//! the 4-agent characterization workflow (wf_5221177a-a5b); see `benchmarks/results/` and
//! `corpus/_m2_adversarial.json`. Trap ids (T1..) below reference that analysis.

use crate::operators::{pow1_power, pow_power, Operators};
use crate::utils::is_numeric_string;

/// How `prefix_to_infix` renders power operators (engine.py:409 `power` param).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Power {
    /// `'func'`: keep engine names (`pow(a, b)`, `pow2(x)`). The DEFAULT and dominant deployment mode.
    Func,
    /// `'**'`: Python-style exponentiation (`a ** b`, `x**2`, `x**(1/2)`).
    StarStar,
}

/// `op_associativity` (engine.py:442-449): a LOCAL hardcoded map, NOT from config. Absent operators
/// default to `'left'`. Only `+,-,*,/` (left) and `**,pow` (right) appear.
fn associativity(op: &str) -> &'static str {
    match op {
        "**" | "pow" => "right",
        _ => "left", // +,-,*,/ and the default
    }
}

/// `right_allows_flatten` (engine.py:457-466): a right operand of EQUAL precedence may omit parens
/// only when `child_root` is in `flatten_map[parent_op]` (`{'+':{'+','-'}, '*':{'*','/'}}`). A `None`
/// child root (a terminal) trivially allows flatten (459-460), though that path is unreachable in the
/// equal-precedence test (a terminal has precedence inf, never == a finite parent precedence).
fn right_allows_flatten(parent_op: &str, child_root: Option<&str>, fixed: bool) -> bool {
    let child_root = match child_root {
        None => return true,
        Some(r) => r,
    };
    if fixed {
        // FIX (#5, render half): disable flattening so equal-precedence right operands keep their
        // parens; paired with the left-assoc parse this preserves prefix<->infix round-trip identity.
        return false;
    }
    match parent_op {
        "+" => child_root == "+" || child_root == "-",
        "*" => child_root == "*" || child_root == "/",
        _ => false, // '-' and '/' as PARENT are ABSENT -> never flatten the right operand (T5)
    }
}

/// A render-stack element (engine.py:455): `(rendered_str, precedence, root_operator)`.
type Item = (String, f64, Option<String>);

/// Faithful port of `prefix_to_infix` (engine.py:409-579). Renders a prefix token list to an infix
/// string with minimal parentheses. Returns `Err` where Python raises `ValueError` (malformed prefix:
/// too few / too many operands) -- the FFI maps that to a Python `ValueError`; the differential checks
/// failure-PARITY, not message text.
///
/// The trap-critical points (all from the characterization, all covered by `_m2_adversarial.json`):
///  * precedence is f64 (`neg`=2.5 sits strictly between `*`=2 and `pow`=3) -- T3.
///  * THREE distinct unary comparison operators: `neg` strict `<`, `inv` `<=`, `pow{N}`-under-`**`
///    `<=` -- T2. Do not unify them.
///  * `inv` PUSHES `op_precedence['/']` (=2) as its node precedence, not its own 4 -- T4.
///  * under `realization=True`, a realization containing `'.'` renders pure func-form and PREEMPTS the
///    neg/inv/pow special-cases (so only `+,-,*` are ever infix) -- T1; checked BEFORE `power`.
///  * `'-'`/`'/'` as a parent never flatten an equal-precedence right operand -- T5.
///  * spacing is load-bearing: `a + b` (spaces) vs `x**2` / `-x` / `1/x` (no spaces) -- T7/T8.
pub fn prefix_to_infix(
    tokens: &[String],
    ops: &Operators,
    power: Power,
    realization: bool,
    fixed: bool,
) -> Result<String, String> {
    if tokens.is_empty() {
        return Ok(String::new()); // engine.py:436-437
    }

    const INF: f64 = f64::INFINITY; // FUNC_PRECEDENCE = TERMINAL_PRECEDENCE (451-452)
    let mut stack: Vec<Item> = Vec::with_capacity(tokens.len());

    for token in tokens.iter().rev() {
        // operator = realization_to_operator.get(token, token) (469): an already-realized input token
        // (e.g. 'simplipy.operators.sin') maps back to canonical first (T12).
        let operator: &str = ops
            .realization_to_operator
            .get(token)
            .map(|s| s.as_str())
            .unwrap_or(token);
        // canonical_operator = operator_aliases.get(operator, operator) (470).
        let canonical: &str = ops
            .operator_aliases
            .get(operator)
            .map(|s| s.as_str())
            .unwrap_or(operator);

        // membership (472-476): 3-way OR.
        let is_op = ops.is_operator_token(canonical)
            || ops.operator_aliases.contains_key(operator)
            || ops.operator_arity_compat.contains_key(canonical);

        if !is_op {
            stack.push((token.clone(), INF, None)); // terminal (570-571)
            continue;
        }

        // arity = operator_arity_compat.get(canonical, 1) (477).
        let arity = ops
            .operator_arity_compat
            .get(canonical)
            .copied()
            .unwrap_or(1) as usize;
        if stack.len() < arity {
            // 479-480: ValueError with the RESOLVED operator var (pre-alias).
            return Err(format!(
                "Invalid prefix expression: Not enough operands for operator '{operator}'"
            ));
        }
        // operands_data = [stack.pop() for _ in range(arity)] -> [0]=left, [1]=right (T10).
        let operands_data: Vec<Item> = (0..arity).map(|_| stack.pop().unwrap()).collect();

        // write_operator (484-488): realization ? realization_map[canonical] : canonical.
        let write_operator: String = if realization {
            ops.operator_realizations
                .get(canonical)
                .cloned()
                .unwrap_or_else(|| canonical.to_string())
        } else {
            canonical.to_string()
        };

        // 491-494: realization '.'-in-name OR arity>2 -> pure func-form, PREEMPTS all special-cases (T1).
        if realization && (write_operator.contains('.') || arity > 2) {
            let joined = join_operands(&operands_data);
            stack.push((
                format!("{write_operator}({joined})"),
                INF,
                Some(canonical.to_string()),
            ));
            continue;
        }

        // current_precedence = op_precedence.get(canonical, op_precedence.get('pow', INF)) (496).
        let mut current_precedence = ops
            .precedence_get(canonical)
            .unwrap_or_else(|| ops.precedence_get("pow").unwrap_or(INF));
        // current_assoc = op_associativity.get(canonical, 'left') (497).
        let mut current_assoc = associativity(canonical);

        if arity == 2 {
            let (mut left_str, left_prec, _left_root) = operands_data[0].clone();
            let (mut right_str, right_prec, right_root) = operands_data[1].clone();
            let mut write_operator = write_operator;

            // pow under power='func' -> func-form (503-506).
            if canonical == "pow" && power == Power::Func {
                stack.push((
                    format!("{write_operator}({left_str}, {right_str})"),
                    INF,
                    Some(canonical.to_string()),
                ));
                continue;
            }
            // pow under power='**' -> switch to infix '**', right-assoc (508-511).
            if canonical == "pow" && power == Power::StarStar {
                write_operator = "**".to_string();
                current_precedence = ops.precedence_get("**").unwrap_or(current_precedence);
                current_assoc = "right";
            }

            // left paren (513-516): left_prec < cur OR (== AND assoc right).
            if left_prec < current_precedence
                || (left_prec == current_precedence && current_assoc == "right")
            {
                left_str = format!("({left_str})");
            }
            // right paren (518-522): right_prec < cur OR (== AND assoc left AND !flatten).
            if right_prec < current_precedence
                || (right_prec == current_precedence
                    && current_assoc == "left"
                    && !right_allows_flatten(canonical, right_root.as_deref(), fixed))
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
            let is_pow_op = pow_power(canonical).is_some(); // r'pow\d+(?!_)' (530)
            let is_frac_pow_op = pow1_power(canonical).is_some(); // r'pow1_\d+' (531)

            if canonical == "neg" {
                // 533-538: parens iff operand_prec STRICT-< current (T2).
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
                // 540-546: parens iff operand_prec <= current; PUSH op_precedence['/'] (=2), not own (T4).
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
                // 548-561: x**N / x**(1/N); operand parens iff operand_prec <= power_prec (T8).
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
            // func fallback (563-564).
            stack.push((
                format!("{write_operator}({operand_str})"),
                INF,
                Some(canonical.to_string()),
            ));
            continue;
        }

        // 567-569: nullary / arity>2 fallback (DEAD for dev_7-3; max arity 2, no nullary ops).
        let joined = join_operands(&operands_data);
        stack.push((
            format!("{write_operator}({joined})"),
            INF,
            Some(canonical.to_string()),
        ));
    }

    if stack.len() != 1 {
        // 573-576: too many operands. The Python message embeds a list-repr of the leftover rendered
        // parts (stack order, reversed vs input); we surface failure-parity, not the exact repr.
        let parts: Vec<String> = stack.iter().map(|(s, _, _)| format!("'{s}'")).collect();
        return Err(format!(
            "Malformed prefix expression: too many operands remain after processing. Stack: [{}]",
            parts.join(", ")
        ));
    }
    Ok(stack.into_iter().next().unwrap().0)
}

/// `', '.join(op_str for op_str, _, _ in operands_data)` (engine.py:492,568): join the rendered
/// strings of the popped operands in pop order ([0]=left, ..).
fn join_operands(operands_data: &[Item]) -> String {
    operands_data
        .iter()
        .map(|(s, _, _)| s.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

// ---- infix_to_prefix (engine.py:581) -------------------------------------------------------------

/// Tokenize the (space-stripped) infix string, faithfully to the Python regex
/// `<constant>|number|[A-Za-z_][\w.]*|\*\*|[-+*/^()]` under `re.findall` semantics: scan left to
/// right, at each position take the FIRST alternative (in pattern order) that matches, and SILENTLY
/// DROP any char that matches no alternative (T6). Numbers/identifiers are emitted as verbatim source
/// substrings. (`\w` is treated as ASCII `[A-Za-z0-9_]`; the deployment corpus is ASCII -- a non-ASCII
/// identifier is the documented out-of-domain boundary.)
fn tokenize_infix(s: &str, fixed: bool) -> Vec<String> {
    let chars: Vec<char> = s.chars().filter(|&c| c != ' ').collect(); // `.replace(' ', '')`
    let n = chars.len();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < n {
        // FIX (`fixed` only): the numeric folder's inf/nan tokens stay ATOMIC -- mirrors the Python
        // `float_special` alternation -- else they split on the '(' / '"'. Leads the scan; the token
        // is then classified as a leaf by `is_ident_start` in `infix_to_prefix`.
        if fixed {
            if let Some(j) = match_float_special(&chars, i) {
                tokens.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
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

/// `re.fullmatch(number_pattern, token)` (engine.py:620): does the WHOLE token parse as a number?
fn is_number_fullmatch(token: &str) -> bool {
    let chars: Vec<char> = token.chars().collect();
    !chars.is_empty() && match_number(&chars, 0) == Some(chars.len())
}

/// `re.match(r'[A-Za-z_][\w.]*', token)` (engine.py:622, unanchored): the token STARTS with an
/// identifier char. (Since the classifier only ever sees tokenizer outputs, "starts with" suffices.)
fn is_ident_start(token: &str) -> bool {
    token
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
}

/// Faithful port of `infix_to_prefix` (engine.py:581-653): a RIGHT-to-LEFT shunting-yard. Never
/// raises (degenerate/malformed inputs produce structurally-degenerate prefix lists, matching Python).
///
/// `fixed` selects the corrected behavior of the deliberate-improvement line (conversion-quirk #4: the
/// `^`->`**` normalization also applies to the unary-minus lookahead). `fixed=false` is the faithful
/// deployed-tag `dev_7-3` behavior.
pub fn infix_to_prefix(infix_expression: &str, ops: &Operators, fixed: bool) -> Vec<String> {
    let mut tokens = tokenize_infix(infix_expression, fixed);
    tokens.reverse(); // 609: right-to-left parse

    let mut stack: Vec<String> = Vec::new();
    let mut prefix_expr: Vec<String> = Vec::new();
    let prec = |t: &str| ops.precedence_get(t).unwrap_or(0.0); // `.get(t, 0)` (638)

    let mut i = 0;
    while i < tokens.len() {
        // current token, with '^' normalized to '**' (616-617). The LOOKAHEAD reads the RAW token.
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
            // operator. Unary-minus detection (633): on the REVERSED stream, tokens[i+1] is the
            // original LEFT neighbor; membership is the FULL precedence_compat keyset (raw '^' absent).
            let next_raw = tokens.get(i + 1).map(|s| s.as_str());
            // FIX (#4, `fixed` only): normalize '^'->'**' for the lookahead too, so `x ^ -y` parses
            // the '-' as unary exactly like `x ** -y`. Faithful mode leaves the raw '^' (absent from
            // the precedence map) so the '-' stays binary.
            let next_norm = match next_raw {
                Some("^") if fixed => Some("**"),
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
            // 637: `token != ')'` is always true here. The ELSE block (643-646) is provably a plain
            // push (`stack.insert(-1)` only runs on an empty stack, where it == push) -- T1.
            // Faithful: pop on `>=` (right-leans left-assoc chains -- #5). FIX (#5 parse half,
            // `fixed`): respect associativity -- pop on strict `>` for left-assoc, `>=` only for
            // right-assoc ('**'/'pow'). Coordinated with the render half (right_allows_flatten off)
            // to preserve prefix<->infix round-trip identity.
            if stack.last().is_some_and(|t| t != ")") {
                let cur = prec(&token);
                let right_assoc = token == "**" || token == "pow";
                while let Some(top) = stack.last() {
                    if top == ")" {
                        break;
                    }
                    let tp = prec(top);
                    let pop = if fixed {
                        tp > cur || (tp == cur && right_assoc)
                    } else {
                        tp >= cur
                    };
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
        prefix_expr.push(op); // 650-651
    }
    prefix_expr.reverse(); // 653: `[::-1]`
    prefix_expr
}

// ---- convert_expression (engine.py:655) ----------------------------------------------------------

/// The nested-list intermediate representation `convert_expression` builds: an arbitrarily-nested
/// list of strings, exactly as Python (`[token]` leaves, `[op, [children]]` nodes, plus the quirky
/// `[base]` extra-nesting the float branch produces). `flatten` linearizes it to a prefix token list.
#[derive(Debug, Clone)]
enum Ir {
    S(String),
    L(Vec<Ir>),
}

/// `flatten_nested_list(list_of_items)[::-1]` (utils.py:362 + the `[::-1]` at engine.py:775,849):
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

/// Replace `node[0]` (a string leaf) in place: `stack[-1][0] = new` (engine.py:690).
fn set_first(ir: &mut Ir, new: String) {
    if let Ir::L(v) = ir {
        if let Some(Ir::S(s)) = v.first_mut() {
            *s = new;
        }
    }
}

/// `re.match(r'-?\d+$', s)` (engine.py:708,737): optional leading `-`, then >=1 digits, whole string.
fn is_int_string(s: &str) -> bool {
    let t = s.strip_prefix('-').unwrap_or(s);
    !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit())
}

/// `re.match(r'pow\d+', s)` (engine.py:794, NO negative lookahead): starts with `pow` + >=1 digit.
/// This DELIBERATELY matches `pow1_3` (sees the `pow1` prefix) -- the faithful chain-absorption bug (T3).
fn matches_int_pow(s: &str) -> bool {
    s.strip_prefix("pow")
        .is_some_and(|r| r.bytes().next().is_some_and(|b| b.is_ascii_digit()))
}

/// `re.match(r'pow1_\d+', s)` (engine.py:794): starts with `pow1_` + >=1 digit.
fn matches_frac_pow(s: &str) -> bool {
    s.strip_prefix("pow1_")
        .is_some_and(|r| r.bytes().next().is_some_and(|b| b.is_ascii_digit()))
}

/// `int(re.match(r'pow(\d+)', op).group(1))` (engine.py:810): the leading digit-run right after `pow`.
/// For `pow1_3` this is `1` (the chain-absorption bug drops the `_3`).
fn int_chain_exp(op: &str) -> Option<i128> {
    let rest = op.strip_prefix("pow")?;
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    // i128 (not i64): a `pow<N>` exponent can exceed i64 (e.g. `x ** 2^63`); Python's int() is
    // arbitrary-precision, so parsing as i64 would drop a real exponent (-> wrong decomposition).
    rest[..end].parse::<i128>().ok()
}

/// `Fraction(x).as_integer_ratio()` reduced (engine.py:719 `Fraction(abs(float(s)))`): the EXACT
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

/// `Fraction(abs(float(s))).limit_denominator()` reduced (engine.py:719). `None` only on the
/// documented out-of-i128-domain pathological inputs (never on the deployment distribution).
fn fraction_limit_denominator(x: f64) -> Option<(i128, i128)> {
    let (num, den) = exact_ratio(x)?;
    Some(limit_denominator(num, den))
}

/// `['pow', [base, exponent]]` -- the KEEP fallback (gate-fail / non-numeric exponent).
fn pow_keep(base: Ir, exponent: Ir) -> Ir {
    Ir::L(vec![Ir::S("pow".into()), Ir::L(vec![base, exponent])])
}

/// `**` handling (engine.py:702-763): integer / float / integer-fraction exponent. `Err` mirrors the
/// dead len==2 float-division branch's `int()` `ValueError` (failure-parity).
fn handle_pow(base: Ir, exponent: Ir, _ops: &Operators, fixed: bool) -> Result<Ir, String> {
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
            if fixed && v == 0 {
                return Ok(Ir::L(vec![Ir::S("1".into())])); // FIX (#2): x**0 -> 1 (not 'pow0')
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
                    if fixed && num == 0 {
                        return Ok(Ir::L(vec![Ir::S("1".into())])); // FIX (#2): x**0.0 -> 1
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
        // exponent[0][0] == '/' and both operands numeric strings (engine.py:735).
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
                    if fixed && numerator == 0 {
                        return Ok(Ir::L(vec![Ir::S("1".into())])); // FIX (#2): x**(0/N) -> 1
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
                    // dead float-division branch: int('2.0') raises BEFORE limit_denominator (T5).
                    Err("invalid literal for int()".into())
                }
            }
            _ => Ok(pow_keep(base, exponent)), // else -> KEEP
        }
    } else {
        Ok(pow_keep(base, exponent))
    }
}

/// Faithful port of `convert_expression` (engine.py:655-849): normalize a prefix expression into the
/// engine's internal form (`**` -> `pow{N}`, chained powers combined, unary negation folded into
/// numeric literals). `Err` mirrors a Python raise (raw unconfigured `powN` token KeyError -- T2; the
/// dead float-division `int()` ValueError -- T5).
///
/// `fixed` selects the deliberate-improvement corrected behavior (conversion-quirks #1 fractional power
/// no longer absorbed/dropped, #2 `x**0`->`1`, #3 neg-of-literal toggles one minus, #6 raw `powN` no
/// longer KeyErrors). `fixed=false` is the faithful deployed-tag `dev_7-3` behavior.
pub fn convert_expression(
    prefix_expr: &[String],
    ops: &Operators,
    fixed: bool,
) -> Result<Vec<String>, String> {
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
        // arity = operator_arity_compat[operator] -- HARD index (KeyError on a raw unconfigured pow).
        let arity = match ops.operator_arity_compat.get(operator) {
            Some(a) => *a as usize,
            None if fixed => 1, // FIX (#6): graceful default like pass-2's `.get`, instead of KeyError
            None => return Err(format!("KeyError: '{operator}'")),
        };
        if operator == "neg" {
            match stack.last().and_then(first_str) {
                Some(s) if is_numeric_string(s) => {
                    let s = s.to_string();
                    let mut top = stack.pop().ok_or("neg: empty stack")?;
                    // Faithful: always prepend '-' (the strip elif is DEAD -> '--5'). FIX (#3, `fixed`):
                    // toggle ONE leading '-' (strip if already negative, else prepend).
                    let new = if fixed && s.starts_with('-') {
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
            stack.push(handle_pow(base, exponent, ops, fixed)?);
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
                        // Faithful: the integer chain uses `matches_int_pow` (no lookahead), which
                        // absorbs a child `pow1_M` and drops the `_M` (#1). FIX (`fixed`): use
                        // `pow_power` (the `(?!_)` lookahead) so a `pow1_M` is NOT absorbed.
                        Ir::S(op0)
                            if (is_frac && matches_frac_pow(op0))
                                || (!is_frac
                                    && if fixed {
                                        pow_power(op0).is_some()
                                    } else {
                                        matches_int_pow(op0)
                                    }) =>
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

            // p = product of the chain's exponents (int family uses the bug-faithful leading-digit run).
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

/// `list(reversed(stack[-arity:]))` (engine.py:786,838): the last `arity` items, reversed (NOT popped).
fn take_reversed_tail(stack: &[Ir], arity: usize) -> Result<Vec<Ir>, String> {
    if stack.len() < arity {
        return Err("pass-2: not enough operands".into());
    }
    Ok(stack[stack.len() - arity..].iter().rev().cloned().collect())
}

/// Build the nested pow chain from a list of operator names around `current_operand`
/// (engine.py:818-828): `[ops[-1], [current]]` innermost, wrapped outward by `ops[-2::-1]`. Empty
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

// ---- parse (engine.py:852) -----------------------------------------------------------------------

/// Faithful port of `parse` (engine.py:852): `infix_to_prefix` -> (if `convert`) `convert_expression`
/// -> (if `mask_numbers`) `numbers_to_constant` -> ALWAYS `remove_pow1`. The high-level entry that
/// closes `simplify(str)` and the flash-ansr canonicalization path. `Err` propagates a
/// `convert_expression` raise.
pub fn parse(
    infix_expression: &str,
    ops: &Operators,
    convert: bool,
    mask_numbers: bool,
    fixed: bool,
) -> Result<Vec<String>, String> {
    let parsed = infix_to_prefix(infix_expression, ops, fixed);
    let parsed = if convert {
        convert_expression(&parsed, ops, fixed)?
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

    /// Branch-discriminating cases, all outputs verified verbatim against the tag Python
    /// (corpus/_m2_adversarial.json). The full 40000-real + 35-adversarial 0-diff gate lives in
    /// benchmarks/diff_prefix_to_infix.py; this pins the killer traps in CI.
    #[test]
    fn prefix_to_infix_traps() {
        let Some(e) = engine() else { return };
        let f = Power::Func;
        let s = Power::StarStar;
        // T2 neg strict-< (equal-prec NO parens) vs T4 inv <= ; T1 realization preempt; T5 flatten.
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
        assert_eq!(
            p2i(&e, &["*", "/", "x1", "x2", "/", "x3", "x4"], f, false).unwrap(),
            "x1 / x2 * x3 / x4"
        );
        // T7/T8 pow rendering + spacing.
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
        // T1 realization=True: only +,-,* infix; neg/div func-form; power ignored.
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

    /// infix_to_prefix traps (outputs verified verbatim vs tag Python). Full 0-diff gate (17030 real
    /// infix + 25 adversarial, each direction) in benchmarks/diff_infix_to_prefix.py.
    #[test]
    fn infix_to_prefix_traps() {
        let Some(e) = engine() else { return };
        let v = |xs: &[&str]| -> Vec<String> { xs.iter().map(|s| s.to_string()).collect() };
        // T1 leading unary (only reachable insert(-1)==push path); T4 standalone '-'.
        assert_eq!(i2p(&e, "-x1"), v(&["neg", "x1"]));
        assert_eq!(i2p(&e, "-"), v(&["neg"]));
        // T3 '^' lookahead asymmetry: '**' -> neg, raw '^' -> binary '-'.
        assert_eq!(i2p(&e, "x1 ** - x2"), v(&["neg", "**", "x1", "x2"]));
        assert_eq!(i2p(&e, "x1 ^ - x2"), v(&["-", "**", "x1", "x2"]));
        // T4 function-name left neighbor -> unary; T2 neg float precedence both ways.
        assert_eq!(i2p(&e, "sin - x1"), v(&["neg", "sin", "x1"]));
        assert_eq!(i2p(&e, "-x1 ** 2"), v(&["neg", "**", "x1", "2"]));
        assert_eq!(i2p(&e, "-x1 * x2"), v(&["*", "neg", "x1", "x2"]));
        // T7 right-assoc via >= pop; T8 round-trip non-identity for inv.
        assert_eq!(
            i2p(&e, "a - b - c - d"),
            v(&["-", "a", "-", "b", "-", "c", "d"])
        );
        assert_eq!(i2p(&e, "1/x1"), v(&["/", "1", "x1"]));
        // T6 tokenizer: '**' before '*', drop unmatched, empty parens.
        assert_eq!(i2p(&e, "x1***x2"), v(&["*", "**", "x1", "x2"]));
        assert_eq!(i2p(&e, "x1 $ x2"), v(&["x1", "x2"]));
        assert_eq!(i2p(&e, "()"), Vec::<String>::new());
        assert_eq!(i2p(&e, ""), Vec::<String>::new());
        // T9 scientific notation single token.
        assert_eq!(i2p(&e, "1.5e-2 * x1"), v(&["*", "1.5e-2", "x1"]));
        assert_eq!(i2p(&e, "<constant> * x1"), v(&["*", "<constant>", "x1"]));
    }

    fn conv(e: &Engine, toks: &[&str]) -> Result<Vec<String>, String> {
        let t: Vec<String> = toks.iter().map(|s| s.to_string()).collect();
        e.convert_expression(&t)
    }

    /// convert_expression traps (outputs verified verbatim vs tag Python). Full 0-diff gate (27030
    /// real convert inputs + 51 adversarial, incl. crash-parity) in benchmarks/diff_convert_expression.py.
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
        assert_eq!(conv(&e, &["**", "x1", "0"]).unwrap(), v(&["pow0", "x1"]));
        // chain factorize order + VE fallback + mixed-chain absorption bug (T3).
        assert_eq!(
            conv(&e, &["**", "x1", "6"]).unwrap(),
            v(&["pow2", "pow3", "x1"])
        );
        assert_eq!(conv(&e, &["**", "x1", "7"]).unwrap(), v(&["pow7", "x1"]));
        assert_eq!(
            conv(&e, &["**", "x1", "30"]).unwrap(),
            v(&["pow2", "pow3", "pow5", "x1"])
        );
        assert_eq!(
            conv(&e, &["pow2", "pow2", "pow2", "x1"]).unwrap(),
            v(&["pow4", "pow2", "x1"])
        );
        assert_eq!(
            conv(&e, &["pow2", "pow1_3", "x1"]).unwrap(),
            v(&["pow2", "x1"])
        );
        assert_eq!(
            conv(&e, &["pow1_3", "pow2", "x1"]).unwrap(),
            v(&["pow1_3", "pow2", "x1"])
        );
        assert_eq!(
            conv(&e, &["**", "**", "x1", "7", "2"]).unwrap(),
            v(&["pow2", "pow7", "x1"])
        );
        // neg-on-number double-minus (T7).
        assert_eq!(conv(&e, &["neg", "5"]).unwrap(), v(&["-5"]));
        assert_eq!(conv(&e, &["neg", "-5"]).unwrap(), v(&["--5"]));
        assert_eq!(
            conv(&e, &["+", "neg", "2", "x1"]).unwrap(),
            v(&["+", "-2", "x1"])
        );
        // float branch (dead on real data) + integer-fraction (live: v^(3/2)).
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
        assert_eq!(conv(&e, &["**", "x1", "0.0"]).unwrap(), v(&["pow0", "x1"]));
        assert_eq!(
            conv(&e, &["**", "x1", "/", "3", "2"]).unwrap(),
            v(&["pow1_2", "pow3", "x1"])
        );
        assert_eq!(
            conv(&e, &["**", "x1", "/", "-2", "3"]).unwrap(),
            v(&["inv", "pow1_3", "pow2", "x1"])
        );
        // crash-parity: raw unconfigured powN token (T2) + dead float-division (T5).
        assert!(conv(&e, &["pow7", "x1"]).is_err());
        assert!(conv(&e, &["pow1", "x1"]).is_err());
        assert!(conv(&e, &["**", "x1", "/", "2.0", "3.0"]).is_err());
    }

    /// parse traps (verified vs tag Python). Full 0-diff gate (17030 real infix x deployment combos)
    /// in benchmarks/diff_parse.py.
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

    /// The CORRECTED (deliberate-improvement) variants -- conversion-quirk fixes #1-#4,#6. Outputs
    /// verified == the Python fix branch (benchmarks/diff_fixed.py, 269857 comparisons, 0 diffs). The
    /// faithful (dev_7-3) variants keep the buggy behavior (pinned in the *_traps tests).
    #[test]
    fn fixed_quirk_behavior() {
        let Some(e) = engine() else { return };
        let v = |xs: &[&str]| -> Vec<String> { xs.iter().map(|s| s.to_string()).collect() };
        let cf = |toks: &[&str]| {
            e.convert_expression_fixed(&toks.iter().map(|s| s.to_string()).collect::<Vec<_>>())
        };
        // #1 fractional power preserved (faithful drops it).
        assert_eq!(
            cf(&["pow2", "pow1_3", "x1"]).unwrap(),
            v(&["pow2", "pow1_3", "x1"])
        );
        assert_eq!(
            e.convert_expression(&v(&["pow2", "pow1_3", "x1"])).unwrap(),
            v(&["pow2", "x1"])
        ); // faithful still buggy
           // #1 no over-fix: genuine chains still combine.
        assert_eq!(cf(&["pow2", "pow2", "x1"]).unwrap(), v(&["pow4", "x1"]));
        assert_eq!(
            cf(&["pow1_2", "pow1_2", "x1"]).unwrap(),
            v(&["pow1_4", "x1"])
        );
        // #2 x**0 -> 1 (faithful emits pow0).
        assert_eq!(cf(&["**", "x1", "0"]).unwrap(), v(&["1"]));
        assert_eq!(cf(&["**", "x1", "0.0"]).unwrap(), v(&["1"]));
        // #3 neg-of-literal toggles one minus (faithful makes --5).
        assert_eq!(cf(&["neg", "-5"]).unwrap(), v(&["5"]));
        assert_eq!(cf(&["neg", "5"]).unwrap(), v(&["-5"]));
        // #4 '^' parses unary-minus like '**'.
        assert_eq!(
            e.infix_to_prefix_fixed("x1 ^ - x2"),
            v(&["neg", "**", "x1", "x2"])
        );
        // #6 raw powN no longer KeyErrors.
        assert_eq!(cf(&["pow7", "x1"]).unwrap(), v(&["pow7", "x1"]));
        // faithful infix_to_prefix unchanged ('^' keeps '-' binary; '/' right-leans -- #5 held there).
        assert_eq!(e.infix_to_prefix("x1 ^ - x2"), v(&["-", "**", "x1", "x2"]));
        assert_eq!(
            e.infix_to_prefix("a - b - c"),
            v(&["-", "a", "-", "b", "c"])
        );
        // #5 fixed = COORDINATED parse+render: left-assoc parse + no-flatten render, round-trip kept.
        assert_eq!(
            e.infix_to_prefix_fixed("1/2 * m * v ** 2"),
            v(&["*", "*", "/", "1", "2", "m", "**", "v", "2"])
        );
        assert_eq!(
            e.infix_to_prefix_fixed("a - b - c"),
            v(&["-", "-", "a", "b", "c"])
        );
        assert_eq!(
            e.infix_to_prefix_fixed("a ** b ** c"),
            v(&["**", "a", "**", "b", "c"])
        ); // right-assoc
        assert_eq!(
            e.prefix_to_infix_fixed(&v(&["+", "a", "+", "b", "c"]), Power::Func, false)
                .unwrap(),
            "a + (b + c)"
        );
        // round-trip identity in fixed mode (the invariant the coordinated fix preserves).
        let pre = v(&["*", "a", "*", "b", "c"]);
        let inf = e
            .prefix_to_infix_fixed(&pre, Power::StarStar, false)
            .unwrap();
        assert_eq!(e.parse_fixed(&inf, false, false).unwrap(), pre);
    }
}
