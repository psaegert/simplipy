//! `prettify`: the DISPLAY of a canonical state (number plan v2, phase 3).
//!
//! The state stays exact; this module only decides how a reader sees it. It is a pure function
//! of the canonical expression and a digit budget `D`, and nothing inside the engine calls it or
//! reads what it writes (a source guard in the test suite holds that line). The display keeps
//! the state's structure -- its bags, nested divisors and function applications -- and decides
//! only how each NUMBER is written:
//!
//! 1. **Digit budget.** A spelling's digits are every digit it writes, a numerator of 1
//!    included (`1/12` counts 3, `3/123` counts 4).
//! 2. **A fraction that fits stays a fraction.** Among a value's fraction spellings -- `p/q`
//!    and every integer over a decimal above 1 (`5/21` is `1/4.2`, `10/37` is `1/3.7`) -- the
//!    shortest is taken; if it fits `D` it is written.
//! 3. **Otherwise one float**: the float64 nearest to the value, in its shortest form, as a
//!    multiplier. The display writes the value that float reads back as, so a coefficient a
//!    hair from 1 shows as no coefficient and an exponent a hair from 1/2 as a square root.
//! 4. **pi** is the literal `3.141592653589793`, recognised by exact arithmetic: when `c / pi`
//!    or `c * pi` has a fraction spelling that fits `D` (pi's digits not counted), the value is
//!    written with the literal factored out (`1/(2*3.141592653589793)`).
//! 5. **Divisors.** A float coefficient whose reciprocal is a decimal float64 holds as written
//!    joins an existing denominator (`x1/(243*x2)`); only a bare `x/float` becomes a
//!    multiplier (`x1/243` is `0.00411522633744856*x1`).
//! 6. **Functions of numbers** fold to one float only when every number inside would itself
//!    be written as a float (`sin(sin(9463/890))` folds, `sin(2)` stays). The fold evaluates
//!    with the deployed evaluator's kernels (`numeric::unary_fn` / `binary_fn`).
//! 7. **Exponents** are numbers like any other.
//! 8. **Even roots**: `x^(1/n)` is `rootn(x, n)` and `x^(-1/n)` is `1/rootn(x, n)` for even `n`.
//! 9. **Layout**: one coefficient in front, the denominator grouped (`x0^2/(x1*x2)`) where the
//!    group re-reads into the state's own reciprocals (numbers, variables, their powers and
//!    roots); any other divisor keeps its own slash, as the state keeps it. Scientific notation
//!    from 10^6 and below 10^-3 with a lowercase `e`.
//! 10. **Range**: nothing folds or combines to inf, 0 or a subnormal.
//!
//! Rule 11 (re-simplify the display once and redo it) needs the rule set, so it lives with the
//! engine (`engine::prettify`); this module is the display alone.

use std::cmp::Ordering;
use std::sync::OnceLock;

use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};

use super::expr::{negate_term, Cx, Ex};
use super::rat::{number_domain, Rat};

/// The literal pi is written as (the float64 nearest to pi, in its shortest form).
pub const PI_LITERAL: &str = "3.141592653589793";

fn pi_value() -> &'static Rat {
    static PI: OnceLock<Rat> = OnceLock::new();
    PI.get_or_init(|| {
        let _exact = number_domain(false);
        Rat::parse_decimal(PI_LITERAL).expect("the pi literal is a decimal")
    })
}

/// A display tree. Numerals are unsigned; signs live in `Neg`, in a product's `neg` flag and in
/// a sum's term flags, so each output form places them by its own doctrine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Node {
    /// A nonnegative numeral: `22`, `4.2`, `9.765625e-4`, `3.141592653589793`.
    Num(String),
    /// A name, spelled per form: `(infix, token)` -- `("pi", "np.pi")`, `("x1", "x1")`.
    Name(String, String),
    /// `-a`.
    Neg(Box<Node>),
    /// A sum: terms with a flag `true` when the term is subtracted.
    Sum(Vec<(bool, Node)>),
    /// A product `±(num...)/(group)/(group)...`; an empty numerator is `1`. Each divisor group
    /// is a product of members; one group re-reads as one divisor.
    Prod {
        neg: bool,
        num: Vec<Node>,
        den: Vec<Vec<Node>>,
    },
    Pow(Box<Node>, Box<Node>),
    Call(String, Vec<Node>),
}

/// How a positive value is written.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Spelling {
    /// An integer numeral: one that fits, or a float that writes as a plain integer.
    Int(String),
    /// A float (rule 3).
    Float(String),
    /// `num/den`, `num` omitted when it is 1.
    Frac { num: Option<String>, den: String },
    /// `num*pi/den`.
    PiNum {
        num: Option<String>,
        den: Option<String>,
    },
    /// `num/(den*pi)`.
    PiDen {
        num: Option<String>,
        den: Option<String>,
    },
}

impl Spelling {
    /// The numerals of the numerator and of the denominator.
    fn parts(&self) -> (Vec<Node>, Vec<Node>) {
        let pi = || Node::Num(PI_LITERAL.to_string());
        let n = |s: &Option<String>| s.iter().map(|t| Node::Num(t.clone())).collect::<Vec<_>>();
        match self {
            Spelling::Int(s) | Spelling::Float(s) => (vec![Node::Num(s.clone())], vec![]),
            Spelling::Frac { num, den } => (n(num), vec![Node::Num(den.clone())]),
            Spelling::PiNum { num, den } => {
                let mut top = n(num);
                top.push(pi());
                (top, n(den))
            }
            Spelling::PiDen { num, den } => {
                let mut bottom = n(den);
                bottom.push(pi());
                (n(num), bottom)
            }
        }
    }
}

/// A fraction spelling `n/d` of a positive value: `d` is `None` for an integer value.
#[derive(Clone, Debug)]
struct Fraction {
    digits: usize,
    num: BigUint,
    den: Option<String>,
}

impl Fraction {
    /// Tie order after the digit count: fewer digits WRITTEN (a numerator of 1 is not written
    /// beside another numerator factor: `x1/4.2` writes two digits, `5*x1/21` three), then an
    /// integer denominator (`x1^(5/21)`, not `x1^(1/4.2)`), then the smaller numerator.
    fn key(&self, unit_free: bool) -> (usize, usize, bool, BigUint) {
        let decimal_den = self.den.as_ref().is_some_and(|d| d.contains('.'));
        let silent = unit_free && self.num.is_one() && self.den.is_some();
        (
            self.digits,
            self.digits - usize::from(silent),
            decimal_den,
            self.num.clone(),
        )
    }

    fn spelling(&self) -> (Option<String>, Option<String>) {
        let num = (!self.num.is_one() || self.den.is_none()).then(|| self.num.to_string());
        (num, self.den.clone())
    }
}

fn ndigits(n: &BigUint) -> usize {
    n.to_string().len()
}

fn power_of(base: u32, k: usize) -> BigUint {
    num_traits::pow(BigUint::from(base), k)
}

/// The shortest fraction spelling of a positive value `a` that fits `d` digits, if any
/// (rules 1 and 2). An integer is its own spelling. A non-integer `p/q` is `p/q` itself or an
/// integer `n = p/g` over the decimal `q/g > 1`, for `g` a product of the 2s and 5s of `p`
/// (`5/21 = 1/4.2`). Only `n` below `10^(d-1)` can fit, which bounds the search. `unit_free`:
/// a numerator of 1 is not written in this position (see [`Fraction::key`]).
fn fraction_fit(a: &Rat, d: usize, unit_free: bool) -> Option<Fraction> {
    let (p, q) = a.big_parts();
    let (p, q) = (p.magnitude().clone(), q.magnitude().clone());
    if q.is_one() {
        let digits = ndigits(&p);
        return (digits <= d).then_some(Fraction {
            digits,
            num: p,
            den: None,
        });
    }
    let mut best: Option<Fraction> = None;
    let mut consider = |f: Fraction| {
        if f.digits <= d
            && best
                .as_ref()
                .is_none_or(|b| f.key(unit_free) < b.key(unit_free))
        {
            best = Some(f);
        }
    };
    consider(Fraction {
        digits: ndigits(&p) + ndigits(&q),
        num: p.clone(),
        den: Some(q.to_string()),
    });
    // p = m * 2^v2 * 5^v5 with m free of 2 and 5; n = m * 2^x * 5^y keeps x <= v2, y <= v5.
    let v2 = p.trailing_zeros().unwrap_or(0) as usize;
    let mut m = &p >> v2;
    let five = BigUint::from(5u32);
    let mut v5 = 0usize;
    while (&m % &five).is_zero() {
        m /= &five;
        v5 += 1;
    }
    if d < 2 || ndigits(&m) >= d {
        return best; // n >= m leaves no digit for the denominator
    }
    let limit = power_of(10, d - 1);
    // 2^x <= 10^(d-1) and 5^y <= 10^(d-1)
    let (max_x, max_y) = (v2.min(4 * d), v5.min(2 * d));
    for x in 0..=max_x {
        for y in 0..=max_y {
            if x == v2 && y == v5 {
                continue; // g = 1, the plain p/q above
            }
            let n = &m * power_of(2, x) * power_of(5, y);
            if n >= limit {
                continue;
            }
            let g = &p / &n;
            let den = {
                let _exact = number_domain(false);
                Rat::from_big(BigInt::from(q.clone()), BigInt::from(g))
            };
            let Some(den) = den else { continue };
            if den.cmp_int(1) != Ordering::Greater {
                continue; // a denominator below 1 is a multiplier in disguise
            }
            let Some(text) = den.exact_decimal() else {
                continue;
            };
            let digits = ndigits(&n) + text.chars().filter(char::is_ascii_digit).count();
            consider(Fraction {
                digits,
                num: n,
                den: Some(text),
            });
        }
    }
    best
}

/// The float form of an exact decimal `s` (rule 9): positional between 10^-3 (inclusive) and
/// 10^6 (exclusive), else `d.ddde<exp>`.
fn float_text(s: &Rat) -> String {
    let parts = s
        .decimal_parts()
        .expect("a shortest float spelling is a finite decimal");
    let digits = parts.mantissa.digits();
    let len = digits.len() as i64;
    let lead = parts.exp + len - 1;
    let sign = if parts.negative { "-" } else { "" };
    if !(-3..6).contains(&lead) {
        let (head, tail) = digits.split_at(1);
        let point = if tail.is_empty() { "" } else { "." };
        return format!("{sign}{head}{point}{tail}e{lead}");
    }
    if parts.exp >= 0 {
        return format!("{sign}{digits}{}", "0".repeat(parts.exp as usize));
    }
    let k = (-parts.exp) as usize;
    if digits.len() > k {
        let (i, f) = digits.split_at(digits.len() - k);
        format!("{sign}{i}.{f}")
    } else {
        format!("{sign}0.{digits:0>k$}")
    }
}

/// Is `y` a float the display may write: finite, nonzero and normal (rule 10)?
fn writable(y: f64) -> bool {
    y.is_finite() && y.abs() >= f64::MIN_POSITIVE
}

/// The DISPLAY of one canonical state under a digit budget.
pub struct Display<'a, 'v> {
    cx: &'a Cx<'v>,
    digits: usize,
}

impl<'a, 'v> Display<'a, 'v> {
    pub fn new(cx: &'a Cx<'v>, digits: usize) -> Self {
        Display { cx, digits }
    }

    /// The spelling of a positive value and the value that spelling reads back as (rules 2-4).
    /// `unit_free`: the value is a coefficient beside a numerator factor, where a numerator of
    /// 1 is not written.
    fn spell(&self, a: &Rat, unit_free: bool) -> (Spelling, Rat) {
        let _exact = number_domain(false);
        if let Some(f) = fraction_fit(a, self.digits, unit_free) {
            let (num, den) = f.spelling();
            let sp = match (num, den) {
                (Some(n), None) => Spelling::Int(n),
                (num, Some(den)) => Spelling::Frac { num, den },
                (None, None) => unreachable!("an integer spelling writes its numerator"),
            };
            return (sp, a.clone());
        }
        if let Some(sp) = self.pi_fit(a, unit_free) {
            return (sp, a.clone());
        }
        let y = a.to_f64_nearest();
        let Some(s) = Rat::shortest_reading_as(y).filter(|_| writable(y)) else {
            // Rule 10: no float can stand for it (only exact-domain values get here).
            return (self.exact_spelling(a), a.clone());
        };
        if s != *a {
            // The display writes the value its float reads back as.
            return self.spell(&s, unit_free);
        }
        let text = float_text(&s);
        if s.is_integer() && !text.contains('e') {
            return (Spelling::Int(text), s);
        }
        (Spelling::Float(text), s)
    }

    /// Rule 4: `a` with the literal pi factored out, when the rest has a fraction that fits:
    /// `a / pi` in front of the literal, or `a * pi` over it, whichever writes fewer digits.
    fn pi_fit(&self, a: &Rat, unit_free: bool) -> Option<Spelling> {
        let pi = pi_value();
        let fit = |r: Option<Rat>| r.and_then(|r| fraction_fit(&r, self.digits, unit_free));
        let over = fit(pi.checked_inv().and_then(|inv| a.checked_mul(&inv)));
        let times = fit(a.checked_mul(pi));
        let unit = |f: &Fraction| {
            let (num, den) = f.spelling();
            (num.filter(|n| n != "1"), den)
        };
        match (over, times) {
            (Some(o), t) if t.as_ref().is_none_or(|t| o.digits < t.digits) => {
                let (num, den) = unit(&o);
                Some(Spelling::PiNum { num, den })
            }
            (_, Some(t)) => {
                let (num, den) = unit(&t);
                Some(Spelling::PiDen { num, den })
            }
            _ => None,
        }
    }

    /// A value no float can stand for, written exactly (the exact domain only).
    fn exact_spelling(&self, a: &Rat) -> Spelling {
        if a.decimal_parts().is_some() {
            return Spelling::Float(float_text(a));
        }
        Spelling::Frac {
            num: Some(a.numer_string()),
            den: a.denom_string(),
        }
    }

    /// The node of a signed value, and the value it reads back as.
    fn number(&self, r: &Rat) -> (Node, Rat) {
        if r.is_zero() {
            return (Node::Num("0".to_string()), Rat::ZERO);
        }
        let (sp, value) = self.spell(&r.abs(), false);
        let (num, den) = sp.parts();
        let node = if den.is_empty() && num.len() == 1 {
            num.into_iter().next().unwrap()
        } else {
            Node::Prod {
                neg: false,
                num,
                den: vec![den].into_iter().filter(|g| !g.is_empty()).collect(),
            }
        };
        if r.is_negative() {
            (negate(node), value.checked_neg().unwrap_or(value))
        } else {
            (node, value)
        }
    }

    /// The display of a canonical expression.
    pub fn node(&self, e: &Ex) -> Node {
        if let Some(v) = self.fold(e) {
            return self.number(&v).0;
        }
        let view = self.cx.view;
        let name = |i: &str, t: &str| Node::Name(i.to_string(), t.to_string());
        match e {
            Ex::Num(r) => self.number(r).0,
            Ex::Pi => name("pi", "np.pi"),
            Ex::E => name("e", "np.e"),
            Ex::PosInf => name("inf", "float(\"inf\")"),
            Ex::NegInf => Node::Neg(Box::new(name("inf", "float(\"inf\")"))),
            Ex::NaN => name("nan", "float(\"nan\")"),
            Ex::Const => name("<constant>", "<constant>"),
            Ex::Leaf(t) => {
                let s = view.resolve_owned(*t);
                Node::Name(s.clone(), s)
            }
            Ex::Fun(f, args) => Node::Call(
                view.resolve_owned(*f),
                args.iter().map(|a| self.node(a)).collect(),
            ),
            Ex::Pow(..) => {
                let (den, node) = self.factor(e);
                if den {
                    Node::Prod {
                        neg: false,
                        num: vec![],
                        den: vec![vec![node]],
                    }
                } else {
                    node
                }
            }
            Ex::Add(v) => self.sum(v),
            Ex::Mul(v) => self.product(v),
        }
    }

    /// A product member: `(true, b^k)` for a divisor `b^-k`, else `(false, node)`.
    fn factor(&self, f: &Ex) -> (bool, Node) {
        if let Ex::Pow(b, x) = f {
            if let Ex::Num(r) = &**x {
                if !matches!(&**b, Ex::Num(z) if z.is_zero()) {
                    let (_, k) = self.number(r);
                    return if k.is_negative() {
                        (true, self.power(b, &k.abs()))
                    } else {
                        (false, self.power(b, &k))
                    };
                }
            }
            return (
                false,
                Node::Pow(Box::new(self.node(b)), Box::new(self.node(x))),
            );
        }
        (false, self.node(f))
    }

    /// `b^k` for a positive exponent as displayed: `b` alone for 1, an even root for `1/n`.
    fn power(&self, b: &Ex, k: &Rat) -> Node {
        if k.is_one() {
            return self.node(b);
        }
        if k.numer().is_one() && k.denom().is_even_integer() {
            return Node::Call(
                "rootn".to_string(),
                vec![self.node(b), Node::Num(k.denom().numer_string())],
            );
        }
        Node::Pow(Box::new(self.node(b)), Box::new(self.number(k).0))
    }

    fn product(&self, v: &[Ex]) -> Node {
        // `-(a - b)` reads as `b - a`, as in every canonical printer.
        if let [Ex::Num(r), Ex::Add(terms)] = v {
            if *r == Rat::NEG_ONE {
                if let Some(flipped) = terms
                    .iter()
                    .map(|t| negate_term(t, self.cx))
                    .collect::<Option<Vec<_>>>()
                {
                    return self.sum(&flipped);
                }
            }
        }
        let mut numbers: Vec<Rat> = Vec::new();
        let mut num: Vec<Node> = Vec::new();
        // Divisors: the GROUP of members a divisor product re-reads into separately (numbers,
        // variables and their powers and roots), and every other divisor on its own, as the
        // state holds it -- grouping a sum `S` into `/(2*S)` would re-read as one joined
        // reciprocal the certificates may refuse to split, a different state.
        let mut den: Vec<Node> = Vec::new();
        let mut alone: Vec<Node> = Vec::new();
        for f in v {
            if let Ex::Num(r) = f {
                numbers.push(r.clone());
            } else if let Some(x) = self.fold(f) {
                numbers.push(x);
            } else {
                let (is_den, node) = self.factor(f);
                if is_den && self.groups(f) {
                    den.push(node);
                } else if is_den {
                    alone.push(node);
                } else {
                    num.push(node);
                }
            }
        }
        let mut neg = false;
        let mut lead: Vec<Node> = Vec::new();
        let mut lead_den: Vec<Node> = Vec::new();
        match combine(&numbers, Rat::checked_mul) {
            Some(c) => {
                neg = c.is_negative();
                let (sp, value) = self.spell(&c.abs(), !num.is_empty());
                if !value.is_one() {
                    match self.divisor(&sp, &c.abs(), !den.is_empty() || !alone.is_empty()) {
                        Some(d) => lead_den.push(d),
                        None => {
                            let (n, d) = sp.parts();
                            lead = n;
                            lead_den = d;
                        }
                    }
                }
            }
            None => {
                // Several numbers whose product leaves the range (rule 10): each its own.
                for r in &numbers {
                    let (node, _) = self.number(r);
                    let (s, node) = split_neg(node);
                    neg ^= s;
                    lead.push(node);
                }
            }
        }
        lead.extend(num);
        lead_den.extend(den);
        let mut groups: Vec<Vec<Node>> = Vec::new();
        if !lead_den.is_empty() {
            groups.push(lead_den);
        }
        groups.extend(alone.into_iter().map(|n| vec![n]));
        if groups.is_empty() && lead.len() == 1 {
            let only = lead.pop().unwrap();
            return if neg { negate(only) } else { only };
        }
        Node::Prod {
            neg,
            num: lead,
            den: groups,
        }
    }

    /// May the divisor `f = b^-k` join the divisor group? When its base is a variable or a
    /// symbol: a product of those re-reads into separate reciprocals in every context.
    fn groups(&self, f: &Ex) -> bool {
        let Ex::Pow(b, _) = f else { return false };
        let base = match &**b {
            Ex::Fun(r, args) if self.cx.view.tok_is(*r, "rootn") => &args[0],
            other => other,
        };
        match base {
            Ex::Pi | Ex::E => true,
            Ex::Leaf(t) => {
                self.cx.view.sigil(*t) == 0
                    && !self.cx.view.with_str(*t, crate::utils::looks_numeric)
            }
            _ => false,
        }
    }

    /// Rule 5: a float coefficient whose reciprocal float64 holds as written joins an existing
    /// denominator as that decimal.
    fn divisor(&self, sp: &Spelling, mag: &Rat, has_den: bool) -> Option<Node> {
        let Spelling::Float(_) = sp else { return None };
        if !has_den || mag.is_integer() {
            return None;
        }
        let _exact = number_domain(false);
        let inv = mag.checked_inv()?;
        inv.decimal_parts()?;
        let y = inv.to_f64_nearest();
        (writable(y) && Rat::shortest_reading_as(y).as_ref() == Some(&inv))
            .then(|| Node::Num(float_text(&inv)))
    }

    fn sum(&self, v: &[Ex]) -> Node {
        let mut numbers: Vec<Rat> = Vec::new();
        let mut terms: Vec<(bool, Node)> = Vec::new();
        for t in v {
            if let Ex::Num(r) = t {
                numbers.push(r.clone());
            } else if let Some(x) = self.fold(t) {
                numbers.push(x);
            } else {
                terms.push(odd_literal_hoist(split_neg(self.node(t)), self.cx));
            }
        }
        let constants: Vec<Rat> = match combine(&numbers, Rat::checked_add) {
            Some(c) if c.is_zero() => vec![],
            Some(c) => vec![c],
            None => numbers,
        };
        for c in &constants {
            terms.push(split_neg(self.number(c).0));
        }
        if terms.is_empty() {
            return Node::Num("0".to_string());
        }
        if terms.len() == 1 {
            let (neg, t) = terms.pop().unwrap();
            return if neg { negate(t) } else { t };
        }
        // Positive terms first, each group in canonical order.
        terms.sort_by_key(|(neg, _)| *neg);
        Node::Sum(terms)
    }

    /// Rule 6: the value of a ground subtree holding a function of numbers that are all written
    /// as floats, evaluated as the deployed evaluator does; `None` when it does not fold.
    fn fold(&self, e: &Ex) -> Option<Rat> {
        if matches!(e, Ex::Num(_)) || !self.foldable(e) {
            return None;
        }
        let mut pieces = Vec::new();
        collect_numbers(e, self.cx, &mut pieces);
        if pieces.is_empty() {
            return None;
        }
        for r in &pieces {
            if r.is_zero() || !matches!(self.spell(&r.abs(), false).0, Spelling::Float(_)) {
                return None;
            }
        }
        let y = evaluate(e, self.cx)?;
        if !writable(y) {
            return None;
        }
        Rat::shortest_reading_as(y)
    }

    /// Ground and non-rational: no variable, slot, constant or symbol, and a function inside.
    fn foldable(&self, e: &Ex) -> bool {
        fn ground(e: &Ex) -> bool {
            match e {
                Ex::Num(_) => true,
                Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN | Ex::Const | Ex::Leaf(_) => {
                    false
                }
                Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().all(ground),
                Ex::Pow(b, x) => ground(b) && ground(x),
            }
        }
        fn has_function(e: &Ex) -> bool {
            match e {
                Ex::Fun(..) => true,
                Ex::Pow(b, x) => !matches!(&**x, Ex::Num(r) if r.is_integer()) || has_function(b),
                Ex::Add(v) | Ex::Mul(v) => v.iter().any(has_function),
                _ => false,
            }
        }
        ground(e) && has_function(e)
    }
}

/// The numbers of a ground subtree that rule 6 reads: every literal except exponents and root
/// indices.
fn collect_numbers(e: &Ex, cx: &Cx, out: &mut Vec<Rat>) {
    match e {
        Ex::Num(r) => out.push(r.clone()),
        Ex::Add(v) | Ex::Mul(v) => v.iter().for_each(|x| collect_numbers(x, cx, out)),
        Ex::Fun(f, v) => {
            if cx.view.tok_is(*f, "rootn") {
                if let Some(x) = v.first() {
                    collect_numbers(x, cx, out);
                }
            } else {
                v.iter().for_each(|x| collect_numbers(x, cx, out));
            }
        }
        Ex::Pow(b, x) => {
            collect_numbers(b, cx, out);
            if !matches!(&**x, Ex::Num(_)) {
                collect_numbers(x, cx, out);
            }
        }
        _ => {}
    }
}

/// The f64 value of a ground subtree under the deployed evaluator's kernels: each literal is
/// the float64 nearest to it, then the operations run in float64 (libm for functions).
fn evaluate(e: &Ex, cx: &Cx) -> Option<f64> {
    match e {
        Ex::Num(r) => Some(r.to_f64_nearest()),
        Ex::Add(v) => v
            .iter()
            .try_fold(0.0, |acc, x| Some(acc + evaluate(x, cx)?)),
        Ex::Mul(v) => v
            .iter()
            .try_fold(1.0, |acc, x| Some(acc * evaluate(x, cx)?)),
        Ex::Pow(b, x) => {
            let (b, x) = (evaluate(b, cx)?, evaluate(x, cx)?);
            crate::numeric::apply_op("pow", &[b, x])
        }
        Ex::Fun(f, args) => {
            let name = cx.view.resolve_owned(*f);
            let vals: Option<Vec<f64>> = args.iter().map(|a| evaluate(a, cx)).collect();
            crate::numeric::apply_op(&name, &vals?)
        }
        _ => None,
    }
}

/// The exact combination of several numbers (`None` when it leaves the representable range or
/// float64's normal range: rule 10). An empty list combines to nothing.
fn combine(numbers: &[Rat], op: fn(&Rat, &Rat) -> Option<Rat>) -> Option<Rat> {
    let (first, rest) = numbers.split_first()?;
    let mut acc = first.clone();
    for r in rest {
        acc = op(&acc, r)?;
    }
    if numbers.len() > 1 && !acc.is_zero() && !writable(acc.to_f64_nearest()) {
        return None;
    }
    Some(acc)
}

/// The canonical infix's sign hoist for a sum term (owner ruling 2026-08-08): an odd function
/// of a negative literal gives its sign to the joiner (`x1 - sin(2)`, not `x1 + sin(-2)`).
fn odd_literal_hoist((neg, node): (bool, Node), cx: &Cx) -> (bool, Node) {
    if let Node::Call(f, args) = &node {
        if let [Node::Neg(inner)] = args.as_slice() {
            if matches!(**inner, Node::Num(_)) && cx.is_odd_fun(cx.view.intern(f)) {
                return (!neg, Node::Call(f.clone(), vec![(**inner).clone()]));
            }
        }
    }
    (neg, node)
}

/// `-node`, with the sign placed where each node keeps it.
fn negate(node: Node) -> Node {
    match node {
        Node::Neg(inner) => *inner,
        Node::Prod { neg, num, den } => Node::Prod {
            neg: !neg,
            num,
            den,
        },
        other => Node::Neg(Box::new(other)),
    }
}

/// `(is negative, magnitude)` of a term.
fn split_neg(node: Node) -> (bool, Node) {
    match node {
        Node::Neg(inner) => (true, *inner),
        Node::Prod {
            neg: true,
            num,
            den,
        } => (
            true,
            Node::Prod {
                neg: false,
                num,
                den,
            },
        ),
        other => (false, other),
    }
}

// ------------------------------------------------------------------------------------------
// The three output forms.

/// Precedence: 1 additive (and a leading sign), 2 multiplicative, 3 power, 4 atom.
fn infix_prec(node: &Node, out: &mut String) -> u8 {
    match node {
        Node::Num(s) => {
            out.push_str(s);
            4
        }
        Node::Name(s, _) => {
            out.push_str(s);
            4
        }
        Node::Neg(inner) => {
            out.push('-');
            infix_at(inner, 2, out);
            1
        }
        Node::Sum(terms) => {
            for (i, (neg, t)) in terms.iter().enumerate() {
                match (i, neg) {
                    (0, true) => out.push('-'),
                    (0, false) => {}
                    (_, true) => out.push_str(" - "),
                    (_, false) => out.push_str(" + "),
                }
                infix_at(t, 2, out);
            }
            1
        }
        Node::Prod { neg, num, den } => {
            if *neg {
                out.push('-');
            }
            if num.is_empty() {
                out.push('1');
            }
            for (i, f) in num.iter().enumerate() {
                if i > 0 {
                    out.push('*');
                }
                infix_at(f, 2, out);
            }
            for group in den {
                if let [one] = group.as_slice() {
                    out.push('/');
                    infix_at(one, 3, out);
                    continue;
                }
                out.push_str("/(");
                for (i, f) in group.iter().enumerate() {
                    if i > 0 {
                        out.push('*');
                    }
                    infix_at(f, 2, out);
                }
                out.push(')');
            }
            if *neg {
                1
            } else {
                2
            }
        }
        Node::Pow(b, x) => {
            infix_at(b, 4, out);
            out.push('^');
            infix_at(x, 4, out);
            3
        }
        Node::Call(f, args) => {
            out.push_str(f);
            out.push('(');
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                infix_at(a, 0, out);
            }
            out.push(')');
            4
        }
    }
}

fn infix_at(node: &Node, ctx: u8, out: &mut String) {
    let mut s = String::new();
    let prec = infix_prec(node, &mut s);
    if prec < ctx {
        out.push('(');
        out.push_str(&s);
        out.push(')');
    } else {
        out.push_str(&s);
    }
}

/// The infix text.
pub fn to_infix(node: &Node) -> String {
    let mut out = String::new();
    infix_at(node, 0, &mut out);
    out
}

/// The explicit binary prefix tokens: right-nested `*` chains, a left-nested chain of `+` and
/// `-`, `/ numerator denominator`, the sign in a leading numeral where there is one (`* -22 x1`)
/// and `neg` otherwise.
pub fn to_prefix(node: &Node) -> Vec<String> {
    let mut out = Vec::new();
    emit_prefix(node, &mut out);
    out
}

fn emit_prefix(node: &Node, out: &mut Vec<String>) {
    match node {
        Node::Num(s) => out.push(s.clone()),
        Node::Name(_, t) => out.push(t.clone()),
        Node::Neg(inner) => match &**inner {
            Node::Num(s) => out.push(format!("-{s}")),
            other => {
                out.push("neg".to_string());
                emit_prefix(other, out);
            }
        },
        Node::Sum(terms) if terms.iter().all(|(neg, _)| *neg) => {
            out.push("neg".to_string());
            let flipped: Vec<(bool, Node)> =
                terms.iter().map(|(_, t)| (false, t.clone())).collect();
            emit_prefix(&Node::Sum(flipped), out);
        }
        Node::Sum(terms) => {
            for (neg, _) in terms[1..].iter().rev() {
                out.push(if *neg { "-" } else { "+" }.to_string());
            }
            let (neg0, t0) = &terms[0];
            if *neg0 {
                emit_prefix(&Node::Neg(Box::new(t0.clone())), out);
            } else {
                emit_prefix(t0, out);
            }
            for (_, t) in &terms[1..] {
                emit_prefix(t, out);
            }
        }
        Node::Prod { neg, num, den } => {
            // `(num / g1) / g2 ...`: a left-nested division chain, one divisor per group.
            for _ in den {
                out.push("/".to_string());
            }
            emit_chain(num, *neg, out);
            for group in den {
                emit_chain(group, false, out);
            }
        }
        Node::Pow(b, x) => {
            out.push("pow".to_string());
            emit_prefix(b, out);
            emit_prefix(x, out);
        }
        Node::Call(f, args) => {
            out.push(f.clone());
            args.iter().for_each(|a| emit_prefix(a, out));
        }
    }
}

/// A right-nested `*` chain; `neg` signs its leading numeral, or wraps it in `neg`.
fn emit_chain(parts: &[Node], neg: bool, out: &mut Vec<String>) {
    let Some((first, rest)) = parts.split_first() else {
        out.push(if neg { "-1" } else { "1" }.to_string());
        return;
    };
    let signed = match (neg, first) {
        (true, Node::Num(s)) => Some(format!("-{s}")),
        (true, _) => {
            out.push("neg".to_string());
            None
        }
        (false, _) => None,
    };
    if !rest.is_empty() {
        out.push("*".to_string());
    }
    match signed {
        Some(s) => out.push(s),
        None => emit_prefix(first, out),
    }
    if !rest.is_empty() {
        emit_chain(rest, false, out);
    }
}

/// The tagged tokens: `<add> ... <sub> ... </add>` and `<mul> ... <div> ... </mul>` bags, the
/// sign of a product in a leading numeral (or a `-1` member), plain prefix for `pow` and
/// functions.
pub fn to_tagged(node: &Node) -> Vec<String> {
    let mut out = Vec::new();
    emit_tagged(node, &mut out);
    out
}

fn emit_tagged(node: &Node, out: &mut Vec<String>) {
    match node {
        Node::Num(s) => out.push(s.clone()),
        Node::Name(_, t) => out.push(t.clone()),
        Node::Neg(inner) => match &**inner {
            Node::Num(s) => out.push(format!("-{s}")),
            other => {
                out.push("neg".to_string());
                emit_tagged(other, out);
            }
        },
        Node::Sum(terms) => {
            if terms.iter().all(|(neg, _)| *neg) {
                out.push("neg".to_string());
                let flipped: Vec<(bool, Node)> =
                    terms.iter().map(|(_, t)| (false, t.clone())).collect();
                emit_tagged(&Node::Sum(flipped), out);
                return;
            }
            out.push("<add>".to_string());
            for (_, t) in terms.iter().filter(|(neg, _)| !neg) {
                emit_tagged(t, out);
            }
            if terms.iter().any(|(neg, _)| *neg) {
                out.push("<sub>".to_string());
                for (_, t) in terms.iter().filter(|(neg, _)| *neg) {
                    emit_tagged(t, out);
                }
            }
            out.push("</add>".to_string());
        }
        Node::Prod { neg, num, den } => {
            out.push("<mul>".to_string());
            match (neg, num.first()) {
                (true, Some(Node::Num(s))) => {
                    out.push(format!("-{s}"));
                    num[1..].iter().for_each(|f| emit_tagged(f, out));
                }
                (true, _) => {
                    out.push("-1".to_string());
                    num.iter().for_each(|f| emit_tagged(f, out));
                }
                (false, _) => {
                    if num.is_empty() {
                        out.push("1".to_string());
                    }
                    num.iter().for_each(|f| emit_tagged(f, out));
                }
            }
            if !den.is_empty() {
                out.push("<div>".to_string());
                // A section inverts each member on its own: every group's members join it.
                den.iter().flatten().for_each(|f| emit_tagged(f, out));
            }
            out.push("</mul>".to_string());
        }
        Node::Pow(b, x) => {
            out.push("pow".to_string());
            emit_tagged(b, out);
            emit_tagged(x, out);
        }
        Node::Call(f, args) => {
            out.push(f.clone());
            args.iter().for_each(|a| emit_tagged(a, out));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(p: i128, q: i128) -> Rat {
        Rat::new(p, q).unwrap()
    }

    fn fit(a: &Rat, d: usize, unit_free: bool) -> Option<String> {
        fraction_fit(a, d, unit_free).map(|f| match f.spelling() {
            (Some(n), None) => n,
            (n, Some(d)) => format!("{}/{d}", n.unwrap_or_else(|| "1".to_string())),
            (None, None) => unreachable!(),
        })
    }

    /// Rules 1 and 2: every digit counts, the shortest fraction spelling that fits is taken,
    /// a silent numerator of 1 breaks a tie beside a numerator factor, an integer denominator
    /// breaks the rest.
    #[test]
    fn fraction_spellings_count_every_digit() {
        assert_eq!(fit(&rat(1, 12), 3, true).as_deref(), Some("1/12"));
        assert_eq!(fit(&rat(1, 12), 2, true), None);
        assert_eq!(fit(&rat(3, 124), 3, false), None);
        assert_eq!(fit(&rat(3, 124), 4, false).as_deref(), Some("3/124"));
        assert_eq!(fit(&rat(5, 21), 3, true).as_deref(), Some("1/4.2"));
        assert_eq!(fit(&rat(5, 21), 3, false).as_deref(), Some("5/21"));
        assert_eq!(fit(&rat(10, 37), 3, true).as_deref(), Some("1/3.7"));
        assert_eq!(fit(&rat(22, 7), 3, true).as_deref(), Some("22/7"));
        assert_eq!(fit(&rat(1, 2), 3, true).as_deref(), Some("1/2"));
        assert_eq!(fit(&rat(1, 1024), 3, true), None);
        assert_eq!(fit(&rat(1024, 1), 3, true), None);
        assert_eq!(fit(&rat(999, 1), 3, true).as_deref(), Some("999"));
        // a denominator below 1 is never written: 1000/3 is not 1/0.003
        assert_eq!(fit(&rat(1000, 3), 9, true).as_deref(), Some("1000/3"));
    }

    /// A big value is searched within the bound `n < 10^(D-1)`, not over all its 2s and 5s.
    #[test]
    fn fraction_search_is_bounded_on_big_values() {
        let _exact = number_domain(false);
        let p = BigInt::from(5u32) * num_traits::pow(BigInt::from(2u32), 1000);
        let a = Rat::from_big(p, BigInt::from(21)).unwrap();
        assert_eq!(fit(&a, 3, true), None);
    }

    /// Rule 9: positional from 10^-3 (inclusive) to 10^6 (exclusive), else scientific.
    #[test]
    fn float_text_switches_to_scientific_outside_the_band() {
        let d = |s: &str| {
            let _exact = number_domain(false);
            Rat::parse_decimal(s).unwrap()
        };
        assert_eq!(float_text(&d("0.001")), "0.001");
        assert_eq!(float_text(&d("0.0009765625")), "9.765625e-4");
        assert_eq!(float_text(&d("999999")), "999999");
        assert_eq!(float_text(&d("1000000")), "1e6");
        assert_eq!(
            float_text(&d("26944002417373990000000000")),
            "2.694400241737399e25"
        );
        assert_eq!(float_text(&d("4.441529999999994")), "4.441529999999994");
        assert_eq!(float_text(&d("-0.5")), "-0.5");
    }

    /// The three forms of one display tree: `-22*x1/(7*x2)/(x1 + 1)`.
    #[test]
    fn one_tree_three_forms() {
        let x = |s: &str| Node::Name(s.to_string(), s.to_string());
        let n = |s: &str| Node::Num(s.to_string());
        let sum = Node::Sum(vec![(false, x("x1")), (false, n("1"))]);
        let tree = Node::Prod {
            neg: true,
            num: vec![n("22"), x("x1")],
            den: vec![vec![n("7"), x("x2")], vec![sum]],
        };
        assert_eq!(to_infix(&tree), "-22*x1/(7*x2)/(x1 + 1)");
        assert_eq!(to_prefix(&tree).join(" "), "/ / * -22 x1 * 7 x2 + x1 1");
        assert_eq!(
            to_tagged(&tree).join(" "),
            "<mul> -22 x1 <div> 7 x2 <add> x1 1 </add> </mul>"
        );
    }
}
