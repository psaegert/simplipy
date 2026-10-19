//! `prettify`: the DISPLAY of a canonical state.
//!
//! The state stays exact; this module only decides how a reader sees it. It is a pure function
//! of the canonical expression and a digit budget `D`, and nothing inside the engine calls it or
//! reads what it writes (a source guard in the test suite holds that line). The display keeps
//! the state's structure -- its bags, divisors and function applications -- and decides only
//! how each NUMBER is written. It evaluates nothing and simplifies nothing: a function of
//! numbers stays a function (`x1*exp(-190.5)`), and what the display writes alike is not
//! collected.
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
//!    joins the denominator (`x1/(243*x2)`); only a bare `x/float` becomes a multiplier
//!    (`x1/243` is `0.00411522633744856*x1`).
//! 6. **Exponents** are numbers like any other.
//! 7. **Even roots**: `x^(1/n)` is `rootn(x, n)` and `x^(-1/n)` is `1/rootn(x, n)` for even `n`.
//! 8. **Layout**: one coefficient in front and ONE SLASH: every divisor of a product joins a
//!    single denominator product (`x1/(x2*exp(x3))`, `x1/(sin(x3)*(x2 + 1))`). The text
//!    `n/(d1*d2)` reads back as one joined reciprocal, so the display writes the product the
//!    canonical constructors build from the state's factors with every reciprocal joined into
//!    one -- the product reading the display back builds -- whether the state held a reciprocal
//!    per factor or a joined one (`(x2*(x3 + 1))^-1`). The joined reciprocal's number divides
//!    the coefficient and its sign stays in the denominator (`x1/(-(x2 + x3))`); beside a factor
//!    that absorbs a coefficient (a fitted constant, an infinity) its number stays there too.
//!    The denominator's members are in canonical factor order. The tagged section inverts each
//!    member on its own, so there a joined reciprocal stays one bag unless its members are
//!    numbers, variables and their powers and roots. Scientific notation from 10^6 and below
//!    10^-3 with a lowercase `e`.
//! 9. **Range**: no numbers combine to inf, 0 or a subnormal.

use std::cell::Cell;
use std::cmp::Ordering;
use std::sync::OnceLock;

use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};

use super::expr::{add, fun, mul, mul_factor_cmp, negate_term, pow, Cx, Ex};
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
    /// A product `±(num...)/(±den...)`; an empty numerator is `1`, an empty denominator none.
    /// `den_neg` is a sign the denominator holds (`x1/(-(x2 + x3))`).
    Prod {
        neg: bool,
        num: Vec<Node>,
        den: Vec<Node>,
        den_neg: bool,
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
    /// The exact values of the numerator and of the denominator it writes, given the `value`
    /// the whole spelling reads back as.
    fn values(&self, value: &Rat) -> Option<(Rat, Rat)> {
        let _exact = number_domain(false);
        let part = |s: &Option<String>| match s {
            Some(t) => Rat::parse_decimal(t),
            None => Some(Rat::ONE),
        };
        Some(match self {
            Spelling::Int(_) | Spelling::Float(_) => (value.clone(), Rat::ONE),
            Spelling::Frac { num, den } => (part(num)?, Rat::parse_decimal(den)?),
            Spelling::PiNum { num, den } => (part(num)?.checked_mul(pi_value())?, part(den)?),
            Spelling::PiDen { num, den } => (part(num)?, part(den)?.checked_mul(pi_value())?),
        })
    }

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

/// The float form of an exact decimal `s` (rule 8): positional between 10^-3 (inclusive) and
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

/// Is `y` a float the display may write: finite, nonzero and normal (rule 9)?
fn writable(y: f64) -> bool {
    y.is_finite() && y.abs() >= f64::MIN_POSITIVE
}

/// How a product's divisors are laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// One denominator product for every divisor (infix text and explicit tokens).
    OneDenominator,
    /// A tagged section, which inverts each member on its own.
    Sections,
}

/// The DISPLAY of one canonical state under a digit budget.
pub struct Display<'a, 'v> {
    cx: &'a Cx<'v>,
    digits: usize,
    layout: Layout,
    /// Set while rebuilding when a sum keeps terms apart that its rebuild would collect.
    kept_apart: Cell<bool>,
}

impl<'a, 'v> Display<'a, 'v> {
    pub fn new(cx: &'a Cx<'v>, digits: usize, layout: Layout) -> Self {
        Display {
            cx,
            digits,
            layout,
            kept_apart: Cell::new(false),
        }
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
            // Rule 9: no float can stand for it (only exact-domain values get here).
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
                den,
                den_neg: false,
            }
        };
        if r.is_negative() {
            (negate(node), value.checked_neg().unwrap_or(value))
        } else {
            (node, value)
        }
    }

    /// The display of a canonical expression. With one denominator it is the display of the
    /// expression reading that display back builds ([`Display::rebuild`]).
    pub fn node(&self, e: &Ex) -> Node {
        match self.layout {
            Layout::OneDenominator => self.write(&self.rebuild(e)),
            Layout::Sections => self.write(e),
        }
    }

    /// The expression reading its one-denominator display back builds: bottom up, every product
    /// with one denominator ([`Display::one_denominator`]), every function, power and sum built
    /// again by the canonical constructors. A sum whose terms that rebuild would collect keeps
    /// its terms apart, as the expression holds them: the display collects nothing.
    fn rebuild(&self, e: &Ex) -> Ex {
        let cx = self.cx;
        match e {
            Ex::Fun(f, args) => fun(*f, args.iter().map(|a| self.rebuild(a)).collect(), cx),
            Ex::Pow(b, x) if positive(e, cx).is_some() => {
                let r = Ex::Pow(Box::new(self.rebuild(b)), x.clone());
                self.one_denominator(std::slice::from_ref(&r)).unwrap_or(r)
            }
            Ex::Pow(b, x) => pow(self.rebuild(b), self.rebuild(x), cx),
            Ex::Mul(v) => {
                let outer = self.kept_apart.replace(false);
                let mut factors: Vec<Ex> = Vec::new();
                for f in v {
                    // A reciprocal factor is laid out with the product it divides.
                    let rebuilt = match f {
                        Ex::Pow(b, x) if positive(f, cx).is_some() => {
                            Ex::Pow(Box::new(self.rebuild(b)), x.clone())
                        }
                        _ => self.rebuild(f),
                    };
                    match rebuilt {
                        Ex::Mul(inner) => factors.extend(inner),
                        other => factors.push(other),
                    }
                }
                // Reading a sum kept apart would collect it: such a product stays as it is.
                let apart = self.kept_apart.get();
                self.kept_apart.set(outer || apart);
                let laid = if apart {
                    None
                } else {
                    self.one_denominator(&factors)
                };
                laid.unwrap_or_else(|| mul(factors, cx))
            }
            Ex::Add(v) => {
                let terms: Vec<Ex> = v.iter().map(|t| self.rebuild(t)).collect();
                let built = add(terms.clone(), cx);
                let kept = match &built {
                    Ex::Add(t) => t.len(),
                    Ex::Num(z) if z.is_zero() => 0,
                    _ => 1,
                };
                if kept == terms.len() {
                    built
                } else {
                    self.kept_apart.set(true);
                    Ex::Add(terms)
                }
            }
            other => other.clone(),
        }
    }

    /// The display of an expression as it stands.
    fn write(&self, e: &Ex) -> Node {
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
                args.iter().map(|a| self.write(a)).collect(),
            ),
            // A power is a product of one member, laid out as every product is.
            Ex::Pow(..) => self.product(std::slice::from_ref(e)),
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
                Node::Pow(Box::new(self.write(b)), Box::new(self.write(x))),
            );
        }
        (false, self.write(f))
    }

    /// `b^k` for a positive exponent as displayed: `b` alone for 1, an even root for `1/n`.
    fn power(&self, b: &Ex, k: &Rat) -> Node {
        if k.is_one() {
            return self.write(b);
        }
        if k.numer().is_one() && k.denom().is_even_integer() {
            return Node::Call(
                "rootn".to_string(),
                vec![self.write(b), Node::Num(k.denom().numer_string())],
            );
        }
        Node::Pow(Box::new(self.write(b)), Box::new(self.number(k).0))
    }

    /// The display of the product `v` as it stands.
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
        // A fitted constant or an infinity absorbs a coefficient beside it, so a joined
        // reciprocal's number cannot move out to the coefficient there.
        let absorbs = v.iter().any(absorbs_a_coefficient);
        let mut numbers: Vec<Rat> = Vec::new();
        // The numbers of the joined reciprocals taken apart: they divide the coefficient.
        let mut divisors: Vec<Rat> = Vec::new();
        let mut num: Vec<Node> = Vec::new();
        let mut den: Vec<(Ex, Node)> = Vec::new();
        let mut den_neg = false;
        for f in v {
            if let Ex::Num(r) = f {
                numbers.push(r.clone());
            } else if let Some(members) = joined_reciprocal(f).filter(|m| self.takes_apart(m)) {
                for m in members {
                    match m {
                        Ex::Num(r) if absorbs && self.layout == Layout::OneDenominator => {
                            den_neg ^= r.is_negative();
                            den.push((m.clone(), self.number(&r.abs()).0));
                        }
                        Ex::Num(r) => divisors.push(r.clone()),
                        _ => {
                            // A divisor of the joined product multiplies: `1/(a/b)` is `b/a`.
                            let (is_den, node) = self.factor(m);
                            if is_den {
                                num.push(node);
                            } else {
                                den.push((m.clone(), node));
                            }
                        }
                    }
                }
            } else {
                let (is_den, node) = self.factor(f);
                match positive(f, self.cx).filter(|_| is_den) {
                    Some(key) => den.push((key, node)),
                    None => num.push(node),
                }
            }
        }
        let view = self.cx.view;
        den.sort_by(|(a, _), (b, _)| mul_factor_cmp(a, b, view));
        let den: Vec<Node> = den.into_iter().map(|(_, node)| node).collect();
        // In one denominator a joined reciprocal's sign stays where the state holds it.
        if self.layout == Layout::OneDenominator {
            den_neg ^= divisors.iter().filter(|r| r.is_negative()).count() % 2 == 1;
            divisors = divisors.iter().map(Rat::abs).collect();
        }
        let mut neg = false;
        let mut lead: Vec<Node> = Vec::new();
        let mut lead_den: Vec<Node> = Vec::new();
        match coefficient(&numbers, &divisors) {
            Some(c) => {
                neg = c.is_negative();
                let (sp, value) = self.spell(&c.abs(), !num.is_empty());
                if !value.is_one() {
                    match self.divisor(&sp, &c.abs(), !den.is_empty()) {
                        Some((d, _)) => lead_den.push(d),
                        None => {
                            let (n, d) = sp.parts();
                            lead = n;
                            lead_den = d;
                        }
                    }
                }
            }
            None => {
                // Numbers whose product leaves the range (rule 9): each its own.
                for r in &numbers {
                    let (s, node) = split_neg(self.number(r).0);
                    neg ^= s;
                    lead.push(node);
                }
                for r in &divisors {
                    let (s, node) = split_neg(self.number(r).0);
                    neg ^= s;
                    lead_den.push(node);
                }
            }
        }
        lead.extend(num);
        lead_den.extend(den);
        if lead_den.is_empty() && lead.len() == 1 && !den_neg {
            let only = lead.pop().unwrap();
            return if neg { negate(only) } else { only };
        }
        Node::Prod {
            neg,
            num: lead,
            den: lead_den,
            den_neg,
        }
    }

    /// The product `v` with one denominator: the product the canonical constructors build from
    /// the display of `v` -- its factors, the coefficient as written, and every reciprocal (its
    /// own `b^-k` and the members of a joined one, `(m1*m2)^-1`) in one joined reciprocal --
    /// which is the product reading the display back builds. `None` when `v` has no reciprocal.
    fn one_denominator(&self, v: &[Ex]) -> Option<Ex> {
        let cx = self.cx;
        let absorbs = v.iter().any(absorbs_a_coefficient);
        let mut numbers: Vec<Rat> = Vec::new();
        let mut divisors: Vec<Rat> = Vec::new();
        let mut items: Vec<Ex> = Vec::new();
        let mut den: Vec<Ex> = Vec::new();
        for f in v {
            if let Ex::Num(r) = f {
                numbers.push(r.clone());
                continue;
            }
            match positive(f, cx) {
                Some(Ex::Mul(members)) => {
                    for m in members {
                        match (&m, positive(&m, cx)) {
                            (Ex::Num(r), _) if !absorbs => divisors.push(r.clone()),
                            // A divisor of the joined product multiplies: `1/(a/b)` is `b/a`.
                            (_, Some(p)) => items.push(p),
                            _ => den.push(m),
                        }
                    }
                }
                Some(p) => den.push(p),
                None => items.push(f.clone()),
            }
        }
        if den.is_empty() && divisors.is_empty() {
            return None;
        }
        let den_neg = divisors.iter().filter(|r| r.is_negative()).count() % 2 == 1;
        let magnitudes: Vec<Rat> = divisors.iter().map(Rat::abs).collect();
        match coefficient(&numbers, &magnitudes) {
            Some(c) => {
                // The coefficient as the display writes it: its numerator beside the factors,
                // its denominator in the joined reciprocal, which keeps the sign it holds.
                let (top, bottom) = self.written(&c.abs(), !items.is_empty(), !den.is_empty())?;
                items.push(Ex::Num(if c.is_negative() {
                    top.checked_neg()?
                } else {
                    top
                }));
                den.push(Ex::Num(if den_neg {
                    bottom.checked_neg()?
                } else {
                    bottom
                }));
            }
            None => {
                items.extend(numbers.into_iter().map(Ex::Num));
                den.extend(divisors.into_iter().map(Ex::Num));
            }
        }
        items.push(pow(mul(den, cx), Ex::Num(Rat::NEG_ONE), cx));
        Some(mul(items, cx))
    }

    /// The exact numerator and denominator a coefficient `a > 0` is written with: its spelling's
    /// parts, or rule 5's divisor.
    fn written(&self, a: &Rat, unit_free: bool, has_den: bool) -> Option<(Rat, Rat)> {
        let (sp, value) = self.spell(a, unit_free);
        if value.is_one() {
            return Some((Rat::ONE, Rat::ONE));
        }
        if let Some((_, inv)) = self.divisor(&sp, a, has_den) {
            return Some((Rat::ONE, inv));
        }
        sp.values(&value)
    }

    /// Is a joined reciprocal with these members taken apart into the denominator? Always with
    /// one denominator; in a tagged section, which inverts each member on its own, only when its
    /// members are numbers, variables and their powers and roots.
    fn takes_apart(&self, members: &[Ex]) -> bool {
        self.layout == Layout::OneDenominator || self.group_members(members)
    }

    /// Are the members of a joined reciprocal all numbers, and positive powers and roots of
    /// variables and symbols?
    fn group_members(&self, members: &[Ex]) -> bool {
        members.iter().all(|m| match m {
            Ex::Num(_) => true,
            Ex::Pow(b, x) => matches!(&**x, Ex::Num(k) if !k.is_negative()) && self.groupable(b),
            other => self.groupable(other),
        })
    }

    /// Is `b` a variable or a symbol, or an even root of one?
    fn groupable(&self, b: &Ex) -> bool {
        let base = match b {
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

    /// Rule 5: a float coefficient whose reciprocal float64 holds as written joins the
    /// denominator as that decimal, when the product has one.
    fn divisor(&self, sp: &Spelling, mag: &Rat, has_den: bool) -> Option<(Node, Rat)> {
        let Spelling::Float(_) = sp else { return None };
        if !has_den || mag.is_integer() {
            return None;
        }
        let _exact = number_domain(false);
        let inv = mag.checked_inv()?;
        inv.decimal_parts()?;
        let y = inv.to_f64_nearest();
        (writable(y) && Rat::shortest_reading_as(y).as_ref() == Some(&inv))
            .then(|| (Node::Num(float_text(&inv)), inv))
    }

    fn sum(&self, v: &[Ex]) -> Node {
        let mut numbers: Vec<Rat> = Vec::new();
        let mut terms: Vec<(bool, Node)> = Vec::new();
        for t in v {
            if let Ex::Num(r) = t {
                numbers.push(r.clone());
            } else {
                terms.push(odd_literal_hoist(split_neg(self.write(t)), self.cx));
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
}

/// The exact combination of several numbers (`None` when it leaves the representable range or
/// float64's normal range: rule 9). An empty list combines to nothing.
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

/// The coefficient of a product: its numbers over the numbers of its joined reciprocals, exact
/// (`None` when that leaves the range, as in [`combine`]). No numbers is no coefficient.
fn coefficient(numbers: &[Rat], divisors: &[Rat]) -> Option<Rat> {
    let top = combine(numbers, Rat::checked_mul);
    if divisors.is_empty() {
        return top;
    }
    if top.is_none() && !numbers.is_empty() {
        return None;
    }
    let bottom = combine(divisors, Rat::checked_mul)?;
    let c = top
        .unwrap_or(Rat::ONE)
        .checked_mul(&bottom.checked_inv()?)?;
    (c.is_zero() || writable(c.to_f64_nearest())).then_some(c)
}

/// The members of a joined reciprocal `(m1*m2*...)^-1`: the reciprocal of a product the state
/// holds whole.
fn joined_reciprocal(f: &Ex) -> Option<&[Ex]> {
    match f {
        Ex::Pow(b, x) => match (&**b, &**x) {
            (Ex::Mul(members), Ex::Num(k)) if *k == Rat::NEG_ONE => Some(members),
            _ => None,
        },
        _ => None,
    }
}

/// Does `f` absorb a coefficient beside it (a fitted constant, an infinity, nan)?
fn absorbs_a_coefficient(f: &Ex) -> bool {
    matches!(f, Ex::Const | Ex::PosInf | Ex::NegInf | Ex::NaN)
}

/// `b^k` for a reciprocal `b^-k` (a negative numeric exponent), built canonically.
fn positive(f: &Ex, cx: &Cx) -> Option<Ex> {
    let Ex::Pow(b, x) = f else { return None };
    let Ex::Num(k) = &**x else { return None };
    if !k.is_negative() || matches!(&**b, Ex::Num(z) if z.is_zero()) {
        return None;
    }
    let k = k.checked_neg()?;
    Some(if k.is_one() {
        (**b).clone()
    } else {
        pow((**b).clone(), Ex::Num(k), cx)
    })
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
        Node::Prod {
            neg,
            num,
            den,
            den_neg,
        } => Node::Prod {
            neg: !neg,
            num,
            den,
            den_neg,
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
            den_neg,
        } => (
            true,
            Node::Prod {
                neg: false,
                num,
                den,
                den_neg,
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
        Node::Prod {
            neg,
            num,
            den,
            den_neg,
        } => {
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
            match (den.as_slice(), den_neg) {
                ([], _) => {}
                ([one], false) => {
                    out.push('/');
                    infix_at(one, 3, out);
                }
                (all, _) => {
                    out.push_str(if *den_neg { "/(-" } else { "/(" });
                    for (i, f) in all.iter().enumerate() {
                        if i > 0 {
                            out.push('*');
                        }
                        infix_at(f, 2, out);
                    }
                    out.push(')');
                }
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
/// `-`, one `/ numerator denominator`, the sign in a leading numeral where there is one
/// (`* -22 x1`) and `neg` otherwise.
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
        Node::Prod {
            neg,
            num,
            den,
            den_neg,
        } => {
            if !den.is_empty() {
                out.push("/".to_string());
            }
            emit_chain(num, *neg, out);
            if !den.is_empty() {
                emit_chain(den, *den_neg, out);
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
        Node::Prod {
            neg,
            num,
            den,
            den_neg,
        } => {
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
                if *den_neg {
                    out.push("-1".to_string());
                }
                den.iter().for_each(|f| emit_tagged(f, out));
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

    /// The three forms of one display tree: `-22*x1/(7*x2*(x1 + 1))`, one slash.
    #[test]
    fn one_tree_three_forms() {
        let x = |s: &str| Node::Name(s.to_string(), s.to_string());
        let n = |s: &str| Node::Num(s.to_string());
        let sum = Node::Sum(vec![(false, x("x1")), (false, n("1"))]);
        let tree = |den_neg| Node::Prod {
            neg: true,
            num: vec![n("22"), x("x1")],
            den: vec![n("7"), x("x2"), sum.clone()],
            den_neg,
        };
        assert_eq!(to_infix(&tree(false)), "-22*x1/(7*x2*(x1 + 1))");
        assert_eq!(
            to_prefix(&tree(false)).join(" "),
            "/ * -22 x1 * 7 * x2 + x1 1"
        );
        assert_eq!(
            to_tagged(&tree(false)).join(" "),
            "<mul> -22 x1 <div> 7 x2 <add> x1 1 </add> </mul>"
        );
        // a sign the denominator holds
        assert_eq!(to_infix(&tree(true)), "-22*x1/(-7*x2*(x1 + 1))");
        assert_eq!(
            to_prefix(&tree(true)).join(" "),
            "/ * -22 x1 * -7 * x2 + x1 1"
        );
    }
}
