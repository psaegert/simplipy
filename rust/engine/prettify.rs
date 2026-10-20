//! The engine side of `prettify`: read an expression's canonical state in a mode's number domain
//! and display it (`ac::prettify`), then read the display back and display it again until it no
//! longer changes. It applies no rules, and nothing in the engine calls it.
//!
//! **Why the rounds.** The display writes a number as its float, so two parts of an expression
//! a hair apart can be written alike, and reading the display back through the canonical
//! constructors collects them: `sin(1.00000000000000000001*x1) - sin(x1)` is written
//! `sin(x1) - sin(x1)`, which reads as `0`; an exponent a hair above 3 is written `3`, and the
//! power then distributes over its base. `prettify` returns the display that is its own display,
//! so displaying a display gives it back by construction. The rounds run at full precision; with
//! a `precision`, the state they reach is displayed once more with its floats shortened, so that
//! display is not read back (a shortened float is not the number it stands for).
//!
//! **One round** writes the display of the state, renders it in the requested form, and reads
//! that rendering back exactly as `prettify` reads its input -- infix text through the strict
//! infix parse, tokens as they stand -- into a canonical state, by the canonical constructors
//! alone: no rule, no search, and no fold of a function (only the display folds a function of
//! floats, rule 6 of `ac::prettify`). The read-back computes in the mode's number domain, where a
//! product or power of numbers that would leave float64's normal range is refused and stays
//! apart, so it creates no 0, inf or subnormal by underflow or overflow; a 0 arises only from the
//! exact cancellation of terms written alike, which is what float64 computes for them.
//!
//! **Termination.** Write `S` for the state a round starts from. The display changes `S` only
//! through its numbers: it writes each number as itself or rounded to a float, and folds a
//! function of floats into one number; everything else it writes as `S` holds it, laid out as the
//! constructors read it back. So a round whose display changes no number's value and folds
//! nothing reads back to the state it displayed, and the next round writes the same display: the
//! rounds stop. Every other round strictly lowers, lexicographically,
//!
//! `m(S) = (I, L, F, U)`: `I` the powers of `S` with a numeric exponent that is not an integer,
//! `L` the occurrences of variables, symbols and fitted constants, `F` the function
//! applications, `U` the numbers whose written value is not their value.
//!
//! The constructors never duplicate a part: they flatten, sort, collect, compute exactly, and
//! distribute an integer power over a product factor by factor; they expand nothing. So no round
//! raises `L`, and a round raises no component without lowering an earlier one:
//! - an exponent rounded to an integer lowers `I` (the power may then distribute or compose,
//!   which raises neither `I` nor `L`);
//! - otherwise a fold of a function of floats lowers `F` (the folded part holds no variable);
//! - otherwise parts written alike are collected or cancel, which lowers `L` (collecting powers of
//!   one base merges their exponents, so `I` does not rise);
//! - otherwise only values change: each rounded number reads back as a number written as itself
//!   (a float as its shortest decimal, which reads back as the same float), and the constructors
//!   combine numbers only where parts are collected or a power distributes or composes (above),
//!   so `U` falls.
//!
//! `m` takes values in the naturals under the lexicographic order, which is well-founded, so the
//! rounds stop. [`PRETTIFY_MAX_ROUNDS`] stops them as well, only as defence in depth: a display
//! that reaches it is reported as not converged, and the test suite fails on any.

use super::ac::RuleMode;
use super::memo::SimplifyCtx;
use super::Engine;
use crate::ac::convert::from_prefix;
use crate::ac::expr::{canon, Cx};
use crate::ac::prettify::{to_infix, to_prefix, to_tagged, Display, Layout, Node};

/// The most displays `prettify` writes for one expression (defence in depth; see the module
/// documentation for why the rounds stop on their own).
pub const PRETTIFY_MAX_ROUNDS: usize = 32;

/// The output form of `prettify`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrettyForm {
    Infix,
    Explicit,
    Tagged,
}

impl PrettyForm {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "infix" => Some(PrettyForm::Infix),
            "explicit" => Some(PrettyForm::Explicit),
            "tagged" => Some(PrettyForm::Tagged),
            _ => None,
        }
    }

    /// The form's divisor layout: one denominator, or a tagged section.
    fn layout(self) -> Layout {
        match self {
            PrettyForm::Infix | PrettyForm::Explicit => Layout::OneDenominator,
            PrettyForm::Tagged => Layout::Sections,
        }
    }
}

/// A displayed expression: infix text or tokens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pretty {
    Text(String),
    Tokens(Vec<String>),
}

/// A display and the rounds that wrote it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrettyRounds {
    /// The display.
    pub shown: Pretty,
    /// The displays written, the last one included: 1 when the first display is its own.
    pub rounds: usize,
    /// Whether `shown` is its own display: `false` only where the round cap stopped the rounds
    /// (a cap of 1 asks for the first display alone) or a display did not read back.
    pub converged: bool,
}

fn render(node: &Node, form: PrettyForm) -> Pretty {
    match form {
        PrettyForm::Infix => Pretty::Text(to_infix(node)),
        PrettyForm::Explicit => Pretty::Tokens(to_prefix(node)),
        PrettyForm::Tagged => Pretty::Tokens(to_tagged(node)),
    }
}

impl Engine {
    /// The display tree of an expression's canonical state, read in `mode`'s number domain with
    /// the certificate-free context every projection uses. `None` on malformed input.
    fn display_node(
        &self,
        tokens: &[String],
        digits: usize,
        precision: Option<usize>,
        mode: RuleMode,
        layout: Layout,
    ) -> Option<Node> {
        let ctx = SimplifyCtx::new(self.tokens.len());
        let toks = self.intern_seq(tokens, &ctx);
        let view = self.view(&ctx);
        let bare = Cx::bare(&view);
        let _domain = crate::ac::rat::number_domain(!matches!(mode, RuleMode::Real));
        let state = canon(from_prefix(&toks, &bare)?, &bare);
        Some(Display::new(&bare, digits, precision, layout).node(&state))
    }

    /// One display of `tokens` in `form`.
    fn display_once(
        &self,
        tokens: &[String],
        digits: usize,
        precision: Option<usize>,
        form: PrettyForm,
        mode: RuleMode,
    ) -> Option<Pretty> {
        let node = self.display_node(tokens, digits, precision, mode, form.layout())?;
        Some(render(&node, form))
    }

    /// The tokens a display is read back from, as `prettify` reads its input: infix text through
    /// the strict infix parse, tokens as they stand.
    fn read_display(&self, shown: &Pretty) -> Option<Vec<String>> {
        match shown {
            Pretty::Text(s) => self.parse(s, true, false, true).ok(),
            Pretty::Tokens(t) => Some(t.clone()),
        }
    }

    /// `prettify`: the display of `tokens` (any dialect) under the digit budget `digits`, a
    /// float shown with `precision` significant digits (`None`: the shortest decimal of its
    /// float64), its numbers read in `mode`'s number domain -- the display that is its own
    /// display (see the module documentation). `None` on malformed input.
    pub fn ac_prettify(
        &self,
        tokens: &[String],
        digits: usize,
        precision: Option<usize>,
        form: PrettyForm,
        mode: RuleMode,
    ) -> Option<Pretty> {
        self.ac_prettify_rounds(tokens, digits, precision, form, mode, PRETTIFY_MAX_ROUNDS)
            .map(|r| r.shown)
    }

    /// [`Engine::ac_prettify`] with its rounds reported, writing at most `max_rounds` displays
    /// (at least one): display, read the display back, display again, until a display is its
    /// own display.
    ///
    /// The rounds run at full precision. A float shortened to `precision` digits is no longer
    /// the number it stands for, so reading it back would decide what the display only
    /// abbreviates: two terms that agree to three digits would cancel to a `0` float64 does not
    /// compute, and the shortened pi literal would read as another number. With a precision, the
    /// state the rounds reach is displayed once more, its floats shortened.
    pub fn ac_prettify_rounds(
        &self,
        tokens: &[String],
        digits: usize,
        precision: Option<usize>,
        form: PrettyForm,
        mode: RuleMode,
        max_rounds: usize,
    ) -> Option<PrettyRounds> {
        let cap = max_rounds.max(1);
        // `state`: the tokens of the state `shown` is the display of.
        let mut state = tokens.to_vec();
        let mut shown = self.display_once(&state, digits, None, form, mode)?;
        let mut rounds = 1;
        let mut converged = false;
        while rounds < cap {
            let Some(read) = self.read_display(&shown) else {
                break;
            };
            let Some(next) = self.display_once(&read, digits, None, form, mode) else {
                break;
            };
            if next == shown {
                state = read;
                converged = true;
                break;
            }
            (state, shown) = (read, next);
            rounds += 1;
        }
        if precision.is_some() {
            shown = self.display_once(&state, digits, precision, form, mode)?;
        }
        Some(PrettyRounds {
            shown,
            rounds,
            converged,
        })
    }
}
