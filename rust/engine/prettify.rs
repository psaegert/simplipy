//! The engine side of `prettify`: read an expression's canonical state in a mode's number domain
//! and display it (`ac::prettify`), writing every number so that the display reads back to the
//! state's structure. It applies no rules, and nothing in the engine calls it.
//!
//! **Numbers that would change the structure.** The display writes a number as its short
//! spelling: exactly where that fits the digit budget of significant digits, else as a short exact
//! sum where that fits (`x1 + 1 + 3e-21`, `(2.5e30 + 1e15)*x1`), which keeps its value, else as
//! its float (`k` significant digits at a precision). A short spelling that changes the number's value can change what
//! reading the display back builds: the number becomes 1 and a multiplier or an exponent drops,
//! an integer where a non-integer exponent stood, or equal to another number so that parts
//! collect, cancel or merge. Such a number is written as its short spelling and the rest to its
//! value, `s + r` or `s - r`, the rest written by the same rules (`(1 + 3.1e-21)*x1` for
//! `1.0000000000000000000031*x1`, `sin((1 + 1e-4)*x1) - sin(x1)` at precision 3). A number
//! written with pi factored out as the symbol is decided the same way, since the symbol reads
//! back as a multiple of pi, which the canonical constructors can build otherwise than the
//! number (`sin(x1 + pi)` is `-sin(x1)`): where it would change the structure, the literal
//! `3.141592653589793` is written.
//!
//! The condition is the one general test, not a list of cases: the display reads back through
//! the canonical constructors alone to a state whose SHAPE (`ac::prettify::shape`: the tree with
//! its numbers abstracted, pi counted as a number, and the display's own notation undone) is not
//! the shape of the state after the display's intended folds of functions of floats. Only where
//! a short spelling changes some number's value, or writes the symbol pi, is that read-back
//! made. Where the shapes differ, the numbers are
//! decided one at a time in the order the display writes them, each with every later one still
//! written at its value: its short spelling where the shape holds, else split, else (at a
//! precision) split with the rest in full, else split with the rest written exactly (which
//! writes the number at its value); a number written with the symbol pi has the literal as its
//! one other spelling. Of two numbers that the
//! short spellings make equal, the one written first keeps its short spelling and the other is
//! written as it plus its rest.

use super::ac::RuleMode;
use super::memo::SimplifyCtx;
use super::Engine;
use crate::ac::convert::from_prefix;
use crate::ac::expr::{canon, Cx};
use crate::ac::prettify::{
    shape, to_infix, to_prefix, to_tagged, Choice, Choices, Display, Layout, Node,
};
use crate::ac::rat::Rat;

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

/// How `prettify` writes a display: the digit budget, the significant digits a float shows
/// (`None`: the shortest decimal of its float64), the output form, and whether every sum and
/// product is written with the grouping of the explicit token form (`ac::prettify`, rule 10).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrettyOptions {
    pub digits: usize,
    pub precision: Option<usize>,
    pub form: PrettyForm,
    pub keep_grouping: bool,
}

/// A displayed expression: infix text or tokens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pretty {
    Text(String),
    Tokens(Vec<String>),
}

/// One display with every number written so that the structure stays.
struct Kept {
    shown: Pretty,
    /// Whether it reads back to the kept shape.
    kept: bool,
    /// The shape it keeps.
    reference: Option<String>,
    /// Whether it writes a number with pi factored out as the symbol, or a number where the
    /// state holds another expression (a fold): either reads back with its own place in the
    /// canonical order.
    reorders: bool,
}

fn render(node: &Node, form: PrettyForm, grouped: bool) -> Pretty {
    match form {
        PrettyForm::Infix => Pretty::Text(to_infix(node, grouped)),
        PrettyForm::Explicit => Pretty::Tokens(to_prefix(node)),
        PrettyForm::Tagged => Pretty::Tokens(to_tagged(node)),
    }
}

impl Engine {
    /// The shape of the state a display reads back as, read exactly as `prettify` reads its
    /// input: infix text through the strict infix parse, tokens as they stand, then the
    /// canonical constructors in `mode`'s number domain.
    fn read_shape(&self, shown: &Pretty, mode: RuleMode) -> Option<String> {
        let tokens = self.read_display(shown)?;
        let ctx = SimplifyCtx::new(self.tokens.len());
        let toks = self.intern_seq(&tokens, &ctx);
        let view = self.view(&ctx);
        let bare = Cx::bare(&view);
        let _domain = crate::ac::rat::number_domain(!matches!(mode, RuleMode::Real));
        let state = canon(from_prefix(&toks, &bare)?, &bare);
        Some(shape(&state, &bare))
    }

    /// `prettify`: the display of `tokens` (any dialect) as `options` say (the digit budget, a
    /// float's significant digits, the form, the grouping of sums), its numbers read in `mode`'s
    /// number domain, every number written so that the display keeps the structure (see the
    /// module documentation). `None` on malformed input.
    pub fn ac_prettify(
        &self,
        tokens: &[String],
        options: PrettyOptions,
        mode: RuleMode,
    ) -> Option<Pretty> {
        self.ac_prettify_structure(tokens, options, mode, false)
            .map(|(shown, _)| shown)
    }

    /// [`Engine::ac_prettify`] and whether its display reads back to the shape of the state
    /// after the display's folds. Where no number changes its value that is not checked unless
    /// `verify` asks (and answers `true`).
    ///
    /// At a precision the display is of the state its shortened numbers read back as. Rounding to
    /// `k` digits can change the order in which reading puts the factors of a product or the
    /// orientation it gives a sum (`(1 - 27/(4*x1))*(27*log(x1)/4 - 91/2)` reads back as
    /// `(91/2 - 27*log(x1)/4)*(27/(4*x1) - 1)`), so the display of that state, whose shape is
    /// checked to be the same, is the one that is its own display. The symbol pi does the same
    /// at every precision: it reads back as a factor, with its own place in the canonical order
    /// of a sum's terms (`3.141592653589793*x2 + sin(x1)` reads back from `pi*x2 + sin(x1)` as
    /// `sin(x1) + pi*x2`). So does a number written where the state holds another expression, a
    /// function of floats folded or numbers combined at their floats: `cos(c) - x2` folds to
    /// `-x2 - 0.997`, which reads back as `-(x2 + 0.997)`. At full precision with neither,
    /// rounding to the nearest float keeps every order, and the display needs no second look.
    pub fn ac_prettify_structure(
        &self,
        tokens: &[String],
        options: PrettyOptions,
        mode: RuleMode,
        verify: bool,
    ) -> Option<(Pretty, bool)> {
        let first = self.display_keeping(tokens, options, mode, verify)?;
        if (options.precision.is_none() && !first.reorders) || !first.kept {
            return Some((first.shown, first.kept));
        }
        let Some(read) = self.read_display(&first.shown) else {
            return Some((first.shown, first.kept));
        };
        match self.display_keeping(&read, options, mode, true) {
            Some(again) if again.kept && self.read_shape(&again.shown, mode) == first.reference => {
                Some((again.shown, true))
            }
            _ => Some((first.shown, first.kept)),
        }
    }

    /// The tokens a display is read back from, as `prettify` reads its input: infix text through
    /// the strict infix parse, tokens as they stand.
    fn read_display(&self, shown: &Pretty) -> Option<Vec<String>> {
        match shown {
            Pretty::Text(s) => self.parse(s, true, false, true).ok(),
            Pretty::Tokens(t) => Some(t.clone()),
        }
    }

    /// One display of `tokens` with every number written so that the structure stays, whether
    /// it does, and the shape it keeps.
    fn display_keeping(
        &self,
        tokens: &[String],
        options: PrettyOptions,
        mode: RuleMode,
        verify: bool,
    ) -> Option<Kept> {
        let PrettyOptions {
            digits,
            precision,
            form,
            keep_grouping,
        } = options;
        let ctx = SimplifyCtx::new(self.tokens.len());
        let toks = self.intern_seq(tokens, &ctx);
        let view = self.view(&ctx);
        let bare = Cx::bare(&view);
        let _domain = crate::ac::rat::number_domain(!matches!(mode, RuleMode::Real));
        let state = canon(from_prefix(&toks, &bare)?, &bare);
        let pass = |choices: &Choices| -> (Pretty, Vec<Rat>, bool) {
            let display = Display::new(&bare, digits, precision, form.layout())
                .with_choices(choices.clone())
                .with_grouping(keep_grouping);
            let node = display.node(&state);
            (
                render(&node, form, keep_grouping),
                display.changed(),
                display.pi_symbol_written() || display.folds(),
            )
        };
        let reference = Some(
            Display::new(&bare, digits, precision, form.layout())
                .with_choices(Choices::all(Choice::Exact))
                .kept_shape(&state),
        );
        let kept = |p: &Pretty| self.read_shape(p, mode) == reference;
        // With the grouping kept at full precision the display evaluates as the state does: its
        // sums and products in the state's order (rule 10) and every number as the state's
        // tokens write it, exactly. A float one rounding from a quotient the tokens divide by
        // (`sin(3.1415926535897e37*x1*x2/3313)`) would not, and a function that amplifies its
        // argument shows the rounding.
        let first = if keep_grouping && precision.is_none() {
            Choices::all(Choice::Exact)
        } else {
            Choices::default()
        };
        let (shown, changed, reorders) = pass(&first);
        if changed.is_empty() {
            // Every number is written at its value.
            let held = !verify || kept(&shown);
            return Some(Kept {
                shown,
                kept: held,
                reference,
                reorders,
            });
        }
        if kept(&shown) {
            return Some(Kept {
                shown,
                kept: true,
                reference,
                reorders,
            });
        }
        let mut choices = Choices::all(Choice::Exact);
        // At full precision a split's rest is already written in full.
        let order: &[Choice] = if precision.is_some() {
            &[
                Choice::Short,
                Choice::Split,
                Choice::SplitFull,
                Choice::SplitExact,
            ]
        } else {
            &[Choice::Short, Choice::Split, Choice::SplitExact]
        };
        for a in changed {
            for &choice in order {
                choices.set(a.clone(), choice);
                if choice == Choice::SplitExact || kept(&pass(&choices).0) {
                    break;
                }
            }
        }
        let (shown, _, reorders) = pass(&choices);
        let held = kept(&shown);
        Some(Kept {
            shown,
            kept: held,
            reference,
            reorders,
        })
    }
}
