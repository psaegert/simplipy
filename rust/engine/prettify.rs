//! The engine side of `prettify` (number plan v2, phase 3): read the canonical state, display
//! it (`ac::prettify`), and -- rule 11 -- simplify the display once and display that answer.
//!
//! Rule 11 is why this lives with the engine: the display rounds, and a rounded value can make
//! another form cheaper (`0.99999999999999999999999999999667` shows as `1`, and the sum it sits
//! in then multiplies out). The display of the re-simplified display is the answer. The
//! dependency points one way: prettify calls simplify; nothing in the engine calls prettify.

use super::ac::{AcForm, RuleMode};
use super::memo::SimplifyCtx;
use super::Engine;
use crate::ac::convert::from_prefix;
use crate::ac::expr::{canon, Cx};
use crate::ac::prettify::{to_infix, to_prefix, to_tagged, Display, Node};

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
}

/// A displayed expression: infix text or tokens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pretty {
    Text(String),
    Tokens(Vec<String>),
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
    fn display_node(&self, tokens: &[String], digits: usize, mode: RuleMode) -> Option<Node> {
        let ctx = SimplifyCtx::new(self.tokens.len());
        let toks = self.intern_seq(tokens, &ctx);
        let view = self.view(&ctx);
        let bare = Cx::bare(&view);
        let _domain = crate::ac::rat::number_domain(!matches!(mode, RuleMode::Real));
        let state = canon(from_prefix(&toks, &bare)?, &bare);
        Some(Display::new(&bare, digits).node(&state))
    }

    /// `prettify`: the display of `tokens` (any dialect) under the digit budget `digits`.
    /// `resimplify` is rule 11's search budget (`None`: rule 11 off): the display is simplified
    /// once in `mode` and the answer's display is returned. `None` on malformed input.
    pub fn ac_prettify(
        &self,
        tokens: &[String],
        digits: usize,
        form: PrettyForm,
        mode: RuleMode,
        resimplify: Option<usize>,
    ) -> Option<Pretty> {
        let node = self.display_node(tokens, digits, mode)?;
        let Some(budget) = resimplify else {
            return Some(render(&node, form));
        };
        let shown = to_prefix(&node);
        let again = if budget > 0 {
            self.ac_explore_proj(&shown, 48, mode, AcForm::Explicit, budget)
        } else {
            self.ac_simplify_proj(&shown, 48, mode, AcForm::Explicit)
        };
        // The display re-reads by construction (a test pins it); should it ever not, the
        // display of the state itself stands.
        let node = again
            .and_then(|t| self.display_node(&t, digits, mode))
            .unwrap_or(node);
        Some(render(&node, form))
    }
}
