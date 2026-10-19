//! The engine side of `prettify`: read an expression's canonical state in a mode's number domain
//! and display it (`ac::prettify`). It simplifies nothing, and nothing in the engine calls it.

use super::ac::RuleMode;
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

    /// `prettify`: the display of `tokens` (any dialect) under the digit budget `digits`, its
    /// numbers read in `mode`'s number domain. `None` on malformed input.
    pub fn ac_prettify(
        &self,
        tokens: &[String],
        digits: usize,
        form: PrettyForm,
        mode: RuleMode,
    ) -> Option<Pretty> {
        let node = self.display_node(tokens, digits, mode)?;
        Some(render(&node, form))
    }
}
