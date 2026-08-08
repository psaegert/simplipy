//! Rule storage: the loaded ruleset, interned into the per-Engine [`TokenTable`] in asset
//! (rules.json) order.
//!
//! The AC engine is the only consumer: `CompiledRules::raw` preserves the global
//! first-match-wins order for the lazy AC translation (`ac::rules::AcRules::translate`).
//! Every rule token is interned into the table at compile, so rule tokens are always TABLE
//! ids -- a query token that only exists in a per-call overlay can, by injectivity, never
//! equal any rule token.

use crate::operators::Operators;
use crate::tokens::{Tok, TokenTable};

/// The loaded ruleset, interned. `raw` is every rule in ASSET ORDER as (lhs, rhs) token
/// sequences: the AC translation needs the global first-match-wins order intact.
#[derive(Debug, Default)]
pub struct CompiledRules {
    pub raw: Vec<(Vec<Tok>, Vec<Tok>)>,
}

impl CompiledRules {
    /// Intern the raw (lhs, rhs) prefix pairs from rules.json into the (per-Engine,
    /// `&mut` here) token table, preserving asset order.
    pub fn compile(
        raw: Vec<(Vec<String>, Vec<String>)>,
        table: &mut TokenTable,
        ops: &Operators,
    ) -> Self {
        let raw_interned = raw
            .into_iter()
            .map(|(lhs, rhs)| {
                (
                    lhs.iter().map(|s| table.intern(s, ops)).collect(),
                    rhs.iter().map(|s| table.intern(s, ops)).collect(),
                )
            })
            .collect();
        Self { raw: raw_interned }
    }
}
