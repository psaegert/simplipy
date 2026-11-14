//! THE SCHEMAS OF PERMISSIVE'S NAMED STEPS (the specification's 4.9), for the judge. `permissive`
//! takes some steps the sound modes refuse inside its constructors, not as rules
//! (`ac::expr::Permissive`), so a rule that is an instance of one is judged by the step's name:
//! the judge rewrites both of its sides by the schemas below, reads the rewritten rule by
//! permissive's own criterion, and records the steps' names as the reason. Each schema is built
//! from the table the constructors read, so the judge reads the engine's own list:
//!
//! * `inverse pairs cancel`: `f(g(t)) = t` for each pair of `INVERSE_PAIRS`, in both orders
//!   (`inverse_pair_collapse`);
//! * `powers compose`: `(t^a)^b = t^(a*b)` for numeric exponents, a root an exponent `1/n`
//!   (`pow`'s and `rootn`'s licence, `as_rational_power`);
//! * `like terms cancel`: the laws whose premise `permissive` drops by this entry (`Law::holds`),
//!   each spelled as its statement says. The catalogue's other partial laws read a sum's sign in
//!   `f64` and are identities of the reals, where `permissive` reads one zero: no step of its own.
//!
//! The other entries are readings, not steps over a rule's structure, and have no schema: a
//! constant's collapse is the collapse alignment the judge reads (`fit::collapse_spans`); a
//! literal's float, a slot that matches any subexpression, float resolution and agreement where
//! the input has no value are permissive's own criterion.
//!
//! A schema is `(lhs, rhs)` in prefix tokens. A slot `_i` matches a part that holds a variable, a
//! `<constant>` or a slot: a fully numeric part is read for its value first, and no named step
//! reaches it (`Cx::permits_on`). A slot `#i` matches a fully numeric part. Any other token
//! matches itself, and `+` and `*` match their operands in either order.

use crate::ac::expr::{Permissive, INVERSE_PAIRS};
use crate::ac::laws::Law;

/// A schema: its left side and its right side, in prefix tokens.
pub type Schema = (Vec<String>, Vec<String>);

fn tokens(s: &str) -> Vec<String> {
    s.split(' ').map(String::from).collect()
}

/// The laws `permissive` applies by [`Permissive::LikeTerms`]: their premise in `f64` is the
/// like terms' definedness or a zero's sign, and `permissive` drops it (`Law::holds`).
pub const LIKE_TERM_LAWS: [Law; 5] = [
    Law::AddZero,
    Law::MulZero,
    Law::AddCancel,
    Law::MulCancel,
    Law::InfinityAbsorbs,
];

/// The schema of a law `permissive` applies by [`Permissive::LikeTerms`] (`LIKE_TERM_LAWS`), its
/// statement (`Law::statement`) in prefix tokens; None for any other law.
pub fn like_term_schema(law: Law) -> Option<Schema> {
    let (lhs, rhs) = match law {
        Law::AddZero => ("+ _0 0", "_0"),
        Law::MulZero => ("* 0 _0", "0"),
        Law::AddCancel => ("- _0 _0", "0"),
        Law::MulCancel => ("/ _0 _0", "1"),
        Law::InfinityAbsorbs => ("+ float(\"inf\") _0", "float(\"inf\")"),
        _ => return None,
    };
    Some((tokens(lhs), tokens(rhs)))
}

/// The schemas of `entry`, in the order the judge tries them; none for an entry that is a
/// reading.
pub fn schemas(entry: Permissive) -> Vec<Schema> {
    match entry {
        Permissive::InversePairs => INVERSE_PAIRS
            .iter()
            .flat_map(|&(f, g)| [(f, g), (g, f)])
            .map(|(outer, inner)| (tokens(&format!("{outer} {inner} _0")), tokens("_0")))
            .collect(),
        Permissive::PowersCompose => vec![
            (tokens("pow pow _0 #1 #2"), tokens("pow _0 * #1 #2")),
            (tokens("rootn pow _0 #1 #2"), tokens("pow _0 / #1 #2")),
            (tokens("pow rootn _0 #1 #2"), tokens("pow _0 / #2 #1")),
        ],
        Permissive::LikeTerms => LIKE_TERM_LAWS
            .iter()
            .filter_map(|&law| like_term_schema(law))
            .collect(),
        Permissive::ConstantCollapse
        | Permissive::LiteralToFloat
        | Permissive::AnySlot
        | Permissive::FloatResolution
        | Permissive::Agreement => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The other partial laws: identities of the reals whose premise in `f64` is a sum's sign,
    /// which `permissive`, reading one zero, does not ask; no step of permissive's own.
    const SUM_SIGN_LAWS: [Law; 2] = [Law::AddUnfactor, Law::AddContent];

    /// The step entries are the three the constructors take on a rule's structure, each schema
    /// one expression a side; the partial laws are the like-term laws, each with its schema, and
    /// the sum-sign laws, so a partial law the catalogue gains must be placed before it is read.
    #[test]
    fn the_schemas_are_the_constructors_tables() {
        let stepped: Vec<&str> = Permissive::ALL
            .iter()
            .filter(|&&p| !schemas(p).is_empty())
            .map(|p| p.name())
            .collect();
        assert_eq!(
            stepped,
            [
                "inverse pairs cancel",
                "powers compose",
                "like terms cancel"
            ]
        );
        assert_eq!(
            schemas(Permissive::InversePairs).len(),
            2 * INVERSE_PAIRS.len()
        );
        for law in Law::ALL {
            let like = LIKE_TERM_LAWS.contains(&law);
            assert_eq!(
                !law.total(),
                like || SUM_SIGN_LAWS.contains(&law),
                "{}",
                law.name()
            );
            assert_eq!(like_term_schema(law).is_some(), like, "{}", law.name());
        }
        let arity = |t: &str| match t {
            "+" | "-" | "*" | "/" | "pow" | "rootn" => 2,
            "asin" | "sin" | "acos" | "cos" | "atan" | "tan" | "exp" | "log" | "asinh" | "sinh"
            | "acosh" | "cosh" | "atanh" | "tanh" => 1,
            _ => 0,
        };
        for entry in Permissive::ALL {
            for (lhs, rhs) in schemas(entry) {
                for side in [&lhs, &rhs] {
                    let need = side.iter().fold(1i32, |need, t| need + arity(t) - 1);
                    assert_eq!(need, 0, "{side:?}");
                }
            }
        }
    }
}
