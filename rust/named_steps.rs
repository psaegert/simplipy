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
//!   `f64` and are identities of the reals, where `permissive` reads one zero: no step of its own;
//! * `a zero factor absorbs the product`: `mul.zero`, whose premise `permissive` drops by this
//!   entry, spelled as its statement says.
//!
//! The other entries are readings, not steps over a rule's structure, and have no schema: a
//! constant's collapse is the collapse alignment the judge reads (`fit::collapse_spans`); a
//! sum's grouping of a shared number is an identity of the reals, which the judge reads; a
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
pub const LIKE_TERM_LAWS: [Law; 4] = [
    Law::AddZero,
    Law::AddCancel,
    Law::MulCancel,
    Law::InfinityAbsorbs,
];

/// The schema of a law `permissive` applies by [`Permissive::LikeTerms`] (`LIKE_TERM_LAWS`), its
/// statement (`Law::statement`) in prefix tokens; None for any other law.
pub fn like_term_schema(law: Law) -> Option<Schema> {
    let (lhs, rhs) = match law {
        Law::AddZero => ("+ _0 0", "_0"),
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
        // `mul.zero`'s statement (`Law::statement`), `0*t = 0`.
        Permissive::ZeroAbsorbs => vec![(tokens("* 0 _0"), tokens("0"))],
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
    const SUM_SIGN_LAWS: [Law; 6] = [
        Law::AddUnfactor,
        Law::AddContent,
        Law::ClassSignTrade,
        Law::ClassPoleOrientation,
        Law::ClassOppositeSums,
        Law::ClassLikeTermsUpToSign,
    ];

    /// The functions' partial laws, the lattice's absorption and the distributive law the search
    /// applies: identities of the reals wherever their left side has a value, whose premise in
    /// `f64` is a float64 kind (`tanh`'s saturation, a flat point, a zero's sign, a NaN on a
    /// region, an infinity on a region), which `permissive` does not ask (`abs(t) = t` keeps the
    /// reals' own `t >= 0`); no step of permissive's own.
    const KIND_LAWS: [Law; 16] = [
        Law::FunAtanhTanh,
        Law::FunLogExp,
        Law::FunAcoshCosh,
        Law::FunExpLog,
        Law::FunLogInverse,
        Law::FunUnitInverses,
        Law::FunAbsNonnegative,
        Law::LatticeAbsorb,
        Law::PowDistribute,
        Law::ParseInverseOfProduct,
        Law::PowHalfRoot,
        Law::PowNegatedSite,
        Law::FunBesseliParity,
        Law::SearchDistribute,
        Law::SearchPowExpand,
        Law::SearchFactor,
    ];

    /// A step of `permissive`'s search alone, with a premise of its own there (a factor of one
    /// strict sign): an identity of the reals where its premise holds, which `f64` does not take,
    /// read on a rule's instances by permissive's own criterion; no schema.
    const SEARCH_LAWS: [Law; 1] = [Law::SearchReciprocalJoin];

    /// The compositions of powers: identities of the reals on t >= 0 whose premise is the
    /// reals' own branch cut, which `permissive` takes by the step `powers compose` (its
    /// schemas).
    const COMPOSE_LAWS: [Law; 3] = [Law::PowCompose, Law::PowRootInverse, Law::PowOddRootCompose];

    /// The like-base merges: identities of the reals on t >= 0 whose premise is the reals' own
    /// branch cut, which `permissive` drops by its reading of agreement (extensions are free); a
    /// reading, no schema.
    const AGREEMENT_LAWS: [Law; 2] = [Law::MulCollect, Law::MulCollectSymbolic];

    /// The step entries are the four the constructors take on a rule's structure, each schema
    /// one expression a side; the partial laws are the like-term laws, each with its schema,
    /// `mul.zero` (the step `a zero factor absorbs the product`), the sum-sign, kind,
    /// composition, agreement and search laws, and the inverse pairs' law, which is the step
    /// `inverse pairs cancel` (its schemas, `INVERSE_PAIRS`), so a partial law the catalogue gains
    /// must be placed before it is read.
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
                "like terms cancel",
                "a zero factor absorbs the product"
            ]
        );
        assert_eq!(
            schemas(Permissive::ZeroAbsorbs),
            [(tokens("* 0 _0"), tokens("0"))]
        );
        assert_eq!(
            schemas(Permissive::InversePairs).len(),
            2 * INVERSE_PAIRS.len()
        );
        for law in Law::ALL {
            let like = LIKE_TERM_LAWS.contains(&law);
            assert_eq!(
                !law.total(),
                like || law == Law::MulZero
                    || SUM_SIGN_LAWS.contains(&law)
                    || KIND_LAWS.contains(&law)
                    || COMPOSE_LAWS.contains(&law)
                    || AGREEMENT_LAWS.contains(&law)
                    || SEARCH_LAWS.contains(&law)
                    || law == Law::PermInversePairs,
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
