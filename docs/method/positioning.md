# Positioning and prior art

Where SimpliPy sits in the literature, stated the way we would want a
referee to read it: neighbours named, differences derived from the domain
rather than from taste, and the limits of every analogy spelled out. The
measured figures on this page were re-verified against the published
`acj-4-3` artifact and the deployed engine (2026-08-17); the citation forms
follow the release audit's verification pass (each of the four flagged
citations was independently confirmed before this page was published).

SimpliPy's design sits in four well-populated neighbourhoods, and we would rather name our
neighbours than be mistaken for a pioneer.

### Rewrite-rule inference

SimpliPy's mining phase is an instance of the enumerate–select–minimize schema for rewrite-rule
inference introduced by Ruler [Nandi et al. 2021], and we adopt it deliberately. The correspondence
is close enough to be worth stating precisely. Ruler's term enumeration modulo the equivalences its
own learned rules already prove is our per-length complete enumeration of the source universe
combined with the AC-judged Kruskal prune, which simplifies each source under the rules mined so far
before searching — the AC engine standing in for Ruler's e-graph. Ruler's characteristic vectors,
which fingerprint each term by evaluating it on a predetermined family of variable assignments, are
our seeded 1024-row evaluation matrix; the heavy-tailed mixture we sample is Ruler's two cvec
strategies blended per element, combining randomly drawn values with a tier of exact algebraic
corner points, for the reason Ruler reports for bitvector-32 — random sampling alone misses the edge
cases. Ruler's handling of partial operators, where a cvec entry may be null and two cvecs match
when their non-null entries agree everywhere and at least one position is informative, is our
`allclose_extends` together with the `min_informative` evidence gate. And Ruler's
`choose_eqs`/`shrink`, which discards a candidate once the retained rules already derive it, is our
`prune_covered_rules` — the contract Enumo [Pal et al. 2023] later stated formally, with our
serve-time engine in the role of the equality-saturation oracle. We claim no novelty for any of this.

Three things differ, and each follows from the domain rather than from taste. First, SimpliPy mines
over the reals with partial transcendental functions, where equality is undecidable. This is the
corner Ruler names and leaves open — "undecidable domains may decide to give up a guarantee of
soundness and use a sampling-based validation" — and which Enumo places outside cvec matching
altogether, reaching trigonometric identities by rewriting through complex-exponential forms rather
than evaluating them. We evaluate directly and replace Ruler's SMT `is_valid` with a numerical
oracle in the style of Herbie [Panchekha et al. 2015]: f64 is an explicit pre-filter, near-miss rows
are re-adjudicated at 1024 bits, and a refutation at that precision must be confirmed at 2048 bits
before it counts. Unlike a decision procedure, this oracle both rejects unsound candidates and
recovers sound ones that finite precision had hidden. Second, our rules carry constants under a
quantifier structure Ruler has no analogue for: a source's constants are universally challenged
across re-drawings crossed with sign and pole grids, while the target's constants are existentially
fitted per challenge by closed-form least squares where the candidate is affine in its parameters
and by restarted Levenberg–Marquardt otherwise. Third, Ruler infers bidirectional equations for a
non-destructive e-graph engine and says plainly that it "generates rules that do not guarantee a
reduction order, since it synthesizes rules like commutativity and associativity." SimpliPy applies
rules destructively and first-match-wins, so it cannot accept an unorientable rule at all:
acceptance is fused with the serve-time reduction ordering, and a candidate is mintable only if it
descends strictly in the same measure the rewrite pass fires under. What is deferred future work for
Ruler is a precondition for us.

Two limits of the analogy are worth naming, since a reader who knows Ruler will otherwise import
guarantees we do not have. Our coverage check is LHS-only in the sense Enumo formalises — we run the
engine on the left side and require it to reach the promised right side or strictly below in the
total serve ordering — which is stronger than the LHS-RHS metric Ruler reports, but is verified on
an enumerated set of instantiation probes rather than on a generic instance, so it is an empirical
claim about those probes and not a theorem about all substitutions. And because our rewriting is
order-sensitive, coverage is a property of this engine with these rules in this order, not of the
equational theory; the reduction measure is fingerprinted into the provenance sidecar for exactly
this reason. Finally, we mine refinements rather than equivalences: a replacement may be defined
where the source is not, which is the right shape for a directed rewrite but differs from Ruler's
symmetric null-matching. Inferring the side conditions that would make such rules conditional
equivalences, rather than refinements, is the subject of Chompy
[Cheung, Nandi & Lerner, FMCAD 2026, to appear], and is the natural next step for the
pole-grid refusal cases the mining pipeline records.

### Equality saturation for symbolic regression

The closest work in the symbolic-regression literature is de França and Kronberger's
equality-saturation line. *Reducing Overparameterization of Symbolic Regression Models with Equality
Saturation* [GECCO '23] observes that GP-based SR systems systematically emit models carrying more
numerical parameters than the model has degrees of freedom, and repairs them by running equality
saturation over a hand-written rule set — unconstrained algebraic rules, constrained rules carrying
explicit side conditions, and parameter-specific re-associations such as `a*x + b → a*(x + b/a)` —
extracting with a cost function that charges 5 to a parameter node and 1 to every other node. They
ground-truth the result against the numeric rank of the Jacobian ∂f/∂θ, obtained by SVD; on Operon
models the merged form attains that rank exactly in 67–74% of cases and is within one parameter in
94–100%. The same machinery has since been pushed inside the search loop (eggp, GECCO '25), turned
into a query interface over all visited candidates (rEGGression, GECCO '25), made the search
operator itself (SymRegg, 2026), and measured for its effect on the parameter optimiser [Kronberger
and de França, JSC 2024].

SimpliPy prices the same axis. Its `<constant>` placeholder is their θ — an unknown a fit will
supply — and μ charges it 1133 bits against 8 bits for a grammar symbol, so one free parameter is
worth 141.6 symbols where their cost model makes it worth five nodes. Two differences are worth
stating. First, that weight is derived rather than chosen: it is the supremum of μ over f64
round-trip spellings plus a sign bit, so `<constant>` is guaranteed to dominate every literal it
could be instantiated to. Second, and more substantially, μ has a middle tier their measure cannot
express. Under μ, `E*x` costs 24 bits, `2.718281828459045*x` costs 112, and `<constant>*x` costs
1149: a named exact constant, a rounded 52-bit literal and a free degree of freedom are three
different prices. Their cost function collapses all three into one bucket, and their pipeline
deliberately materialises constant subexpressions before saturation ("we do not keep expressions
such as `exp(2+4)` in the e-graph and simply reduce it to its evaluated value"). That is the right
choice in their regime, where every constant will be refitted and its exact value therefore carries
no information, and the wrong one in ours, where the output is a symbolic artefact read by a human
or a tokenizer. The systems solve adjacent problems under different terminal conditions.

### SMT-verified rule synthesis, and why we do not use a solver

A second family verifies rewrite rules with an SMT solver rather than discovering them numerically.
Souper synthesises LLVM peepholes by counterexample-guided inductive synthesis against Z3; Alive and
Alive2 check peephole optimizations and whole-function translations, the latter finding 47 new LLVM
bugs; and Newcomb et al. translate every rule of Halide's simplifier into a Z3 query, prove
termination, and uncover four incorrect rules and eight that could loop forever. Halide is the
instructive case, because its theory — the integers with division, modulo, min and max — is already
undecidable, and the authors still discharged the great majority of the ruleset automatically,
proving the residue — 141 rules under the revised division semantics — by hand in Coq. Ruler, whose discovery loop ours most closely resembles, uses
Z3 by default.

SimpliPy does not, and the reason is not that solvers are inconvenient. It is that no decision
procedure exists for the theory its operators generate, and that the strongest approximate
procedures that do apply discharge a strictly weaker obligation than the one a mined rule states.

Tarski's decidability of real closed fields does not survive the addition of transcendental
functions. Richardson [1968] showed that for expressions built from the rationals, π, ln 2, a
variable, addition, multiplication, composition, sin, exp and abs, deciding whether an expression is
identically zero — or nonnegative everywhere — is undecidable; Laczkovich [2003] sharpened this to
the integers, a variable and sin alone. SimpliPy's operator set contains sin, cos, tan and their
inverses, the hyperbolics, exp, log, abs and general powers, and so sits inside that class with room
to spare. Nor is there a conditional escape: Macintyre and Wilkie [1996] showed that the real
exponential field is decidable if Schanuel's conjecture holds, but that result covers exp and not
sin, and sin is exactly what defines the integers inside the reals.

Nor is there a standard SMT theory to encode the rules into. SMT-LIB's theory of Reals is
real-closed-field arithmetic and contains no transcendental functions; SMT-LIB's FloatingPoint
theory, which would otherwise model our f64 semantics precisely, provides addition, subtraction,
multiplication, division, fused multiply-add, square root and remainder, and no transcendental
operations at all.

The tools that reach further are sound but weaker than the obligation. Incremental linearization
[Cimatti et al. 2018] abstracts transcendental functions as uninterpreted and refines them with
piecewise-linear bounds; it is sound and incomplete, and its authors open by noting the problem is
undecidable in general. MetiTarski [Akbarpour and Paulson 2010] proves quantified inequalities over
special functions by substituting rational upper and lower bounds and discharging the result with a
real-closed-field procedure; it decides inequalities, not identities. The most tempting option is
delta-complete decision procedures and their implementation dReal [Gao, Avigad and Clarke 2012; Gao,
Kong and Clarke 2013], which handle sin, exp, log and their relatives directly — but their guarantee
is defined for sentences with **bounded quantifiers only**, and δ-strengthening perturbs inequality
atoms. An identity, encoded as a conjunction of two non-strict inequalities, has a δ-strengthening
that is unsatisfiable for every positive δ, so the procedure is permitted to report "δ-False" on a
true identity and the guarantee is **vacuous on precisely the sentences we need**. What dReal
genuinely certifies is the ε-identity: an `unsat` answer to "does |lhs − rhs| ≥ ε somewhere in box
B" is a real proof, and a strictly stronger object than sampling, that lhs and rhs agree to within ε
at every point of B. It is bounded, and it is approximate; our rules are claimed on an unbounded
domain up to a null set, and they are exact.

The obligation is in fact stronger still: a mined rule is a for-all-source-constants,
there-exist-target-constants statement whose acceptance is decided against the serve-time reduction
ordering, and whose certification licenses the replacement to be defined where the source is not.
Delta-decision procedures for exists-forall problems over the reals exist [Kong, Solar-Lezama and
Gao 2018] but inherit both restrictions, and the ordering and domain-extension conditions are not
first-order sentences about ℝ at all.

We therefore certify numerically, which is the course the rule-synthesis literature already
prescribes for this case. Ruler states it directly: small domains may use model checking, larger
domains may use SMT, and undecidable domains may give up the guarantee of soundness and use
sampling-based validation — and Ruler's own continuous domain is validated by random testing. Our
oracle is correspondingly a high-power falsifier and not a proof, and every guarantee in this
document is stated at that strength. A δ-complete check over a compact box would strictly strengthen
it on the fragment dReal's expression language supports, and we regard adding it as an independent
second gate as the natural next hardening; it would not, however, close the unbounded-domain gap,
because bounded quantification is a hypothesis of δ-decidability itself.

### Interval enclosures with validity annotations

`rust/interval.rs` computes an over-approximating enclosure of an expression's value set and
annotates it with what is known about the expression's domain. That architecture is standard: IEEE
Std 1788-2015 standardises it as *decorations* (`com`, `dac`, `def`, `trv`, `ill`; decoration
semantics here per the public P1788.1/D9.7 draft, which became IEEE 1788.1-2017 and whose
decoration system matches 1788-2015), and Rival —
the interval library inside the Herbie floating-point repair tool — implements it as *error
intervals*, a three-valued distinction between "all points in this box are a domain error", "some
point may be", and "none are" [Flatt and Panchekha, ARITH 2023]. SimpliPy's `Vs` adds one thing to
that picture: it grades the "some points" case by *Lebesgue measure*, distinguishing a finite value
that exists on a null set of inputs from one that exists on a set of positive measure. Rival has no
reason to make that distinction — Herbie samples floating-point values, so its measure is the
counting measure over a finite set, in which nothing is null. SimpliPy's rewrite licences are stated
over the reals, where the distinction decides whether a rewrite is legal. The dichotomy itself is
classical: in an o-minimal structure a definable set of dimension < n is Lebesgue-null [van den
Dries 1998]. What we believe is uncommon is carrying that grading *inside* a propagating interval
abstraction with per-operator transfer rules.

### Precision escalation

`rust/hiprec.rs` re-evaluates near-miss rows at 1024 and 2048 bits. Escalating precision until the
answer is trusted is Ziv's onion-peeling strategy [Ziv 1991]; Ziv's version is a *proof* because
each step carries a rigorous error bound and stops when the resulting *interval* decides the
question, and ours is not, because it compares point values. We therefore use it only to refute,
never to accept, and a disagreement between the two precisions decides nothing. This is forced
rather than chosen: no fixed precision schedule can be a proof (the Table Maker's Dilemma), and zero
recognition for elementary constants is undecidable [Richardson 1968]. The residual risk is
*correlated* error — two precisions sharing a saturation cliff agree and are both wrong — which our
fixed ladder mitigates by schedule rather than by criterion. A principled criterion exists: Rival's
*movability flags* mark an endpoint immovable when higher-precision recomputation provably cannot
change it, giving a sound "escalation is futile" test [Flatt and Panchekha, ARITH 2023]. Adopting it
would replace the ladder's guess.

### Complexity measures that price exactness

Charging a numeric literal by the size of its exact value is not new. Wolfram's default
`ComplexityFunction` ranks forms by leaf count "with corrections to treat integers with more digits
as more complex", charging an integer its decimal digit count and a rational the sum over numerator
and denominator; Maple's `length()` returns the decimal digit count of an integer and underpins
`simplify(expr, size)`; and Carette [ISSAC 2004] gives the MDL account, measuring literals in bits
of their exact value. MDL objectives in symbolic regression price constants the same way. What μ
adds is two things. First, the treatment of the *inexact* literal: Mathematica charges any inexact
real a flat 2 and SymPy's `count_ops` charges a Float 0, so both rank the rounded surrogate as
strictly *simpler* than the exact rational it approximates — the incentive is inverted, not merely
absent. μ prices the written decimal by its exact value as a rational, so `1/3` costs 3,000
millibits and `0.3333333333333333` costs 104,717, a factor of 35. (Maple is the honest exception: it
does charge floats by representation, so it does not invert the incentive.) Second, the *role*: in
Mathematica and Maple the measure is a search-time ranking heuristic over candidates, and in Carette
it is a definition rather than an algorithm, whereas in SimpliPy μ is the well-founded descent order
that gates every rewrite. "No rewrite may replace an exact literal by a costlier-to-describe rounded
surrogate" is therefore a property of the rewrite relation rather than a policy. Standard
Knuth–Bendix orderings assign each symbol a fixed weight; making a numeral's weight a function of
its magnitude, *inside a reduction order*, is not anticipated in the term-rewriting literature.

### Equality saturation as an engine choice

SimpliPy is not an equality-saturation engine, and that is a design choice rather than an oversight.
An e-graph represents equivalence classes of terms, but it has no native representation for
associativity and commutativity: AC has to be supplied as ordinary rewrite rules, and once it is,
the e-graph grows without bound. Given twelve standard AC-and-distributivity rules and the nine-node
term `x1*(x2+x3) − x1*x2`, egg 0.11.0 reaches 2,011,908 e-nodes across 959,082 e-classes in 11.9 s
and is still growing when the clock runs out. Saturation — the fixpoint that makes equality
saturation complete, and that justifies its cost — is simply not reachable here, so in practice one
falls back on iteration budgets and rule-banning schedulers: heuristic search, with the completeness
guarantee gone. This is a known open problem rather than a quirk of one library. As of August 2026
no shipped e-graph system has AC-native representation: egglog 2.0 offers container sorts and a
back-off scheduler that bans rules which grow the e-graph too quickly, and recent work on e-graphs
modulo theories [Zucker, EGRAPHS 2025] and on multiset-encoded commutative operators points at real
solutions but remains prototype-stage. SimpliPy takes the other branch: it builds AC into the term
representation itself, so commutative rearrangement is free and never enters a search space at all.

The trade runs in both directions, and it is worth being plain about the losing half. Because
SimpliPy normalizes rather than saturates, it will not find rewrites that require exploring a space.
On that same term, egg proves `x1*(x2+x3) − x1*x2 = x1*x3` at iteration 5 in 0.52 ms, while SimpliPy
— which has AC cancellation but no distributivity rule — returns the expression unchanged in 24 µs.
What SimpliPy buys in exchange is throughput at ruleset scale on the workload it targets. Its
shipped asset is a mined identity table rather than a hand-written rewrite system: 97.1% of left-hand
sides contain no wildcard (re-measured on the published acj-4-3, 2026-08-17), none is non-linear, and none exceeds four tokens. A bucketed index with a
Bloom prefilter answers that shape in about 1.72 ns per candidate rule, which is what makes
normalizing millions of machine-generated expressions practical. Equality saturation is built for a
different question — find the best member of an equivalence class under a small, carefully chosen
ruleset — and it remains the right tool for that question.

### What we do not claim

The almost-everywhere finiteness certificate is discharged over a bounded box plus a set of
structural rules. It is not a decision procedure for the general question, and there cannot be one:
for a vocabulary containing `sin`, `exp` and `abs`, deciding whether an expression is identically
zero is undecidable [Richardson 1968]. Modern numerical decision procedures over the reals are
correspondingly restricted to bounded quantifiers [Gao, Avigad and Clarke 2012]. Where rigorous
methods do reach into the unbounded complement, they do so by compactification or projective
transformation [Schichl, Neumaier, Markót and Domes, CPAIOR 2013], which is the shape of the tail
arm we implement. And no finite paving can ever certify *nullity* — it can only bound the
exceptional set's outer measure by some ε > 0 [Jaulin and Walter 1993] — which is why the pole cases
go through a structural arm instead of through subdivision.

---


## References

**Rewrite-rule inference and equality saturation**

- Chandrakana Nandi, Max Willsey, Amy Zhu, Yisu Remy Wang, Brett Saiki, Adam Anderson, Adriana
  Schulz, Dan Grossman, Zachary Tatlock. 2021. *Rewrite Rule Inference Using Equality Saturation.*
  PACMPL 5(OOPSLA), Article 119. doi:10.1145/3485496 (arXiv:2108.10436). Distinguished Paper.
- Anjali Pal, Brett Saiki, Ryan Tjoa, Cynthia Richey, Amy Zhu, Oliver Flatt, Max Willsey, Zachary
  Tatlock, Chandrakana Nandi. 2023. *Equality Saturation Theory Exploration à la Carte.* PACMPL
  7(OOPSLA2), Article 258. doi:10.1145/3622834.
- Andrew Cheung, Chandrakana Nandi, Sorin Lerner. 2026. *Conditional Rewrite Rule Synthesis Using
  E-Graphs and Implication Propagation* (Chompy). FMCAD 2026, to appear (short paper).
  Artifact: github.com/ninehusky/chompy.
- Max Willsey, Chandrakana Nandi, Yisu Remy Wang, Oliver Flatt, Zachary Tatlock, Pavel Panchekha.
  2021. *egg: Fast and Extensible Equality Saturation.* PACMPL 5(POPL), Article 23.
  doi:10.1145/3434304.
- Yihong Zhang et al. 2023. *Better Together: Unifying Datalog and Equality Saturation* (egglog).
  PACMPL 7(PLDI), Article 125. doi:10.1145/3591239.
- Philip Zucker. 2025. *Omelets Need Onions: E-graphs Modulo Theories via Bottom-up E-matching.*
  EGRAPHS 2025 @ PLDI. arXiv:2504.14340.

**Symbolic regression and parameter redundancy**

- F. O. de França, G. Kronberger. 2023. *Reducing Overparameterization of Symbolic Regression Models
  with Equality Saturation.* GECCO '23, 1064–1072. doi:10.1145/3583131.3590346.
- F. O. de França, G. Kronberger. 2025. *Improving Genetic Programming for Symbolic Regression with
  Equality Graphs* (eggp). GECCO '25, 989–998. arXiv:2501.17848.
- F. O. de França, G. Kronberger. 2025. *rEGGression.* GECCO '25, 4–12. arXiv:2501.17859.
- F. O. de França, G. Kronberger. 2026. *Equality graph-assisted symbolic regression* (SymRegg).
  Phil. Trans. R. Soc. A 384(2317):20240597. arXiv:2511.01009.
- G. Kronberger, F. O. de França. 2024. *Effects of reducing redundant parameters in parameter
  optimization for symbolic regression using genetic programming.* J. Symbolic Computation
  129:102413.

**SMT-verified rule synthesis**

- R. Sasnauskas, Y. Chen, P. Collingbourne, J. Ketema, G. Lup, J. Taneja, J. Regehr. 2017. *Souper:
  A Synthesizing Superoptimizer.* arXiv:1711.04422.
- N. P. Lopes, D. Menendez, S. Nagarakatte, J. Regehr. 2015. *Provably Correct Peephole
  Optimizations with Alive.* PLDI 2015, 22–32.
- N. P. Lopes, J. Lee, C.-K. Hur, Z. Liu, J. Regehr. 2021. *Alive2: Bounded Translation Validation
  for LLVM.* PLDI 2021, 65–79.
- J. L. Newcomb, A. Adams, S. Johnson, R. Bodik, S. Kamil. 2020. *Verifying and improving Halide's
  term rewriting system with program synthesis.* PACMPL 4(OOPSLA), Article 166.

**Decidability**

- D. Richardson. 1968. *Some unsolvable problems involving elementary functions of a real variable.*
  J. Symbolic Logic 33(4):514–520.
- M. Laczkovich. 2003. *The removal of π from some undecidable problems involving elementary
  functions.* Proc. AMS 131:2235–2240.
- A. Macintyre, A. J. Wilkie. 1996. *On the decidability of the real exponential field.* In
  *Kreiseliana*, A K Peters, 441–467.
- S. Gao, J. Avigad, E. M. Clarke. 2012. *δ-Complete Decision Procedures for Satisfiability over the
  Reals.* IJCAR 2012, LNCS 7364:286–300. / *δ-Decidability over the Reals.* LICS 2012,
  arXiv:1204.6671.
- S. Gao, S. Kong, E. M. Clarke. 2013. *dReal: An SMT Solver for Nonlinear Theories over the Reals.*
  CADE-24, LNCS 7898:208–214.
- S. Kong, A. Solar-Lezama, S. Gao. 2018. *Delta-Decision Procedures for Exists-Forall Problems over
  the Reals.* CAV 2018, LNCS 10981:219–235.
- A. Cimatti, A. Griggio, A. Irfan, M. Roveri, R. Sebastiani. 2018. *Incremental Linearization for
  Satisfiability and Verification Modulo Nonlinear Arithmetic and Transcendental Functions.* ACM
  TOCL 19(3):1–52.
- B. Akbarpour, L. C. Paulson. 2010. *MetiTarski: An Automatic Theorem Prover for Real-Valued
  Special Functions.* J. Automated Reasoning 44(3):175–205.
- SMT-LIB theory definitions: Reals; FloatingPoint. https://smt-lib.org/theories-FloatingPoint.shtml

**Interval arithmetic and validated numerics**

- IEEE Std 1788-2015, *IEEE Standard for Interval Arithmetic.* Decoration semantics quoted from the
  public P1788.1/D9.7 draft (became IEEE 1788.1-2017; decoration system matches 1788-2015 — the 2015
  normative text itself is paywalled), corroborated by secondary sources.
- O. Flatt, P. Panchekha. 2023. *Making Interval Arithmetic Robust to Overflow.* ARITH 2023, 44–47.
  (Preprint: *An Interval Arithmetic for Robust Error Estimation*, arXiv:2107.05784.)
- P. Panchekha, A. Sanchez-Stern, J. R. Wilcox, Z. Tatlock. 2015. *Automatically Improving Accuracy
  for Floating Point Expressions.* PLDI 2015. doi:10.1145/2737924.2737959.
- A. Ziv. 1991. *Fast evaluation of elementary mathematical functions with correctly rounded last
  bit.* ACM TOMS 17(3):410–423.
- V. Lefèvre, J.-M. Muller. *The Table Maker's Dilemma.*
- L. Jaulin, E. Walter. 1993. *Set inversion via interval analysis for nonlinear bounded-error
  estimation.* Automatica 29(4):1053–1064.
- H. Schichl, A. Neumaier, M. C. Markót, F. Domes. 2013. *On solving mixed-integer constraint
  satisfaction problems with unbounded variables.* CPAIOR 2013, LNCS 7874:216–233.
- L. van den Dries. 1998. *Tame Topology and O-minimal Structures.* LMS Lecture Notes 248, CUP.
  (Cited as textbook background for a classical fact.)

**Complexity measures**

- Wolfram Language documentation, `ComplexityFunction` (the default formula is published verbatim
  on that page).
- Maple documentation, `length` and `simplify/size`.
- J. Carette. 2004. *Understanding Expression Simplification.* ISSAC 2004.

