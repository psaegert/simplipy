"""Type stub for the compiled Rust inline core (``simplipy._core``).

Hand-written to mirror the ``#[pymethods]`` surface in ``rust/lib.rs``; keep in sync on changes.
Compiled extensions are opaque to static type-checkers, so this stub is what makes the ``Engine``
surface (and the module constants) visible to mypy/pyright downstream and in this package.
``simplipy._core`` is internal: it carries no stability promise.
"""
from typing import Any, Literal

__build__: str
NEAR_MISS_FRAC: float
ARTIFACT_ENV_SWITCHES: list[str]

#: Raised by a call passed ``own_set=True`` when the core holds no set of its own for the mode.
class MissingModeSet(LookupError): ...

class CandidateLibrary:
    @property
    def n_candidates(self) -> int: ...
    @property
    def n_filtered(self) -> int: ...

class Engine:
    @staticmethod
    def from_strs(config_yaml: str, rules_json: str, canon: str | None = ...) -> Engine: ...
    def ac_simplify_in_mode(
        self,
        tokens: list[str],
        max_passes: int = ...,
        form: Literal["tagged", "explicit"] = ...,
        explore_budget: int | None = ...,  # None: until a round finds nothing
        *,
        rule_mode: Literal["default", "real", "permissive"],
        own_set: bool = ...,  # serve only the mode's own set: MissingModeSet where it has none
    ) -> list[str]: ...
    #: Simplify in `rule_mode` with artifact rows (and their minted orientation twins)
    #: suppressed at the matcher's fire site -- behaviorally the engine built without
    #: those rows, up to the cross-rule translate state `ac_shadow_census` enumerates.
    def ac_simplify_suppressed(
        self,
        tokens: list[str],
        max_passes: int = ...,
        form: Literal["tagged", "explicit"] = ...,
        explore_budget: int | None = ...,  # None: until a round finds nothing
        suppressed_rows: list[int] = ...,
        *,
        rule_mode: str,
    ) -> list[str]: ...
    #: `ac_simplify_in_mode` with the rule steps it took: (answer, [(lhs instance, rhs
    #: instance)]), each fire's rule instance as prefix tokens, in firing order (F8's steps).
    def ac_simplify_fires(
        self,
        tokens: list[str],
        max_passes: int = ...,
        form: Literal["tagged", "explicit"] = ...,
        explore_budget: int | None = ...,
        *,
        rule_mode: str,
    ) -> tuple[list[str], list[tuple[list[str], list[str]]]]: ...
    def ac_simplify_infix_in_mode(
        self,
        tokens: list[str],
        max_passes: int = ...,
        explore_budget: int | None = ...,  # None: until a round finds nothing
        *,
        rule_mode: Literal["default", "real", "permissive"],
        own_set: bool = ...,
    ) -> str: ...
    def prettify(
        self,
        tokens: list[str],
        digits: int = ...,
        form: Literal["infix", "explicit", "tagged"] = ...,
        precision: int | None = ...,  # significant digits of a float; None: its shortest decimal
        *,
        rule_mode: Literal["default", "real", "permissive"],  # the number domain; no rules
        check_structure: bool = ...,  # answer (display, whether it keeps the structure)
        keep_grouping: bool = ...,  # every sum grouped as the explicit tokens group it
    ) -> str | list[str] | tuple[str | list[str], bool]: ...
    def ac_simplify_explore(
        self,
        tokens: list[str],
        max_passes: int = ...,
        form: Literal["tagged", "explicit"] = ...,
        explore_budget: int = ...,
        *,
        rule_mode: Literal["default", "real", "permissive"],
    ) -> list[str]: ...
    def ac_ordered_below_in(self, a: list[str], b: list[str], *, rule_mode: str) -> bool: ...
    #: Census of the nodes `mu` charges, over the bare canonical forms of a batch.
    def ac_node_census(self, exprs: list[list[str]]) -> dict[str, int]: ...
    def ac_canonical_keys(
        self, exprs: list[list[str]], *, rule_mode: str, canon: str | None = ...
    ) -> list[list[str] | None]: ...
    def mine_sources(self, enumerated: list[list[str]], *, rule_mode: str) -> list[list[str]]: ...
    def mine_mark(self, source: list[str], *, rule_mode: str) -> list[str]: ...
    def ac_translate_preview(
        self, rules: list[tuple[list[str], list[str]]], *, rule_mode: str
    ) -> list[tuple[list[str], list[str], int]]: ...
    def law_instance(self, lhs: list[str], rhs: list[str], *, rule_mode: str) -> str | None: ...
    def ac_complexity(self, tokens: list[str], canon: Literal["default", "mode"] = ..., *, rule_mode: Literal["default", "real", "permissive"]) -> int: ...
    def ac_complexity_certified(self, tokens: list[str], canon: Literal["default", "mode"] = ..., *, rule_mode: Literal["default", "real", "permissive"]) -> int: ...
    def ac_rules_info_in_mode(self, *, rule_mode: str) -> tuple[int, int, int, int]: ...
    def mode_rules_len(self, *, rule_mode: str) -> int | None: ...
    def mode_rules_canon(self, *, rule_mode: str) -> str: ...
    #: (lhs, rhs, source_index) per SERVED rule -- the matcher's own set, including the
    #: orientation twins minted at load, which no artifact row corresponds to.
    def ac_served_rules_in_mode(
        self, *, rule_mode: str
    ) -> list[tuple[list[str], list[str], int]]: ...
    def ac_registry_dropped(self) -> int: ...
    def ac_rules_drop_census(self, *, rule_mode: str) -> dict[str, int]: ...
    #: The shadow census: every translate-time skip caused by cross-rule state, as
    #: (victim_row, owner_row, owner_was_twin, kind) -- kind is "twin-shadow".
    #: Removing an owner row resurrects its victims' entries; see
    #: `ac_simplify_suppressed`.
    def ac_shadow_census(self, *, rule_mode: str) -> list[tuple[int, int, bool, str]]: ...
    def interval_domain_extension_p(
        self,
        source: list[str],
        src_params: list[float],
        target: list[str],
        tgt_params: list[float],
    ) -> float | None: ...
    def find_rule(
        self,
        source: list[str],
        simplified_length: int,
        max_target: int | None,
        candidates: list[list[str]],
        var_names: list[str],
        x_flat: list[float],
        n_rows: int,
        challenges: int = ...,
        retries: int = ...,
        seed: int = ...,
        rtol: float = ...,
        atol: float = ...,
        min_informative: int | None = ...,
        *,
        rule_mode: str,
    ) -> list[str] | None: ...
    def registry_mint_refusal(
        self, source: list[str], mark: list[str], target: list[str], var_names: list[str]
    ) -> str | None: ...
    #: The load's verdict on one rule read in `rule_mode`: the refusing gate's name
    #: (`const-count-increase`, `not-ordered-below`, `engine-owned-law`, ...) or None.
    def load_refusal(self, lhs: list[str], rhs: list[str], *, rule_mode: str) -> str | None: ...
    def is_valid(self, tokens: list[str]) -> bool: ...
    def prefix_to_infix(
        self, tokens: list[str], power: Literal["func", "**"] = ..., realization: bool = ...
    ) -> str: ...
    def to_tagged_syntactic(self, tokens: list[str]) -> list[str]: ...
    def to_prefix_syntactic(self, tokens: list[str]) -> list[str]: ...
    def check_form(self, tokens: list[str]) -> None: ...
    def infix_to_prefix(self, infix_expression: str) -> list[str]: ...
    def convert_expression(self, prefix_expr: list[str]) -> list[str]: ...
    def parse(
        self, infix_expression: str, convert_expression: bool = ..., mask_numbers: bool = ...,
        strict: bool = ...
    ) -> list[str]: ...
    def operators_to_realizations(self, tokens: list[str]) -> list[str]: ...
    def realizations_to_operators(self, tokens: list[str]) -> list[str]: ...
    def evaluate_constant_subtree(self, tokens: list[str]) -> str | None: ...
    @staticmethod
    def py_float_repr(x: float) -> str: ...
    def evaluate_batch(
        self,
        tokens: list[str],
        var_names: list[str],
        x_flat: list[float],
        n_rows: int,
        params: list[float],
    ) -> list[float]: ...
    @staticmethod
    def allclose(
        a: list[float], b: list[float], rtol: float = ..., atol: float = ...
    ) -> bool: ...
    def equivalent_no_const(
        self,
        source: list[str],
        candidate: list[str],
        var_names: list[str],
        x_flat: list[float],
        n_rows: int,
        challenges: int = ...,
        rtol: float = ...,
        atol: float = ...,
        min_informative: int | None = ...,
        seed: int = ...,
    ) -> bool: ...
    def interval_finite_ae(self, tokens: list[str]) -> bool: ...
    def interval_class(self, tokens: list[str]) -> str: ...
    def interval_value_components(
        self, tokens: list[str]
    ) -> tuple[bool, bool, bool, bool] | None: ...
    def interval_horizon_misses(self) -> int: ...
    def interval_node_budget_misses(self) -> int: ...
    def interval_unanalyzable_misses(self) -> int: ...
    #: Rewrite steps the step cap refused, process-wide (expected 0).
    def step_cap_binds(self) -> int: ...
    def interval_value_set_box(
        self, tokens: list[str], los: list[float], his: list[float], real: bool = False
    ) -> tuple[bool, bool, bool, bool, float, float] | None: ...
    def interval_domain_extension(
        self, source: list[str], target: list[str]
    ) -> float | None: ...
    def mining_progress(self) -> tuple[int, int]: ...
    def build_candidate_library(
        self,
        candidates: list[list[str]],
        var_names: list[str],
        x_flat: list[float],
        n_rows: int,
        *,
        rule_mode: str,
    ) -> CandidateLibrary: ...
    def find_rule_lib(
        self,
        source: list[str],
        library: CandidateLibrary,
        max_target: int | None = ...,
        challenges: int = ...,
        retries: int = ...,
        seed: int = ...,
        rtol: float = ...,
        atol: float = ...,
        min_informative: int | None = ...,
        *,
        rule_mode: str,
        confirm_x: list[float] | None = ...,
        confirm_rows: int = ...,
        confirm_min_informative: int | None = ...,
        confirm_seed: int = ...,
    ) -> list[str] | None: ...
    def judge_hint(
        self,
        source: list[str],
        hint: list[str],
        library: CandidateLibrary,
        challenges: int = ...,
        retries: int = ...,
        seed: int = ...,
        rtol: float = ...,
        atol: float = ...,
        min_informative: int | None = ...,
        *,
        rule_mode: str,
        confirm_x: list[float] | None = ...,
        confirm_rows: int = ...,
        confirm_min_informative: int | None = ...,
        confirm_seed: int = ...,
    ) -> tuple[bool, str]: ...
    def judge_pair(
        self,
        source: list[str],
        target: list[str],
        var_names: list[str],
        x_flat: list[float],
        n_rows: int,
        challenges: int = ...,
        retries: int = ...,
        seed: int = ...,
        rtol: float = ...,
        atol: float = ...,
        min_informative: int | None = ...,
        *,
        rule_mode: str,
    ) -> tuple[bool, str]: ...
    def set_mode_rules(
        self, rule_mode: str, rules: list[tuple[list[str], list[str]]] | None, *,
        canon: str | None = ...
    ) -> None: ...
    def rules_in_sync(self, rules: list[tuple[list[str], list[str]]]) -> bool: ...
    def mine_one_length(
        self,
        sources: list[list[str]],
        library: CandidateLibrary,
        max_target: int | None = ...,
        challenges: int = ...,
        retries: int = ...,
        seed: int = ...,
        rtol: float = ...,
        atol: float = ...,
        min_informative: int | None = ...,
        relaxed_kruskal: bool = ...,
        *,
        rule_mode: str,
        index_offset: int = ...,
        # the global source index of each entry (default: index_offset + position)
        indices: list[int] | None = ...,
        # global source index -> targets the judge loop refused for that source
        exclusions: dict[int, list[list[str]]] | None = ...,
        confirm_x: list[float] | None = ...,
        confirm_rows: int = ...,
        confirm_min_informative: int | None = ...,
        confirm_seed: int = ...,
    ) -> list[tuple[int, list[str], list[str]]]: ...
    def exist_constants_fit_linear(
        self,
        candidate: list[str],
        var_names: list[str],
        x_flat: list[float],
        n_rows: int,
        y_target: list[float],
        rtol: float = ...,
        atol: float = ...,
    ) -> bool | None: ...
    def exist_constants_fit(
        self,
        candidate: list[str],
        var_names: list[str],
        x_flat: list[float],
        n_rows: int,
        y_target: list[float],
        rtol: float = ...,
        atol: float = ...,
        n_restarts: int = ...,
        seed: int = ...,
    ) -> bool: ...

def core_serialization_ops() -> dict[str, dict[str, Any]]: ...
#: Set the interval layer's node budget (None: the default); returns the previous one.
def set_interval_node_budget(budget: int | None) -> int: ...
#: The numeral grammar (`rust/utils.rs`), the one definition `simplipy.utils` calls.
def is_numeric_string(token: str) -> bool: ...
#: Reserved numeric spellings (`inf`, `1_000`, `0x10`, `1/0`, `--5`): values to a standard
#: numeric reader or malformed numerals, refused at every boundary; the one definition
#: `simplipy.utils` calls.
def reserved_numeric_spelling(token: str) -> bool: ...
#: The f64 realisation bar of a rule spelled with a 0.15 head (`rust/battery.rs`): None when the
#: ordinary realisation axis decides, else the deployed bar's (agree, probes, first differing
#: probe). The contract judge ANDs it into `realised`; the f64 mine applies the same function.
def f64_bar(lhs: list[str], rhs: list[str], var_names: list[str],
            arity: dict[str, int]) -> tuple[bool, int, list[float] | None] | None: ...
#: The f64 bar as the contract judge reads it: each variable probed over its slot sort's values
#: (`sorts[i]` one of `?!$_`), and the probes that differ where an operation overflowed or
#: underflowed returned for the judge: (agree, probes, first differing probe, range probes).
def f64_bar_sorted(lhs: list[str], rhs: list[str], var_names: list[str], sorts: str,
                   arity: dict[str, int]
                   ) -> tuple[bool, int, list[float] | None, list[list[float]]] | None: ...
#: The ground-composite policy on a rule's right side (a variable-free composite).
def policy_ground_composite(rhs: list[str]) -> bool: ...
#: The no-duplication policy: does the right side use a `_` slot more often than the left?
def policy_duplicates(lhs: list[str], rhs: list[str]) -> bool: ...
#: The judge's measure (clause (b)): the volume of the largest certified cell where `source` is
#: undefined and `target` has a value, read in the table's real values; None when undecided.
def table_extension_width(source: list[str], src_params: list[float], target: list[str],
                          tgt_params: list[float], arity: dict[str, int]) -> float | None: ...
#: Clause (c)'s measure: is the set where `tokens` is +-inf certified null (every constant)?
def table_infinite_null(tokens: list[str], arity: dict[str, int]) -> bool: ...
#: Permissive's own simplifications (spec 4.9), as the core names them.
def permissive_simplifications() -> list[str]: ...
#: Evidence (policy 4.1): is `tokens` finite on a positive-measure set of its variables and
#: constants, in float64's values (`f64`) or the table's real ones? None when undecided.
def table_evidence(tokens: list[str], arity: dict[str, int], f64: bool) -> bool | None: ...
#: THE TABLE (`rust/table.rs`): an expression compiled against each mode's values.
class TableExpr:
    def __init__(self, tokens: list[str], names: list[str], arity: dict[str, int]) -> None: ...
    @property
    def names(self) -> list[str]: ...
    def eval_real(
        self, env: list[tuple[str, str, str]], dps: int
    ) -> tuple[tuple[str, str, str], bool, bool, bool, tuple[str, str, str],
               tuple[str, str, str]]: ...
    def eval_float_resolution(
        self, env: list[tuple[str, str, str]], dps: int
    ) -> tuple[str, str, str] | None: ...
    def eval_f64(self, env: list[float]) -> tuple[float, bool]: ...
    #: `eval_f64` at many points (`cols[i]` the i-th name's value per row): values, in-range flags.
    def eval_f64_rows(self, cols: list[list[float]]) -> tuple[list[float], list[bool]]: ...
#: The table's real reading of a literal token: (kind, a, b), or None.
def table_literal_real(token: str) -> tuple[str, str, str] | None: ...
#: The table's f64 reading of a literal token: the nearest double, or None.
def table_literal_f64(token: str) -> float | None: ...
#: The table's quantifier of a slot sort: [(spec, tag)], the tag `real`, `atom` or `null`.
def table_sort_atoms(sort: str, permissive: bool) -> list[tuple[tuple[str, str, str], str]]: ...
#: The table's class comparison of two values' kinds, or None when both are finite.
def table_compare_classes(left: str, right: str) -> str | None: ...
#: The table's f64 agreement of two doubles (within 8 units in the last place).
def table_f64_agree(a: float, b: float) -> str: ...
#: The structural f64 comparison (has the deployed algebra diverged?).
def table_f64_structural(a: float, b: float) -> str: ...
#: The units in the last place between two doubles; None across a sign change.
def table_ulp_gap(a: float, b: float) -> int | None: ...
TABLE_REALISED_ULP: int
#: P4's move (`table::ISOLATION_STEP`): the fraction of a slot's magnitude it moves off a point.
TABLE_ISOLATION_STEP: float
#: The f64 range event (`rust/numeric.rs`): did the operation `op` on `args` overflow or
#: underflow to `value`? None for an operator outside the table. The one definition the
#: miner's scan and `simplipy.verify` share.
def range_event(op: str, args: list[float], value: float) -> bool | None: ...
#: Every operator's float64 kernel (`rust/numeric/kernels.rs`), element-wise over aligned,
#: C-contiguous float64 buffers with the GIL released: `out[i] = name(x[i])`. An operand of one
#: element is broadcast; `out` must be writable and overlap no operand. ValueError for an
#: unknown kernel or mismatched lengths.
def kernel_unary(name: str, x: Any, out: Any) -> None: ...
def kernel_binary(name: str, a: Any, b: Any, out: Any) -> None: ...
#: The scalar paths of the two maps: the same kernels on one float (ValueError for an unknown name).
def kernel_unary_scalar(name: str, x: float) -> float: ...
def kernel_binary_scalar(name: str, a: float, b: float) -> float: ...
