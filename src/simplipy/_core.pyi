"""Type stub for the compiled Rust inline core (``simplipy._core``).

Hand-written to mirror the ``#[pymethods]`` surface in ``rust/lib.rs``; keep in sync on changes.
Compiled extensions are opaque to static type-checkers, so this stub is what makes the ``Engine``
surface (and the module constants) visible to mypy/pyright downstream and in this package.
``simplipy._core`` is internal: it carries no stability promise.
"""
from typing import Any, Literal

__build__: str
DEFAULT_WORK: int | None
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
        work_budget: int | None = ...,  # per search, in ac::work units; None: unbounded
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
    def ac_simplify_infix_in_mode(
        self,
        tokens: list[str],
        max_passes: int = ...,
        explore_budget: int | None = ...,  # None: until a round finds nothing
        work_budget: int | None = ...,  # per search, in ac::work units; None: unbounded
        *,
        rule_mode: Literal["default", "real", "permissive"],
        own_set: bool = ...,
    ) -> str: ...
    def prettify(
        self,
        tokens: list[str],
        digits: int = ...,
        form: Literal["infix", "explicit", "tagged"] = ...,
        resimplify: bool = ...,  # simplify the display once and display that
        explore_budget: int | None = ...,
        work_budget: int | None = ...,
        *,
        rule_mode: Literal["default", "real", "permissive"],
        own_set: bool = ...,
    ) -> str | list[str]: ...
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
        self, tokens: list[str], los: list[float], his: list[float]
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
