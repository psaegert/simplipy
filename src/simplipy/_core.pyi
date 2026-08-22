"""Type stub for the compiled Rust inline core (``simplipy._core``).

Hand-written to mirror the ``#[pymethods]`` surface in ``rust/lib.rs``; keep in sync on changes.
Compiled extensions are opaque to static type-checkers, so this stub is what makes the ``Engine``
surface (and the module constants) visible to mypy/pyright downstream and in this package.
(Re-synced 2026-08-09, extreme-lane campaign: the file had drifted several methods behind the
FFI -- ``from_strs`` among them, which mypy flagged on ``engine.py`` -- and the drift class is
exactly what this header warns about.)
"""
from typing import Any, Literal

__build__: str

class CandidateLibrary:
    @property
    def n_candidates(self) -> int: ...
    @property
    def n_filtered(self) -> int: ...

class Engine:
    @staticmethod
    def from_strs(config_yaml: str, rules_json: str) -> Engine: ...
    def ac_simplify(
        self,
        tokens: list[str],
        max_passes: int = ...,
        wildcard_all: bool = ...,
        form: Literal["tagged", "explicit"] = ...,
    ) -> list[str]: ...
    def ac_simplify_in_mode(
        self,
        tokens: list[str],
        max_passes: int = ...,
        rule_mode: Literal["default", "real", "corpus"] = ...,
        form: Literal["tagged", "explicit"] = ...,
    ) -> list[str]: ...
    def ac_simplify_infix(
        self,
        tokens: list[str],
        max_passes: int = ...,
        wildcard_all: bool = ...,
    ) -> str: ...
    def ac_simplify_infix_in_mode(
        self,
        tokens: list[str],
        max_passes: int = ...,
        rule_mode: Literal["default", "real", "corpus"] = ...,
    ) -> str: ...
    def ac_simplify_explore(
        self,
        tokens: list[str],
        max_passes: int = ...,
        wildcard_all: bool = ...,
        form: Literal["tagged", "explicit"] = ...,
        explore_budget: int = ...,
    ) -> list[str]: ...
    def ac_ordered_below(self, a: list[str], b: list[str]) -> bool: ...
    def ac_judge(
        self, tokens: list[str], max_passes: int
    ) -> tuple[int, int, list[str]]: ...
    def ac_canonical_keys(
        self, exprs: list[list[str]]
    ) -> list[list[str] | None]: ...
    def ac_complexity(self, tokens: list[str]) -> int: ...
    def ac_complexity_certified(self, tokens: list[str]) -> int: ...
    def ac_rules_info(self) -> tuple[int, int, int, int]: ...
    def ac_rules_info_in_mode(self, rule_mode: str = ...) -> tuple[int, int, int, int]: ...
    def mode_rules_len(self, rule_mode: str = ...) -> int | None: ...
    def ac_served_rules_in_mode(
        self, rule_mode: str = ...
    ) -> list[tuple[list[str], list[str], int]]: ...
    def ac_registry_dropped(self) -> int: ...
    #: (lhs, rhs, source_index) per SERVED rule -- the matcher's own set, including the
    #: orientation twins minted at load, which no artifact row corresponds to.
    def ac_served_rules(self) -> list[tuple[list[str], list[str], int]]: ...
    def ac_rules_drop_census(self) -> dict[str, int]: ...
    def ac_odd_neg_carriers(
        self, tokens: list[str], max_passes: int
    ) -> list[tuple[list[str], str, bool, bool, list[tuple[list[str], bool, bool]]]]: ...
    def interval_zero_set_null_generic_const(self, tokens: list[str]) -> bool: ...
    def interval_entire_analytic_composition(self, tokens: list[str]) -> bool: ...
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
        challenges: int,
        retries: int,
        seed: int,
        rtol: float,
        atol: float,
        min_informative: int | None,
    ) -> list[str] | None: ...
    def registry_mint_refusal(
        self, source: list[str], mark: list[str], target: list[str], var_names: list[str]
    ) -> str | None: ...
    def registry_pole_refusals(self) -> int: ...
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
        self, infix_expression: str, convert_expression: bool = ..., mask_numbers: bool = ...
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
    ) -> CandidateLibrary: ...
    def find_rule_lib(
        self,
        source: list[str],
        simplified_length: int,
        max_target: int | None,
        library: CandidateLibrary,
        challenges: int = ...,
        retries: int = ...,
        seed: int = ...,
        rtol: float = ...,
        atol: float = ...,
        min_informative: int | None = ...,
        mark: list[str] | None = ...,
    ) -> list[str] | None: ...
    def set_rules(self, rules: list[tuple[list[str], list[str]]]) -> None: ...
    def set_mode_rules(
        self, rule_mode: str, rules: list[tuple[list[str], list[str]]] | None
    ) -> None: ...
    def rules_in_sync(self, rules: list[tuple[list[str], list[str]]]) -> bool: ...
    def mine_one_length(
        self,
        sources: list[list[str]],
        library: CandidateLibrary,
        max_target: int | None,
        challenges: int = ...,
        retries: int = ...,
        seed: int = ...,
        rtol: float = ...,
        atol: float = ...,
        min_informative: int | None = ...,
        relaxed_kruskal: bool = ...,
    ) -> list[tuple[list[str], list[str]]]: ...
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
