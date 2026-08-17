## SimpliPy Engine

::: simplipy.engine
    options:
      heading_level: 3
      show_root_toc_entry: false

## Asset Management

::: simplipy.asset_manager
    options:
      heading_level: 3
      show_root_toc_entry: false

## Operators

::: simplipy.operators
    options:
      heading_level: 3
      show_root_toc_entry: false

## Utilities

<!-- DP-C (D11 column, ratified 2026-08-16): only the declared surface is
rendered; being rendered here was never a stability promise, and the
undeclared helpers are now omitted to keep the reference honest. -->
::: simplipy.utils
    options:
      heading_level: 3
      show_root_toc_entry: false
      members:
        - codify
        - deduplicate_rules
        - explicit_constant_placeholders
        - remap_expression
        - substitute_constants
        - construct_expressions
        - numbers_to_constant
        - is_numeric_string
        - enumerate_expressions
        - count_expressions
        - sample_expression
        - compositions

## Normalization

::: simplipy.normalization
    options:
      heading_level: 3
      show_root_toc_entry: false

## I/O Functions

::: simplipy.io
    options:
      heading_level: 3
      show_root_toc_entry: false

## Masking

::: simplipy.masking
    options:
      heading_level: 3
      show_root_toc_entry: false

## Trust

::: simplipy.trust
    options:
      heading_level: 3
      show_root_toc_entry: false

## Compatibility

::: simplipy.compat
    options:
      heading_level: 3
      show_root_toc_entry: false

## Mining

::: simplipy.mining
    options:
      heading_level: 3
      show_root_toc_entry: false

## Verification

::: simplipy.verify
    options:
      heading_level: 3
      show_root_toc_entry: false

## Promotion

<!-- The promotion internals (_ladder, _hp_equiv, ...) are a verbatim external
port and stay private; only the entry point is public (D11 column R29). -->
::: simplipy.promotion
    options:
      heading_level: 3
      show_root_toc_entry: false
      members:
        - promote