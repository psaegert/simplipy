# API reference

This page renders the declared public surface: the names in each module's
`__all__`. Names that are reachable but not declared carry no stability promise
and are not rendered here — see the
[compatibility policy](compatibility.md) for what that means in practice.

Most work goes through `SimpliPyEngine`. `simplipy.verify`, `simplipy.promotion`
and `simplipy.mining` are power-user surfaces with documented sharp edges.

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

<!-- Only the declared surface is rendered; the undeclared helpers in this
module are omitted deliberately. -->
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

<!-- The promotion internals stay private; only the entry point is public. -->
::: simplipy.promotion
    options:
      heading_level: 3
      show_root_toc_entry: false
      members:
        - promote
