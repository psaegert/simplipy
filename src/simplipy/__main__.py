import argparse
import json
import sys
import os
from simplipy import SimpliPyEngine
from simplipy.io import load_config
from simplipy.asset_manager import (
    install_asset, uninstall_asset, list_assets, get_path
)


def main(argv: str = None) -> None:
    parser = argparse.ArgumentParser(description='SimpliPy CLI Tool')
    subparsers = parser.add_subparsers(dest='command_name', required=True)

    find_simplifications_parser = subparsers.add_parser("find-rules")
    find_simplifications_parser.add_argument(
        '-e', '--engine', type=str, required=True,
        help='Name of an official engine (e.g., acj-4-3) or a local path to an engine configuration file'
    )
    find_simplifications_parser.add_argument('-c', '--config', type=str, required=True, help='Path to the rule-finding configuration file')
    find_simplifications_parser.add_argument('-o', '--output-file', type=str, required=True, help='Path to the output json file')
    find_simplifications_parser.add_argument('-s', '--save-every', type=int, default=100_000, help='Save the simplifications every n rules')
    find_simplifications_parser.add_argument('--reset-rules', action='store_true', help='Reset the rules before finding new ones')
    find_simplifications_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')

    # Prune-rules command
    prune_covered_rules_parser = subparsers.add_parser("prune-covered-rules", help="Remove rules that the remaining rules cover behaviorally")
    prune_covered_rules_parser.add_argument(
        '-e', '--engine', type=str, required=True,
        help='Name of an official engine (e.g., acj-4-3) or a local path to an engine configuration file'
    )
    prune_covered_rules_parser.add_argument('-o', '--output-file', type=str, required=True, help='Path to save the pruned rules json file')
    prune_covered_rules_parser.add_argument('-v', '--verbose', action='store_true', help='Print progress information')

    # Resolve-rules command
    resolve_rules_parser = subparsers.add_parser("resolve-rules", help="Replace <constant> with actual numeric values in all-numeric rules")
    resolve_rules_parser.add_argument(
        '-e', '--engine', type=str, required=True,
        help='Name of an official engine (e.g., acj-4-3) or a local path to an engine configuration file'
    )
    resolve_rules_parser.add_argument('-o', '--output-file', type=str, required=True, help='Path to save the resolved rules json file')
    resolve_rules_parser.add_argument('-v', '--verbose', action='store_true', help='Print progress information')

    # Install command
    install_parser = subparsers.add_parser("install", help="Install an official asset from Hugging Face")
    install_parser.add_argument('name', type=str, help='Name of the asset to install (e.g. acj-4-3)')
    install_parser.add_argument('--force', action='store_true', help='Force reinstall even if already installed')

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove an installed asset")
    remove_parser.add_argument('name', type=str, help='Name of the asset to remove (e.g. acj-4-3)')

    # List command
    list_parser = subparsers.add_parser("list", help="List available or installed assets")
    list_parser.add_argument('--type', choices=['engine', 'test-data', 'all'], default='all', help='Type of asset to list')
    list_parser.add_argument('--installed', action='store_true', help='List only installed assets')

    args = parser.parse_args(argv)

    # Execute the command
    match args.command_name:
        case 'find-rules':
            try:
                engine_config_path = get_path(args.engine)
            except (FileNotFoundError, ValueError, RuntimeError) as e:
                print(f'Error: {e}', file=sys.stderr)
                sys.exit(1)

            if args.verbose:
                print(f'Finding simplifications with engine {engine_config_path}')

            # SimpliPyEngine.from_config now receives a guaranteed valid path
            engine = SimpliPyEngine.from_config(engine_config_path)

            output_dir = os.path.dirname(args.output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            rule_finding_config = load_config(args.config)

            # FAIL-CLOSED config validation: a key the CLI does not forward would
            # otherwise be ignored silently, and the mined artifact would not match
            # what its config claims (this shipped once: `prune: covered` never ran).
            known_keys = {
                'max_source_pattern_length', 'max_target_pattern_length',
                'dummy_variables', 'extra_internal_terms', 'n_samples',
                'constants_fit_challenges', 'constants_fit_retries', 'rtol', 'atol',
                'min_informative', 'seed', 'confirm', 'source_sample_per_length',
                'candidate_fold_filter', 'relaxed_kruskal', 'prune', 'proposals',
                'promote_sorts', 'symbolic_gate', 'snapshot_at'}
            unknown_keys = sorted(set(rule_finding_config) - known_keys)
            if unknown_keys:
                print(f'Error: unknown key(s) in {args.config}: {", ".join(unknown_keys)}. '
                      f'Known keys: {", ".join(sorted(known_keys))}', file=sys.stderr)
                sys.exit(1)

            # KEY-AWARE path resolution (D3): `proposals` is an INPUT that lives next to
            # the config, so a relative spelling resolves against the CONFIG's directory
            # (the former io.py value-sniffing pass only caught `./`-prefixed spellings;
            # `proposals.json` silently stayed CWD-relative). `snapshot_at` values are
            # OUTPUTS and resolve against the output file's directory below.
            proposals = rule_finding_config.get('proposals', None)
            if isinstance(proposals, str) and not os.path.isabs(proposals):
                proposals = os.path.join(
                    os.path.dirname(os.path.abspath(args.config)), proposals)

            engine.find_rules(
                max_source_pattern_length=rule_finding_config['max_source_pattern_length'],
                max_target_pattern_length=rule_finding_config['max_target_pattern_length'],
                dummy_variables=rule_finding_config.get('dummy_variables', None),
                extra_internal_terms=rule_finding_config.get('extra_internal_terms', None),
                X=rule_finding_config['n_samples'],
                constants_fit_challenges=rule_finding_config.get('constants_fit_challenges', 16),
                constants_fit_retries=rule_finding_config['constants_fit_retries'],
                rtol=rule_finding_config.get('rtol', 1e-9),
                atol=rule_finding_config.get('atol', 1e-12),
                min_informative=rule_finding_config.get('min_informative', None),
                seed=rule_finding_config.get('seed', 42),
                confirm=rule_finding_config.get('confirm', True),
                source_sample_per_length={
                    int(k): int(v) for k, v in
                    (rule_finding_config.get('source_sample_per_length') or {}).items()},
                candidate_fold_filter=rule_finding_config.get('candidate_fold_filter', True),
                relaxed_kruskal=rule_finding_config.get('relaxed_kruskal', True),
                proposals=proposals,
                promote_sorts=rule_finding_config.get('promote_sorts', True),
                symbolic_gate=rule_finding_config.get('symbolic_gate', True),
                # REQUIRED when the vocabulary omits `mult{k}`/`div{k}`: the cancellation emit
                # site would otherwise intern `mult3` as an undefined overlay token, because its
                # factor ceiling comes from the `pow\d+` operators but it spells the ADDITIVE
                # class with `mult{k}`. Mining with a reduced vocabulary and this flag off
                # produces expressions no consumer can parse.
                # Ladder re-use: {source_length: path} emits the shorter cells of the same
                # `j` as the climb passes through them. Paths are resolved relative to the
                # output file's directory so a config stays portable across boxes.
                snapshot_at={
                    int(k): v if os.path.isabs(v) else os.path.join(
                        os.path.dirname(os.path.abspath(args.output_file)), v)
                    for k, v in (rule_finding_config.get('snapshot_at') or {}).items()},
                output_file=args.output_file,
                save_every=args.save_every,
                reset_rules=args.reset_rules,
                prune=rule_finding_config.get('prune', False),
                verbose=args.verbose)

        case 'prune-covered-rules':
            try:
                engine_config_path = get_path(args.engine)
            except (FileNotFoundError, ValueError, RuntimeError) as e:
                print(f'Error: {e}', file=sys.stderr)
                sys.exit(1)

            engine = SimpliPyEngine.from_config(engine_config_path)
            n_before = len(engine.simplification_rules)
            n_pruned = engine.prune_covered_rules(verbose=args.verbose)

            output_dir = os.path.dirname(args.output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            with open(args.output_file, 'w') as f:
                json.dump(engine.simplification_rules, f, indent=4)

            print(f'Pruned {n_pruned} covered rules ({n_before} -> {len(engine.simplification_rules)})')
            print(f'Saved to {args.output_file}')

        case 'resolve-rules':
            try:
                engine_config_path = get_path(args.engine)
            except (FileNotFoundError, ValueError, RuntimeError) as e:
                print(f'Error: {e}', file=sys.stderr)
                sys.exit(1)

            engine = SimpliPyEngine.from_config(engine_config_path)
            n_resolved = engine.resolve_constant_rules(verbose=args.verbose)

            output_dir = os.path.dirname(args.output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            with open(args.output_file, 'w') as f:
                json.dump(engine.simplification_rules, f, indent=4)

            print(f'Resolved {n_resolved} rules ({len(engine.simplification_rules)} total)')
            print(f'Saved to {args.output_file}')

        case 'install':
            try:
                installed = install_asset(args.name, force=args.force)
            except (FileNotFoundError, ValueError, RuntimeError) as e:
                print(f'Error: {e}', file=sys.stderr)
                sys.exit(1)
            if not installed:
                sys.exit(1)

        case 'remove':
            try:
                removed = uninstall_asset(args.name)
            except (FileNotFoundError, ValueError, RuntimeError) as e:
                print(f'Error: {e}', file=sys.stderr)
                sys.exit(1)
            if not removed:
                sys.exit(1)

        case 'list':
            if args.type in ['engine', 'all']:
                list_assets('engine', installed_only=args.installed)
            if args.type == 'all':
                print()  # Spacer
            if args.type in ['test-data', 'all']:
                list_assets('test-data', installed_only=args.installed)

        case _:
            parser.print_help()
            sys.exit(1)


if __name__ == '__main__':
    main()
