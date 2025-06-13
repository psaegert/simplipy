import argparse
import sys


def main(argv: str = None) -> None:
    parser = argparse.ArgumentParser(description='SimpliPy CLI Tool')
    subparsers = parser.add_subparsers(dest='command_name', required=True)

    find_simplifications_parser = subparsers.add_parser("find-rules")
    find_simplifications_parser.add_argument('-e', '--engine', type=str, required=True, help='Path to the engine configuration file')
    find_simplifications_parser.add_argument('-c', '--config', type=str, required=True, help='Path to the rule-finding configuration file')
    find_simplifications_parser.add_argument('-o', '--output-file', type=str, required=True, help='Path to the output json file')
    find_simplifications_parser.add_argument('-s', '--save-every', type=int, default=10000, help='Save the simplifications every n rules')
    find_simplifications_parser.add_argument('--reset-rules', action='store_true', help='Reset the rules before finding new ones')
    find_simplifications_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')

    # Evaluate input
    args = parser.parse_args(argv)

    # Execute the command
    match args.command_name:
        case 'find-rules':
            if args.verbose:
                print(f'Finding simplifications with engine {args.engine}')
            import os
            from simplipy import SimpliPyEngine
            from simplipy.utils import substitute_root_path, load_config

            engine = SimpliPyEngine.from_config(substitute_root_path(args.engine))

            resolved_output_file = substitute_root_path(args.output_file)

            if not os.path.exists(os.path.dirname(resolved_output_file)):
                os.makedirs(os.path.dirname(resolved_output_file), exist_ok=True)

            rule_finding_config = load_config(substitute_root_path(args.config), resolve_paths=True)

            engine.find_rules(
                max_source_pattern_length=rule_finding_config['max_source_pattern_length'],
                max_target_pattern_length=rule_finding_config['max_target_pattern_length'],
                dummy_variables=rule_finding_config.get('dummy_variables', None),
                extra_internal_terms=rule_finding_config.get('extra_internal_terms', None),
                X=rule_finding_config['n_samples'],
                C=rule_finding_config['n_constants'],
                constants_fit_retries=rule_finding_config['constants_fit_retries'],
                output_file=resolved_output_file,
                save_every=args.save_every,
                reset_rules=args.reset_rules,
                verbose=args.verbose)

        case _:
            parser.print_help()
            sys.exit(1)


if __name__ == '__main__':
    main()
