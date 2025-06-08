import argparse
import sys


def main(argv: str = None) -> None:
    parser = argparse.ArgumentParser(description='Neural Symbolic Regression')
    subparsers = parser.add_subparsers(dest='command_name', required=True)

    find_simplifications_parser = subparsers.add_parser("find-simplifications")
    find_simplifications_parser.add_argument('-e', '--expression-space', type=str, required=True, help='Path to the expression space configuration file')
    find_simplifications_parser.add_argument('-n', '--max_n_rules', type=int, default=None, help='Maximum number of rules to find')
    find_simplifications_parser.add_argument('-l', '--max_pattern_length', type=int, default=7, help='Maximum length of the patterns to find')
    find_simplifications_parser.add_argument('-t', '--timeout', type=int, default=None, help='Timeout for the search of simplifications in seconds')
    find_simplifications_parser.add_argument('-d', '--dummy-variables', type=int, nargs='+', default=None, help='Dummy variables to use in the simplifications')
    find_simplifications_parser.add_argument('-m', '--max-simplify-steps', type=int, default=5, help='Maximum number of simplification steps')
    find_simplifications_parser.add_argument('-x', '--X', type=int, default=1024, help='Number of samples to use for comparison of images')
    find_simplifications_parser.add_argument('-c', '--C', type=int, default=1024, help='Number of samples of constants to put in to placeholders')
    find_simplifications_parser.add_argument('-r', '--constants-fit-retries', type=int, default=5, help='Number of retries for fitting the constants')
    find_simplifications_parser.add_argument('-o', '--output-file', type=str, required=True, help='Path to the output json file')
    find_simplifications_parser.add_argument('-s', '--save-every', type=int, default=100, help='Save the simplifications every n rules')
    find_simplifications_parser.add_argument('--reset-rules', action='store_true', help='Reset the rules before finding new ones')
    find_simplifications_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')

    # Evaluate input
    args = parser.parse_args(argv)

    # Execute the command
    match args.command_name:
        case 'find-simplifications':
            if args.verbose:
                print(f'Finding simplifications with expression space {args.expression_space}')
            import os
            from simplipy import SimpliPyEngine
            from simplipy.utils import substitute_root_path

            expression_space = SimpliPyEngine.from_config(substitute_root_path(args.expression_space))

            resolved_output_file = substitute_root_path(args.output_file)

            if not os.path.exists(os.path.dirname(resolved_output_file)):
                os.makedirs(os.path.dirname(resolved_output_file), exist_ok=True)

            expression_space.find_rules(
                max_n_rules=args.max_n_rules,
                max_pattern_length=args.max_pattern_length,
                timeout=args.timeout,
                dummy_variables=args.dummy_variables,
                max_simplify_steps=args.max_simplify_steps,
                X=args.X,
                C=args.C,
                constants_fit_retries=args.constants_fit_retries,
                output_file=resolved_output_file,
                save_every=args.save_every,
                reset_rules=args.reset_rules,
                verbose=args.verbose)

        case _:
            parser.print_help()
            sys.exit(1)


if __name__ == '__main__':
    main()
