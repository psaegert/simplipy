import argparse
import sys
import os
from simplipy import SimpliPyEngine
from simplipy.utils import substitute_root_path, load_config
from simplipy.asset_manager import (
    install_asset, remove_asset, list_assets, get_asset_path
)


def main(argv: str = None) -> None:
    parser = argparse.ArgumentParser(description='SimpliPy CLI Tool')
    subparsers = parser.add_subparsers(dest='command_name', required=True)

    find_simplifications_parser = subparsers.add_parser("find-rules")
    find_simplifications_parser.add_argument(
        '-e', '--engine', type=str, required=True,
        help='Name of an official engine (e.g., dev_7-3) or a local path to an engine configuration file'
    )
    find_simplifications_parser.add_argument('-c', '--config', type=str, required=True, help='Path to the rule-finding configuration file')
    find_simplifications_parser.add_argument('-o', '--output-file', type=str, required=True, help='Path to the output json file')
    find_simplifications_parser.add_argument('-s', '--save-every', type=int, default=100_000, help='Save the simplifications every n rules')
    find_simplifications_parser.add_argument('--reset-rules', action='store_true', help='Reset the rules before finding new ones')
    find_simplifications_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')

    # Install command
    install_parser = subparsers.add_parser("install", help="Install official assets from Hugging Face")
    install_parser.add_argument('name', type=str, help='Name of the asset to install')
    install_parser.add_argument('--type', choices=['ruleset', 'test-data'], default='ruleset', help='Type of asset to install')
    install_parser.add_argument('--force', action='store_true', help='Force reinstall even if already installed')

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove installed assets")
    remove_parser.add_argument('name', type=str, help='Name of the asset to remove')
    remove_parser.add_argument('--type', choices=['ruleset', 'test-data'], default='ruleset', help='Type of asset to remove')

    # List command
    list_parser = subparsers.add_parser("list", help="List available or installed assets")
    list_parser.add_argument('--type', choices=['ruleset', 'test-data', 'all'], default='all', help='Type of asset to list')
    list_parser.add_argument('--installed', action='store_true', help='List only installed assets')

    args = parser.parse_args(argv)

    # Execute the command
    match args.command_name:
        case 'find-rules':
            # Resolve engine name to a config path (will auto-install if needed)
            engine_config_path = get_asset_path('ruleset', args.engine)
            if not engine_config_path:
                sys.exit(1)  # get_asset_path prints the error

            if args.verbose:
                print(f'Finding simplifications with engine {engine_config_path}')

            # SimpliPyEngine.from_config now receives a guaranteed valid path
            engine = SimpliPyEngine.from_config(engine_config_path)

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
                constants_fit_retries=rule_finding_config['constants_fit_retries'],
                output_file=resolved_output_file,
                save_every=args.save_every,
                reset_rules=args.reset_rules,
                verbose=args.verbose)

        case 'install':
            if not install_asset(args.type, args.name, force=args.force):
                sys.exit(1)

        case 'remove':
            if not remove_asset(args.type, args.name):
                sys.exit(1)

        case 'list':
            if args.type in ['ruleset', 'all']:
                list_assets('ruleset', installed_only=args.installed)
            if args.type == 'all':
                print()  # Spacer
            if args.type in ['test-data', 'all']:
                list_assets('test-data', installed_only=args.installed)

        case _:
            parser.print_help()
            sys.exit(1)


if __name__ == '__main__':
    main()
