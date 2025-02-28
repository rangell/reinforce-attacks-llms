"""Helper for local execution."""

import argparse
import json
import os
import shlex
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import warnings

from seml.experiment.config import read_config, generate_configs


parser = argparse.ArgumentParser(
    description='Execute experiments contained in yaml.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--config-file', type=str,
                    default=os.path.join('experiments', 'gcg_reinforce.yaml'),
                    help='Config YAML files. The script deduplicates the configs, but does not check them.')
parser.add_argument('--kwargs', type=json.loads, default='{}', help='Will overwrite the loaded config')
parser.add_argument('--output-file', type=str, default='./commands.sh',
                    help="File that will contain all the commands for each experiment.")


def main(args: argparse.Namespace):
    print(f'Parsing config file {args.config_file}')
    configs, executable = build_configs([args.config_file], None, args.kwargs)

    print(f'Writing commands to {args.output_file} ...')
    with open(args.output_file, "w") as f:
        for i, config in enumerate(configs):
            command = f'python {executable} with overwrite={i} '
            command += " ".join([f'{k}={shell_escape(v)}' for k, v in config.items()])
            command += '\n'
            f.write(command)
    print('... done')


def build_configs(config_files: Sequence[str], executable: Optional[str] = None,
                  kwargs: Dict[str, Any] = {}) -> Tuple[List[Dict[str, Any]], Callable]:
    """Returns all (deduplicated) configs provided in `config_files`. You can, e.g., pass the
    config via the `config_updates` argument (see Example below).

    Parameters
    ----------
    config_files : Sequence[str]
        Config (`.yaml`) files of same experiment (all must refer to the same potentially provided executable).
    executable : str, optional
        Optionally the name of the executable, by default None.
    kwargs : Dict[str, Any], optional
        Overwrite/add certain configs (please make sure they are valid!), by default {}.

    Returns
    -------
    Tuple[List[Dict[str, Any]], str]
        Configs for the experiements and the path to the executable.
    """
    configs = []
    executable = None
    for config_file in config_files:
        seml_config, _, experiment_config = read_config(config_file)
        if executable is None:
            executable = seml_config['executable']
        elif executable != seml_config['executable']:
            raise ValueError(
                f'All configs must be for the same executable! Found {executable} and {seml_config["executable"]}.'
            )
        configs.extend(generate_configs(experiment_config))

    # Overwrite/add configs
    for key, value in kwargs.items():
        for config in configs:
            config[key] = value

    deduplicate_index = {
        json.dumps(config, sort_keys=True): i
        for i, config
        in enumerate(configs)
    }
    configs = [configs[i] for i in deduplicate_index.values()]

    return configs, executable


def shell_escape(value: Any) -> str:
    """Escape a value for shell execution.

    Parameters
    ----------
    config_files : Any
        Value to be escaped.
    """
    if isinstance(value, list) or isinstance(value, dict):
        value = json.dumps(value)

    if isinstance(value, (int, float, bool)):
        return str(value)  # Numbers are safe as-is
    elif isinstance(value, str):
        return shlex.quote(value)  # Properly escape strings
    else:
        warnings.warn(f'Unhandled type {type(value)} for value {value} in `shell_escape`.')
        return str(value)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    main(args)
