"""Create components."""
import fire
from edo.create_wrapper_components import create_wrapper_components

from edo.cookiecutter_sync import get_data, train, evaluate, analyze
from edo.cookiecutter_sync._utils import (
    get_output_path, get_params)
from edo.cookiecutter_sync.train_me.train_hp_vertex import train_hp_vertex


def create_components(filename='src/edo/cookiecutter_sync/train_me/components.py'):
    """Create components."""
    components = [get_data, get_output_path, get_params,
                  analyze, train, train_hp_vertex, evaluate,
                  ]
    print(f"{filename=}")
    create_wrapper_components(components, install_kfp_package=False, filename=filename)


def main():
    """Execute main program."""
    fire.Fire(create_components)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
