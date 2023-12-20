"""Main module."""
import os
from importlib.resources import files

import fire
from streamlit import config as _config
from streamlit.web.bootstrap import run

from edo import cookiebaker


def run_app(app_name: str = 'app'):
    """robyn app."""
    print('run app')
    print(vars())

    _config.set_option("server.headless", True)

    package_path = files(cookiebaker)
    app_name_ = package_path.joinpath("app", f"{app_name}.py")
    app_name_ = str(app_name_)
    if not os.path.exists(app_name_):
        raise FileNotFoundError(f"File {app_name_} not found")
    run(app_name_, '', [], [])


def main():
    """Execute main program."""
    fire.Fire(run_app)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
