"""Analyze the data."""
import os

import fire
import importlib.resources as pkg
from edo.create_wrapper_components import BunchKfp, Output, HTML
from edo.mkt import ml, bq


def analyze(output_path: str,
            data_path: str = None,  # noqa
            report: bool = True,
            target_field: str = 'LABEL',
            artifact_html: Output[HTML] = None) -> str:
    """Analyze data."""
    print('analyze\n')
    if artifact_html is None:
        artifact_html = BunchKfp()
    output_path = os.path.join(output_path, 'analyze')

    if report:
        args_notebook = dict(output_path=output_path,
                             data_path=data_path,
                             target_field=target_field)

        repo_path = pkg_resources.resource_filename(__name__, '')
        path = os.path.join(output_path, 'analyze.html')

        with bq.blob_open(path, 'w', encoding='utf-8') as f:
            ml.render_step('analyze_notebook', f, args_notebook=args_notebook,
                           repo_path=repo_path,
                           args_notebook_file='tmp/args_notebook.yml',
                           exclude_input=True)
        artifact_html.path = path
        return path
    return


def main():
    """Execute main program."""
    fire.Fire(analyze)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
