import os
import subprocess

import fire

from edo.cookiebaker.predict import get_prediction

from git import Repo

def git_diff(file, reference_file, tmp_dir='tmp'):
    # reference_file = os.path.basename(reference_file)
    # file = os.path.basename(file)
    # # os.chdir(tmp_dir)
    # diff_filename = 'diff.txt'
    # cmd = f'sh ./git_diff.sh'
    # print(cmd)
    # subprocess.run(cmd, check=True)
    #
    # with open(diff_filename) as f:
    #     diff = f.read()
    filename = 'diff.txt'
    with open(filename) as f:
        diff = f.read()
    return diff


def test_prompt():
    reference_filename = 'tmp/evaluate_reference.py'
    # with open(reference_filename) as f:
    #     reference_file = f.read()

    filename = 'tmp/evaluate.py'
    # with open(filename) as f:
    #     file = f.read()

    diff = git_diff(filename, reference_filename)
    print(diff)
    p = get_prediction(diff=diff)
    # print(p)

def main():
    """Execute main program."""
    fire.Fire(test_prompt)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
