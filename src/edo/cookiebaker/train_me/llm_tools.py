"""Main module."""

from edo.mkt.ml.llm import tool


@tool
def code_reader(code: str):
    """Reads the provided Python code.

    Args:
        code: A string containing the Python code to be read.
        """
    print('code_interpreter')
    code = code.strip()
    print(code)
    return code
