import pytest

from edo.cookiecutter_sync.cookiecutter_sync import cookiecutter_sync


def test_cookiecutter_sync():
    with pytest.raises(TypeError):
        cookiecutter_sync()


