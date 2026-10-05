"""The simulated "python loader.py" of the scenario tests must not accept wrong fixes."""

import pytest

from utils import simulate_loader


@pytest.mark.parametrize("source, ok, error", [
    ("print('time:' + time.time())", False, "NameError"),
    ("import time\nprint('time:' + time.time())", False, "TypeError"),
    ("import time\nprint('time:' + str(time.time()))", True, None),
    # the fix of the bash-only run of article 02-b: validated by the old substring check, wrong for real
    ("from time import time\nprint('time:' + str(time.time()))", False, "AttributeError"),
    ("from time import time\nprint('time:' + str(time()))", True, None),
    ("import time\nprint('time:' + str(time.time())", False, "SyntaxError"),
])
def test_simulation_matches_python(source, ok, error):
    result, output = simulate_loader(source)
    assert result is ok
    if error:
        assert output.startswith(error)


def test_py_compile_only_compiles():
    assert simulate_loader("print('time:' + time.time())", compile_only=True) == (True, "")
