import os
import shutil

import numpy as np
import pytest
from skopt.space import Integer

from xpot.loaders import (
    _create_dir,
    autoplex_return,
    convert_numpy_types,
    directory_setup,
    get_defaults,
    get_input_hypers,
    get_optimisable_params,
    list_to_string,
    merge_hypers,
    prep_dict_for_dump,
    reconstitute_lists,
    recursive_items,
    return_best_potential,
    trim_empty_values,
    validate_hypers,
    validate_subdict,
)


def test_create_dir():
    dir = "tests/test_dir"
    _create_dir(dir)
    assert os.path.isdir(dir)


def test_dir_exists_but_empty():
    dir = "tests/test_dir"
    with pytest.warns(Warning):
        _create_dir(dir)


def test_dir_exists_but_not_empty():
    dir = "tests/test_dir"
    open("tests/test_dir/not_empty.txt", "a").close()
    with pytest.raises(FileExistsError):
        _create_dir(dir)
    shutil.rmtree(dir)


def test_returns():
    dir = "tests/mock_runs"
    result = autoplex_return(dir)
    assert result == (2, 0.3, 8)

    result = return_best_potential(dir)
    assert result == "tests/mock_runs/10/"


def test_convert_numpy_str():
    mydict = {"a": {"b": [1, 2, np.str_("3")]}, "c": np.str_("4")}
    result = convert_numpy_types(mydict)
    assert result == {"a": {"b": [1, 2, "3"]}, "c": "4"}


def test_directory_setup():
    dir = "tests/setup_dir"
    directory_setup(dir)
    os.chdir("../..")
    assert os.path.exists(dir)
    shutil.rmtree(dir)


def test_defaults():
    mydefault = "tests/inputs/defaults.json"
    result = get_defaults(mydefault)
    assert result == {
        "cutoff": 5,
        "seed": 42,
        "heinousness": "high",
        "rand_list": [1, 2, 3],
    }


def test_hyper_flow():
    mydefault = "tests/inputs/defaults.json"
    defaults = get_defaults(mydefault)
    assert defaults == {
        "cutoff": 5,
        "seed": 42,
        "heinousness": "high",
        "rand_list": [1, 2, 3],
    }

    myhypers = "tests/inputs/hypers.hjson"
    hypers = get_input_hypers(myhypers)
    assert hypers == {
        "seed": "skopt.space.Integer(1,10)",
        "heinousness": "low",
        "rand_list": [1, 2, 4],
    }
    opts = get_optimisable_params(hypers)
    assert opts == {
        (("seed",),): Integer(
            low=1, high=10, prior="uniform", transform="identity"
        ),
    }

    merged = merge_hypers(defaults, hypers)
    assert merged == {
        "cutoff": 5,
        "seed": "skopt.space.Integer(1,10)",
        "heinousness": "low",
        "rand_list": [1, 2, 4],
    }
    # Check the validation functions for dictionaries.
    if validate_subdict(merged, hypers):
        pass
    else:
        raise ValueError("Subdict not valid")
    validate_hypers(merged, hypers)
    try:
        validate_hypers(defaults, hypers)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Hypers should not be valid, but are not raising an error."
        )


def test_list_to_string():
    mydict = {"a": ["d", "e", "f"], "b": {"c": ["g", "h", "i"]}}
    result = list_to_string(mydict)
    assert result == {"a": "d e f", "b": {"c": "g h i"}}


def test_deconstruct():
    mydict = {
        "a": 5.5,
        "b": {
            "c": "test",
        },
    }
    result = list(recursive_items(mydict))

    assert result == [(("a",), 5.5), (("b", "c"), "test")]


def test_deconstruct_with_list():
    mydict = {
        "a": 5.5,
        "b": {
            "c": [1, 2, 3],
        },
    }
    result = list(recursive_items(mydict))

    assert result == [
        (("a",), 5.5),
        (("b", "c", "0"), 1),
        (("b", "c", "1"), 2),
        (("b", "c", "2"), 3),
    ]


def test_reconstitute_list_with_val():
    mydict = {
        "a": 5.5,
        "b": {
            "c": [1, 2, 3],
        },
    }

    result_replace = {(("b", "c", "2"),): 15}
    result = reconstitute_lists(mydict, result_replace)

    assert result == mydict


def test_numpy_type():
    mydict = {
        "a": np.float64(5.5),
        "b": {
            "c": [1, 2, np.int64(3)],
        },
        "d": np.int64(4),
    }

    result = prep_dict_for_dump(mydict)
    assert result["a"] == 5.5
    assert result["b"]["c"][2] == 3
    assert result["d"] == 4


def test_trim_empty_vals():
    mydict = {
        "a": "",
        "b": {
            "c": [1, 2, 3],
        },
    }

    result = trim_empty_values(mydict)

    with pytest.raises(KeyError):
        return result["a"]
