import os

import pandas as pd
from xpot.convert import pkl2xyz, pot2ase_calc, split_xyz, xyz2pkl


def test_data_converters():
    myfile = "tests/inputs/test.xyz"

    xyz2pkl(myfile, ref_e={"Si": 0})
    assert os.path.exists("tests/inputs/test.xyz.pkl.gzip")
    os.remove("tests/inputs/test.xyz.pkl.gzip")

    xyz2pkl(myfile)
    assert os.path.exists("tests/inputs/test.xyz.pkl.gzip")

    pkl2xyz("tests/inputs/test.xyz.pkl.gzip")
    assert os.path.exists("tests/inputs/test.xyz.pkl.gzip.xyz")

    df = pd.read_pickle("tests/inputs/test.xyz.pkl.gzip", compression="gzip")
    assert df["energy_corrected"].values[0] == 0.5
    assert df["energy_corrected"].values[1] == -2.4

    split_xyz("tests/inputs/test.xyz", test_frac=0.5, by_config=False)
    assert os.path.exists("tests/inputs/test-test.xyz")
    assert os.path.exists("tests/inputs/train-test.xyz")
    os.remove("tests/inputs/test-test.xyz")
    os.remove("tests/inputs/train-test.xyz")

    split_xyz("tests/inputs/test.xyz", test_frac=0.5, by_config=True)
    assert os.path.exists("tests/inputs/test-test.xyz")
    assert os.path.exists("tests/inputs/train-test.xyz")

    # split_pkl("tests/inputs/test.xyz.pkl.gzip", test_frac=0.5,by_config=False)
    # assert os.path.exists("tests/inputs/test-test.xyz.pkl.gzip")
    # assert os.path.exists("tests/inputs/train-test.xyz.pkl.gzip")

    xyz2pkl("tests/inputs/test-test.xyz")
    xyz2pkl("tests/inputs/train-test.xyz")
    assert os.path.exists("tests/inputs/test-test.xyz.pkl.gzip")
    assert os.path.exists("tests/inputs/train-test.xyz.pkl.gzip")


def test_pot2ase_calc():
    # TODO: Implement testing for various potential types
    ...
