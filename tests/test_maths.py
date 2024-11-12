from xpot.maths import get_mae, get_rmse, scale_list_values


def test_get_errors():
    my_data = [4, 2, 4, 2]
    atoms = [2, 2, 2, 1]
    assert get_rmse(my_data) == 10**0.5
    assert get_mae(my_data) == 3.0
    assert scale_list_values(my_data, atoms) == [2, 1, 2, 2]
