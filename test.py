import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt

import pytest


def test_col():
    from anaphora import Col

    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]})
    actual = Col('y').values(df)
    expected = df['y'].values
    assert isinstance(actual, np.ndarray)
    npt.assert_array_equal(actual, expected)

    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]})
    actual = Col('y').map({1: '1', 2: '2'}).astype('category')(df)
    expected = df['y'].map({1: '1', 2: '2'}).astype('category')
    pdt.assert_series_equal(actual, expected)

    actual = ((Col('y') + 3) * 10)(df)
    expected = (df['y'] + 3) * 10
    pdt.assert_series_equal(actual, expected)


def test_with_column():
    from anaphora import Col, anaphora_options, with_column

    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]}, index=list('abc'))

    with pytest.skip('TODO -- fix interaction between subsetting and adding new columns'):
        with anaphora_options(copy=True):
            actual = with_column(df, 'z', Col('x') * 10)
            expected = df.copy()
            expected['z'] = df['y'] * 10
            pdt.assert_frame_equal(actual, expected)

            actual = with_column(df, 'z', Col('x') * 10, loc=['a', 'b'])
            expected = df.copy()
            expected.loc[['a', 'b'], 'z'] = df.loc[['a', 'b'], 'y'] * 10
            pdt.assert_frame_equal(actual, expected)


def test_mutate():
    from anaphora import anaphora_options, Col, mutate, mutate_sequential

    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]})
    expected = pd.DataFrame({'y': [0,1,2], 'x': [5,6,7]})
    actual = mutate(df, y=Col('x')-1, z=('y')+1)

    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]})
    expected = pd.DataFrame({'y': [0,1,2], 'x': [1,2,3]})
    actual = mutate(df, y=Col('x')-1, z=('y')+1)

    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]})
    expected = df
    with anaphora_options(copy=True):
        mutate(df, y=Col()-1)
    actual = df
    pdt.assert_frame_equal(expected, actual)  # df should remain unchanged

    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]})
    expected = df.copy()
    with anaphora_options(copy=False):
        mutate(df, y=Col()-1)
    try:
        pdt.assert_frame_equal(expected, actual)  # df should be changed
    except AssertionError:
        pass
    else:
        raise AssertionError


def test_options():
    from anaphora import anaphora_options, anaphora_get_options

    with anaphora_options(copy=True):
        assert anaphora_get_options('copy')

    with anaphora_options(copy=False):
        assert not anaphora_get_options('copy')
