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
    from anaphora import Col, with_column

    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]}, index=list('abc'))

    actual = with_column(df, 'z', Col('x') * 10)
    expected = df.copy()
    expected['z'] = df['y'] * 10
    pdt.assert_frame_equal(actual, expected)

    with pytest.skip('TODO -- fix interaction between subsetting and adding new columns'):
        actual = with_column(df, 'z', Col('x') * 10)
        expected = df.copy()
        expected['z'] = df['y'] * 10
        pdt.assert_frame_equal(actual, expected)

        actual = with_column(df, 'z', Col('x') * 10, loc=['a', 'b'])
        expected = df.copy()
        expected.loc[['a', 'b'], 'z'] = df.loc[['a', 'b'], 'y'] * 10
        pdt.assert_frame_equal(actual, expected)


def test_mutate():
    from anaphora import Col, mutate, mutate_sequential

    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]})
    expected = pd.DataFrame({'y': [0,1,2], 'x': [5,6,7]})
    actual = mutate(df, y=Col('x')-1, z=Col('y')+1)

    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]})
    expected = pd.DataFrame({'y': [0,1,2], 'x': [1,2,3]})
    actual = mutate(df, y=Col('x')-1, z=Col('y')+1)

    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]})
    expected = df
    mutate(df, y=Col()-1)
    actual = df
    pdt.assert_frame_equal(expected, actual)  # df should remain unchanged

