import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt

import pytest


def test_col():
    from anaphora import Col

    # attribute access
    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]})
    actual = Col('y').values(df)
    expected = df['y'].values
    assert isinstance(actual, np.ndarray)
    npt.assert_array_equal(actual, expected)

    # attribute chaining (!!)
    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]})
    actual = Col('y').values(df).dtype
    expected = df['y'].values.dtype
    npt.assert_array_equal(actual, expected)

    # method chaining
    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]})
    actual = Col('y').map({1: '1', 2: '2'}).astype('category')(df)
    expected = df['y'].map({1: '1', 2: '2'}).astype('category')
    pdt.assert_series_equal(actual, expected)

    # magic method chaining
    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]})
    actual = ((Col('y') + 3) * 10)(df)
    expected = (df['y'] + 3) * 10
    pdt.assert_series_equal(actual, expected)

    # loc, scalar output
    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]}, index=list('abc'))
    actual = Col('x').loc['c'](df)
    expected = 6
    assert int(actual) == expected

    # loc, vector output
    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]}, index=list('abc'))
    actual = Col('x').loc[['a','c']](df)
    expected = pd.Series([4,6], index=['a','c'], name='x')
    pdt.assert_series_equal(actual, expected)


def test_with_column():
    from anaphora import Col, with_column

    # replace a column
    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]}, index=list('abc'))
    actual = with_column(df, 'x', Col() * 10)
    expected = df.copy()
    expected['x'] = df['x'] * 10
    pdt.assert_frame_equal(actual, expected)

    # add a column
    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]}, index=list('abc'))
    actual = with_column(df, 'z', Col('x') * 10)
    expected = df.copy()
    expected['z'] = df['x'] * 10
    pdt.assert_frame_equal(actual, expected)

    # subset with scalar loc
    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]}, index=list('abc'))
    actual = with_column(df, 'z', Col('y') * 10, loc='b')
    expected = df.copy()
    expected.loc['b', 'z'] = df.loc['b', 'y'] * 10
    pdt.assert_frame_equal(actual, expected)

    # subset with scalar iloc
    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]}, index=list('abc'))
    actual = with_column(df, 'z', Col('y') * 10, iloc=1)
    expected = df.copy()
    expected.loc[expected.index[1], 'z'] = df['y'].iloc[1] * 10
    pdt.assert_frame_equal(actual, expected)

    # subset with vector loc
    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]}, index=list('abc'))
    actual = with_column(df, 'z', Col('y') * 10, loc=['a', 'b'])
    expected = df.copy()
    expected.loc[['a', 'b'], 'z'] = df.loc[['a', 'b'], 'y'] * 10
    pdt.assert_frame_equal(actual, expected)

    # subset with vector iloc
    df = pd.DataFrame({'y': [1,2,3], 'x': [4,5,6]}, index=list('abc'))
    actual = with_column(df, 'z', Col('y') * 10, iloc=[1, 2])
    expected = df.copy()
    expected.loc[expected.index[[1, 2]], 'z'] = df['y'].iloc[[1,2]] * 10
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

