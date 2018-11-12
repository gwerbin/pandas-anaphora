Anaphoric functions for columns in Pandas data frames

Inspired by the Spark/PySpark dataframe API, with a wistful eye towards Dplyr in R.

This is ALPHA software, use with caution.

# Example

With this setup:

```python
import pandas as pd
from pandas_anaphora import register_anaphora_methods, Col
register_anaphora_methods()

data = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]}, index=['this', 'that', 'other'])
```

Let's say you want to do the following operations:

1. Create a new column `"x"` that is equal to `"a"` when `"b"` is greater than 4, and null otherwise
2. When column `"a"` is 3, replace it with 300.
3. At the index `"this"`, replace the the value of column `"b"` with its negative

This is how you might do it in typical Pandas code:

```python
b_above_4 = data['b'] > 4
data.loc[b_above_4, 'x'] = data.loc[b_above_4, 'a']

a_is_3 = data['a']
data.loc[a_is_3, 'a'] = 300

data.loc['this', 'b'] = -data.loc['this', 'b']
```

And this is how you would do it with Anaphora:

```python
data = data\
    .with_column('x', Col('a'), Col('b') > 4)\
    .with_column('a', 300, loc=Col() == 3)\
    .with_column('b', -Col('b').loc['this'])
```

In my opinion, the latter is easier to read, and easier to write in an interactive session like a Jupyter notebook.


# Installation

```bash
pip install pandas-anaphora
```

Tested with:
- Python 3.6 (should work with 3.5 and probably 3.4)
- Pandas 0.23 (will probably not work in older versions, maybe >= 0.20 is fine)

Note that the [Toolz](https://pypi.org/project/toolz/) library is required, but if
[Cytoolz](https://pypi.org/project/cytoolz/) is available it will be used instead. This could improve
performance if you are reusing a single `Col()` many times inside a loop.

# API

- _class_ `Col`: an "anaphoric" placeholder for a column in a data frame. An instance of `Col` is a callable that
  keeps track of operations applied to it. For now, only `pandas.Series` methods are tracked; in the future, there
  will be a way to make any function "`Col`-aware", analogous to registering a UDF in (Py)Spark. When a `Col`-aware
  method is used on a `Col`, the return value is always another `Col`. A column spec can be any indexer accepted by
  `pd.DataFrame.loc`.
- _function_ `with_column`: like the (Py)Spark `withColumn` method. Create a shallow copy of a data frame with a
  new column appended. The first argument is a data frame, the second argument is a new or existing column index,
  and the third argument is a `Col` object. If the column specification is omitted (i.e. `Col()` without any
  arguments) then the assigned column name is used. Finally, the `loc` and `iloc` keyword arguments specify a
  subset of the data frame, using any allowed indexer for `pd.Dataframe.loc` or `pd.DataFrame.iloc`, respectively.
- _function_ `mutate`: like the Dplyr `mutate` function. Applies multiple `with_column` operations simultaneously.
  Operations are specified with keyword arguments: keys are column names (new or existing), and arguments are
  `Col`s. Mutation operations in the same `mutate` call *cannot* access each others' results.
- _function_ `mutate_sequential`: like `mutate`, but mutation operations *can* access each others' results. Note
  that the order of operations is non-deterministic in Python versions prior to 3.7! This method is probably a bad
  idea and might be removed.
- _function_ `anaphora_register_methods`: register the `with_column`, `mutate`, and `mutate_sequential` methods on
  the `pd.DataFrame` class. This is **module-global**.

See docstrings for more info.
