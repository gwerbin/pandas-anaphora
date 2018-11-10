Anaphoric functions for columns in Pandas data frames

Inspired by the Spark/PySpark dataframe API, with a wistful eye towards Dplyr in R.

This is ALPHA software, use with caution.

## Installation

Pip install coming soon.

Tested with:
- Python 3.6 (should work with 3.5 and probably 3.4)
- Pandas 0.23 (will probably not work in older versions, maybe >= 0.20 is fine)

## API

- class `Col`
- function `with_column`
- function `mutate`
- function `mutate_inplace`
- function `anaphora_register_methods`

See docstrings for more info (for now)

## Example

With this setup:

```python
import pandas as pd
from pandas_anaphora import register_anaphora, Col
register_anaphora_methods()

data = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]}, index=['this', 'that', 'other'])
```

The old way:

```python
b_above_4 = data1['b'] > 4
data1.loc[b_above_4, 'x'] = data1.loc[b_above_4, 'a']

a_is_3 = data1['a']
data1.loc[a_is_3, 'a'] = 300

data1.loc['this', 'b'] = -data1.loc['this', 'b']
```

The Anaphora way:

```
data2 = data1\
    .with_column('x', Col('a'), Col('b') > 4)\
    .with_column('a', 300, loc=Col() == 3)\
    .with_column('b', -Col('b').loc['this'])
```
