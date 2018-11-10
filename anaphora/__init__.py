""" Anaphora for Pandas DataFrame columns (because writing `lambda x:` sucks)

Inspired by the Spark/PySpark dataframe API, with a wistful eye towards Dplyr in R...

Tested with:
- Python 3.6 (should work with 3.5 and probably 3.4)
- Pandas 0.23 (will probably not work in older versions, maybe >= 0.20 is fine)

Example:
    import pandas as pd
    from pandas_anaphora import register_anaphora, Col
    register_anaphora_methods()

    data = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]}, index=['this', 'that', 'other'])

    # the old way
    data1 = data.copy()
    b_above_4 = data1['b'] > 4
    data1.loc[b_above_4, 'x'] = data1.loc[b_above_4, 'a']
    a_is_3 = data1['a']
    data1.loc[a_is_3, 'a'] = 300
    data1.at['this', 'b'] = -data1.at['this', 'b']

    # the new way
    data2 = data1\
        .with_column('x', Col('b') > 4)\
        .with_column('a', 300, subset=Col() == 3)\
        .with_column('b', -Col(), at='this')

    pd.testing.assert_frame_equal(data1, data2)
"""
import operator as op
from copy import copy as py_shallowcopy

try:
    import cytoolz as tz
except ImportError:
    import toolz as tz
import pandas as pd


__all__ = (
    'anaphora_register_methods',
    'Col',
    'with_column',
    'mutate',
    'mutate_sequential',
    'anaphora_options',
    'anaphora_get_options',
    'anaphora_set_options',
)

# heaven help me if option setting ever needs true thread-safety
_ANAPHORA_OPTIONS_LOCKED = False

_ANAPHORA_OPTIONS = {
    'copy': True  # If True, call df.copy() in functions
}


def anaphora_register_methods(include=(), exclude=()):
    """ Register functions as Pandas object methods

    Registered methods:
    - anaphora.with_column -> pandas.DataFrame.with_column
    - anaphora.mutate -> pandas.DataFrame.mutate
    """
    methods = {
        'with_column': with_column,
        'mutate': mutate
    }

    if not include:
        include = set(methods.keys())
    exclude = set(exclude)
    whitelist = include - exclude

    for name, method in methods.items():
        if name in whitelist:
            setattr(pd.DataFrame, name, method)


## Standardized error messages

def _error_unknown_option(key):
    return KeyError('Unknown option: {}'.format(key))


def _error_mutually_exclusive_kwargs(keys):
    pretty_key_list = ', '.join(map("'{]'".format, subset_types.keys()))
    return ValueError('Arguments are mutually exclusive: {}'.format(pretty_key_list))


## Core data type: the anaphoric "Col"

# TODO: should there be an anaphoric Subset for row slicing?

class ColType(type):
    """ Metaclass for Col objects

    Used to explicitly whitelist dunder methods, which are blacklisted by the custom ``Col.__getattr__``. Also,
    Python "magic method" lookup skips both ``__getattr__`` and ``__getattributes__``, and instead looks directly
    on the class. So dunders need to be explicitly present on the class to be recognized by Python. E.g. in
    ``Col('y') + 1``, Python skips both ``Col.__getattributes__('add')`` and ``Col.__getattr__('add')``; it just
    looks for ``Col.__add__``. See https://docs.python.org/3.7/reference/datamodel.html#special-lookup
    """
    class GetterPusher:
        # A descriptor that acts like Col._MethodArgCollector
        def __init__(self, name=None):
            # We only need to have this becuase __set_name__ isn't triggered by setattr() ... bug?
            self.name = name

        def __set_name__(self, cls, name):
            self.name = name

        def __get__(self, col_obj, cls):
            self.col = col_obj
            return self

    class SeriesAttr(GetterPusher):
        def __call__(self, *args, **kwargs):
            self.col.push(op.attrgetter(self.name, *args, **kwargs))
            return self.col

    class SeriesMethod(GetterPusher):
        def __call__(self, *args, **kwargs):
            self.col.push(op.methodcaller(self.name, *args, **kwargs))
            return self.col

    # TODO: do i even need the RHS of this table? i'm not using it right now.
    #       original intent was to use e.g. op.add directly instead of op.methodcaller('add')
    dunder_whitelist = {
        '__add__'      : op.add,
        '__xor__'      : op.xor,
        '__lt__'       : op.lt,
        '__le__'       : op.le,
        '__eq__'       : op.eq,
        '__ne__'       : op.ne,
        '__gt__'       : op.gt,
        '__ge__'       : op.ge,
        '__add__'      : op.add,
        '__sub__'      : op.sub,
        '__mul__'      : op.mul,
        '__matmul__'   : op.matmul,
        '__truediv__'  : op.truediv,
        '__floordiv__' : op.floordiv,
        '__mod__'      : op.mod,
        '__divmod__'   : None,
        '__pow__'      : op.pow,
        '__lshift__'   : op.lshift,
        '__rshift__'   : None,
        '__and__'      : op.and_,
        '__xor__'      : op.xor,
        '__or__'       : op.or_,
        '__radd__'     : None,
        '__rsub__'     : None,
        '__rmul__'     : None,
        '__rmatmul__'  : None,
        '__rtruediv__' : None,
        '__rfloordiv__': None,
        '__rmod__'     : None,
        '__rdivmod__'  : None,
        '__rpow__'     : None,
        '__rlshift__'  : None,
        '__rrshift__'  : None,
        '__rand__'     : None,
        '__rxor__'     : None,
        '__ror__'      : None,
        '__iadd__'     : op.iadd,
        '__isub__'     : op.isub,
        '__imul__'     : op.imul,
        '__imatmul__'  : op.imatmul,
        '__itruediv__' : op.itruediv,
        '__ifloordiv__': op.ifloordiv,
        '__imod__'     : op.imod,
        '__idivmod__'  : None,
        '__ipow__'     : op.ipow,
        '__ilshift__'  : op.ilshift,
        '__irshift__'  : op.irshift,
        '__iand__'     : op.iand,
        '__ixor__'     : op.ixor,
        '__ior__'      : op.ior,
        '__neg__'      : op.neg,
        '__pos__'      : op.pos,
        '__abs__'      : abs,
        '__invert__'   : op.invert,
        '__round__'    : round
    }

    def __new__(meta, name, bases, attrs):
        cls = super().__new__(meta, name, bases, attrs)

        for name, fn in meta.dunder_whitelist.items():
            setattr(cls, name, meta.SeriesMethod(name))

        return cls


class Col(metaclass=ColType):
    """ A proxy for a Series in a DataFrame

    Example:
        import pandas as pd
        from pandas_anaphora import Col

        data = pd.DataFrame(
        add_one_to_y = Col('y') + 1
        pd.testing.assert_series_equal(
            data['y'] + 1,
            add_one_to_y(data)
        )
    """
    class _MethodArgCollector:
        # Collects method args and pushes them onto the fn stack
        # __getattr__ has to return something 1) callable and 2) connected to Col

        # Not sure we even need this; maybe in the future should whitelist all methods/attrs --
        # note the duplicated logic with the SeriesMethod descriptor
        def __init__(self, col, name):
            self.col = col
            self.name = name

        def __call__(self, *args, **kwargs):
            self.col.push(op.methodcaller(self.name, *args, **kwargs))
            return self.col

    def __getattr__(self, name):
        method_or_attr = getattr(pd.Series, name)

        # do this after getattr() so it can raise AttributeError if needed
        if name.startswith('_'):
            raise AttributeError('Attribute/method access not implemented for pd.Series.{}'.format(name))

        if callable(method_or_attr):
            return Col._MethodArgCollector(self, name)
        else:
            self.push(op.attrgetter(name))
            return self

    def __init__(self, spec=None):
        # spec can be any valid column indexer accepted by .loc (TODO: document this...)
        # .iloc isn't supported; in the future there might be iloc=True
        self.spec = spec
        self.fns = []

    # TODO: reprlib?
    def __repr__(self):
        return "Col({})".format(repr(self.spec))

    def push(self, fn):
        self.fns.append(fn)

    def compute(self, df):
        col = df.loc[:, self.spec]
        if not self.fns:
            return col
        return tz.compose(*reversed(self.fns))(col)

    def __call__(self, df):
        return self.compute(df)

    
## Col-aware methods

def _apply_col(df, colname, val):
    if callable(val):
        if isinstance(val, Col) and val.spec is None:
            val = Col(colname)
        return val(df)
    else:
        return val


def with_column(df, colname, fn, loc=None, iloc=None, copy=True):
    """ Assign a column to a DataFrame """
    if _get_option('copy') or copy:
        df = df.copy()

    ## Figure out subset type

    # TODO: is at/iat viable? need a special case, because at/iat require both row and column index
    subset_types = {
        'loc': loc,
        'iloc': iloc
    }

    subset_types_given = [key for key, value in subset_types.items() if value is not None]
    n_subset_types_given = len(subset_types_given)

    if n_subset_types_given > 1:
        raise _error_mutally_exclusive_kwargs(subset_types_given)
    elif n_subset_types_given == 0:
        subset_types_given.append('loc')
        subset_types['loc'] = slice(None)  # foo[slice(None)] is equivalent to foo[:]
        
    subset_type = subset_types_given[0]
    subset_value = subset_types[subset_type]

    ## Resolve subset

    subset_value = _apply_col(df, colname, subset_value)
    df_subset = getattr(df, subset_type)[subset_value]

    ## Resolve fn

    # old version of pandas used to complain when you set a value on a slice/subset,
    # not sure when they removed that behavior
    value = _apply_col(df, colname, fn)
    df_subset[colname] = value
    # FIXME !! this won't work if colname isn't already a column in df !!

    return df


def _mutate_impl(df, mutations, sequential=False):
    if _get_option('copy'):
        df = df.copy()
        if sequential:
            newdf = df
        else:
            newdf = df.copy()  # don't allow access to other mutations
    else:
        newdf = df

    for name, val in mutations.items():
        newdf[name] = _apply_col(df, name, val)

    return newdf


def mutate(df, **mutations):
    """ Like dplyr::mutate in R

    DOES NOT allow access to LHS from the RHS

    Example:
        df = pd.DataFrame({'y': [1,2,3]})
        mutate(df, x=Col('y')*10)
    """
    return _mutate_impl(df, mutations, sequential=False)


def mutate_sequential(df, **mutations):
    """ Like dplyr::mutate in R with evil powers

    DOES allow access to LHS from the RHS

    Example:
        df = pd.DataFrame({'y': [1,2,3]})
        mutate(df, x=Col('y')*10, y=Col('x')+1)
    """
    return _mutate_impl(df, mutations, sequential=True)


## Mostly unnecessary (?) option-setting machinery

def _get_option(option):
    try:
        return _ANAPHORA_OPTIONS[option]
    except KeyError as exc:
        raise _error_unknown_option(key) from exc


def _set_options(**options):
    if _ANAPHORA_OPTIONS_LOCKED:
        raise RuntimeError('A lock has been acquired on _ANAPHORA_OPTIONS and must be released')

    for key, value in options.items():
        if key not in _ANAPHORA_OPTIONS:
            raise _error_unknown_option(key)
        _ANAPHORA_OPTIONS[key] = value


class anaphora_options:
    """ Context manager for temporarily setting global options
    
    NOT THREAD-SAFE. OR ANYTHING-SAFE. MAYBE KINDA A LITTLE BIT?
    """
    def __init__(self, **options):
        self.prev_options = {}
        self.options = options

    def __enter__(self):
        self.prev_options = py_shallowcopy(_ANAPHORA_OPTIONS)
        _ANAPHORA_OPTIONS_LOCKED = True
        _set_options(**self.options)

    def __exit__(self, exc_type, exc_value, traceback):
        self.options = self.prev_options
        self.prev_options = {}
        _set_options(**self.options)
        _ANAPHORA_OPTIONS_LOCKED = False

    
def anaphora_get_options(opt, *opts):
    """ Get global options

    NOT THREAD-SAFE. OR ANYTHING-SAFE.
    """
    if opt is None:
        if opts:
            raise TypeError()  # TODO: what the hell should this error message even be
        return py_shallowcopy(_ANAPHORA_OPTIONS)

    if opts:
        return {option: _get_option(option) for option in (opt, *opts)}
    else:
        return _get_option(opt)


def anaphora_set_options(**setoptions):
    """ Set global options

    NOT THREAD-SAFE. OR ANYTHING-SAFE.
    """
    _set_options(**setoptions)
