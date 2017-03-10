import ast
import array
import datetime
import types
import math
import re

from dateutil import parser as date_parser

from xframes.deps import HAS_NUMPY
if HAS_NUMPY:
    import numpy

from xframes.deps import pandas, HAS_PANDAS

from pyspark.sql.types import StringType, BooleanType, \
    DoubleType, FloatType, \
    ShortType, IntegerType, LongType, DecimalType, \
    ArrayType, MapType, TimestampType, NullType
import pyspark

# Type utilities
def possible_date(dt_str):
    """
    Detect if the given string is a possible date.

    Accepts everything that dateutil.parser considers a date except:
      must set the year
      cannot be a number (integer or float)

    Parameters
    ----------
    dt_str : str
        The string to be tested for a date.

    Returns
    -------
    out : boolean
        True if the input string is possibly a date.
    """
    if len(dt_str) == 0 or dt_str.isdigit() or dt_str.replace('.', '', 1).isdigit():
        return False
    try:
        dt = date_parser.parse(dt_str, default=datetime.datetime(1, 1, 1, 0, 0, 0))
        if dt.year == 1:
            return False
        return dt
    except ValueError:
        return False


def classify_type(s):
    if s.startswith('-'):
        rest = s[1:]
        if rest.isdigit():
            return int
        if rest.replace('.', '', 1).isdigit():
            return float
    if s.isdigit():
        return int
    if s.replace('.', '', 1).isdigit():
        return float
    if s.startswith('['):
        return list
    if s.startswith('{'):
        return dict
    if possible_date(s):
        return datetime.datetime
    return str


def infer_type(rdd):
    """
    From an RDD of strings, find what data type they represent.

    If all classify as a single type, then select that one.
    If they are all either int or float, then pick float.
    If they differ in other ways, then we will call it a string.

    Parameters
    ----------
    rdd : XRdd
        An XRdd of single values.

    Returns
    -------
    out : type
        The type of the values in the rdd.
    """
    head = rdd.take(100)
    types = [classify_type(s) for s in head]
    unique_types = set(types)
    if len(unique_types) == 1:
        dtype = types[0]
    elif unique_types == {int, float}:
        dtype = float
    else:
        dtype = str
    return dtype


def infer_types(rdd):
    """
    From an RDD of tuples of strings, find what data type each one represents.

    Parameters
    ----------
    rdd : XRdd
        An XRdd of tuples.

    Returns
    -------
    out : list(type)
        A list of the types of the values in the rdd.
    """
    head = rdd.take(100)
    n_cols = len(head[0])

    def get_col(head, i):
        return [row[i] for row in head]
    try:
        return [infer_type_of_list(get_col(head, i)) for i in range(n_cols)]
    except IndexError:
        raise ValueError('rows are not the same length')


def is_numeric_type(typ):
    if HAS_NUMPY:
        numeric_types = (float, int, long, numpy.float64, numpy.int64)
    else:
        numeric_types = (float, int, long)
    if typ is None:
        return False
    return issubclass(typ, numeric_types)


def is_numeric_val(val):
    return is_numeric_type(type(val))


def is_date_type(typ):
    if typ is None:
        return False
    date_types = (datetime.datetime, )
    return issubclass(typ, date_types)


def is_sortable_type(typ):
    if typ is None:
        return False
    if HAS_NUMPY:
        sortable_types = (str, float, int, long, numpy.float64, numpy.int64, datetime.datetime)
    else:
        sortable_types = (str, float, int, long, datetime.datetime)
    return issubclass(typ, sortable_types)


def infer_type_of_list(data):
    """
    Look through an iterable and get its data type.
    Use the first type, and check to make sure the rest are of that type.
    Missing values are skipped.
    Since these come from a program, do not attempt to parse strings info numbers, datetimes, etc.

    Parameters
    ----------
    data : list
        A list of values

    Returns
    -------
    out : type
        The type of values in the list.
    """
    def most_general(type1, type2):
        types = {type1, type2}
        if float in types:
            return float
        # Handle long type like an int
        return int

    candidate = None
    for d in data:
        if d is None:
            continue
        d_type = type(d)
        if candidate is None:
            candidate = d_type
        if d_type != candidate:
            if is_numeric_type(d_type) and is_numeric_type(candidate):
                candidate = most_general(d_type, candidate)
                continue
            raise TypeError('Infer_type_of_list: mixed types in list: {} {}'.format(d_type, candidate))
    return candidate


def infer_type_of_rdd(rdd):
    return infer_type_of_list(rdd.take(100))


def classify_auto(data):
    if isinstance(data, list):
        # if it is a list, Get the first type and make sure
        # the remaining items are all of the same type
        return infer_type_of_list(data)
    elif isinstance(data, array.array):
        return infer_type_of_list(data)
    elif HAS_PANDAS and isinstance(data, pandas.Series):
        # if it is a pandas series get the dtype of the series
        dtype = pytype_from_dtype(data.dtype)
        if dtype == object:
            # we need to get a bit more fine grained than that
            dtype = infer_type_of_list(data)
        return dtype

    elif HAS_NUMPY and isinstance(data, numpy.ndarray):
        # if it is a numpy array, get the dtype of the array
        dtype = pytype_from_dtype(data.dtype)
        if dtype == object:
            # we need to get a bit more fine grained than that
            dtype = infer_type_of_list(data)
        if len(data.shape) == 2:
            # we need to make it an array or a list
            if dtype == float or dtype == int:
                dtype = array.array
            else:
                dtype = list
            return dtype
        elif len(data.shape) > 2:
            raise TypeError('Cannot convert Numpy arrays of greater than 2 dimensions.')

    elif isinstance(data, str):
        # if it is a file, we default to string
        return str
    else:
        return None


def is_missing(x):
    """
    Tests for missing values.

    Parameters
    ----------
    x : object
        The value to test.

    Returns
    -------
    out : boolean
        True if the value is missing.
    """
    if x is None:
        return True
    if isinstance(x, (str, dict, list)) and len(x) == 0:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return False


def is_missing_or_empty(val):
    """
    Tests for missing or empty values.

    Parameters
    ----------
    val : object
        The value to test.

    Returns
    -------
    out : boolean
        True if the value is missing or empty.
    """
    if is_missing(val):
        return True
    if isinstance(val, (list, dict)):
        if len(val) == 0:
            return True
    return False


def is_xframe_type(typ):
    if HAS_PANDAS and issubclass(typ, pandas.DataFrame):
        return True
    if issubclass(typ, dict):
        return True
    if issubclass(typ, array.array):
        return True
    if issubclass(typ, list):
        return True
    if issubclass(typ, pyspark.sql.DataFrame):
        return True
    if issubclass(typ, basestring):
        return True
    if is_numeric_type(typ):
        return True
    if issubclass(typ, datetime.datetime):
        return True
    raise ValueError('Cannot infer input type for data {}.'.format(data))


def pytype_from_dtype(dtype):
    # Converts from string name of a pandas type to a type.
    if dtype == 'float':
        return float
    if dtype == 'float32':
        return float
    if dtype == 'float64':
        return float
    if dtype == 'int':
        return int
    if dtype == 'int32':
        return int
    if dtype == 'int64':
        return int
    if dtype == 'bool':
        return bool
    if dtype == 'datetime64[ns]':
        return datetime.datetime
    if dtype == 'object':
        return object
    if dtype == 'str':
        return str
    if dtype == 'string':
        return str
    return None


def to_ptype(schema_type):
    # converts a parquet schema type to python type
    if isinstance(schema_type, BooleanType):
        return bool
    if isinstance(schema_type, (IntegerType, ShortType, LongType, DecimalType)):
        return int
    if isinstance(schema_type, (DoubleType, FloatType)):
        return float
    if isinstance(schema_type, StringType):
        return str
    if isinstance(schema_type, ArrayType):
        return list
    if isinstance(schema_type, MapType):
        return dict
    if isinstance(schema_type, TimestampType):
        return datetime.datetime
    return str


def hint_to_schema_type(hint):
    # Given a type hint, return the corresponding schema type
    if hint == 'None':
        # this does not work -- gives an exception
        # use string type instead
        return StringType()
    if hint == 'int':
        return IntegerType()
    if hint == 'long':
        return LongType()
    if hint == 'decimal':
        # this does not work -- gives an excepton
        # use string type instead
        return StringType()
    if hint == 'bool':
        return BooleanType()
    if hint == 'float':
        return FloatType()
    if hint == 'datetime':
        return TimestampType()
    if hint == 'str':
        return StringType()
    m = re.match(r'list\[\s*(\S+)\s*\]', hint)
    if m is not None:
        inner = hint_to_schema_type(m.group(1))
        if not inner:
            raise ValueError('List element type is not recognized: {}.'.format(inner))
        return ArrayType(inner)
    m = re.match(r'dict\{\s*(\S+)\s*:\s*(\S+)\s*\}', hint)
    if m is not None:
        key = hint_to_schema_type(m.group(1))
        if key is None:
            raise ValueError('Map key type is not recognized: {}.'.format(key))
        val = hint_to_schema_type(m.group(2))
        if val is None:
            raise ValueError('Map value type is not recognized: {}.'.format(val))
        return MapType(key, val)
    return None


def to_schema_type(typ, elem):
    if typ is None:
        return hint_to_schema_type('None')
    if issubclass(typ, basestring):
        return hint_to_schema_type('str')
    if issubclass(typ, bool):
        return hint_to_schema_type('bool')
    if issubclass(typ, float):
        return hint_to_schema_type('float')
    if issubclass(typ, (int, long)):
        # Some integers cannot be stored in long, but we cannot tell this
        #  from the column type.  Let it fail in spark.
        return hint_to_schema_type('int')
    if issubclass(typ, datetime.datetime):
        return hint_to_schema_type('datetime')
    if issubclass(typ, list):
        if elem is None or len(elem) == 0:
            raise ValueError('Schema type cannot be determined.')
        elem_type = to_schema_type(type(elem[0]), None)
        if elem_type is None:
            raise TypeError('Element type cannot be determined')
        return ArrayType(elem_type)
    if issubclass(typ, dict):
        if elem is None or len(elem) == 0:
            raise ValueError('Schema type cannot be determined.')
        key_type = to_schema_type(type(elem.keys()[0]), None)
        if key_type is None:
            raise TypeError('Key type cannot be determined')
        val_type = to_schema_type(type(elem.values()[0]), None)
        if val_type is None:
            raise TypeError('Value type cannot be determined')
        return MapType(key_type, val_type)
    if issubclass(typ, types.NoneType):
        return None
    return hint_to_schema_type('str')


def safe_cast_val(val, typ):
    if val is None:
        return None
    if isinstance(val, basestring) and len(val) == 0:
        if issubclass(typ, int):
            return 0
        if issubclass(typ, float):
            return 0.0
        if issubclass(typ, basestring):
            return ''
        if issubclass(typ, dict):
            return {}
        if issubclass(typ, list):
            return []
        if issubclass(typ, datetime.datetime):
            return datetime.datetime(1, 1, 1)
    try:
        if issubclass(typ, dict):
            return ast.literal_eval(val)
    except ValueError:
        return {}
    try:
        if issubclass(typ, list):
            return ast.literal_eval(val)
    except ValueError:
        return []
    try:
        return typ(val)
    except UnicodeEncodeError:
        return ''
    except TypeError:
        return None
    except ValueError:
        return None
