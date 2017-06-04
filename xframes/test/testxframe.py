from __future__ import absolute_import
import six

import pytest
import os
import math
import copy
from datetime import datetime
import array
import pickle

from pyspark.sql.types import StructType, StructField, IntegerType, StringType

import pandas

# pytest testxframe.py
# pytest testxframe.py::TestXFrameVersion
# pytest testxframe.py::TestXFrameVersion::test_version

from xframes import XArray
from xframes import XFrame
from xframes.spark_context import CommonSparkContext
from xframes import object_utils
from xframes.aggregate import SUM, ARGMAX, ARGMIN, MAX, MIN, COUNT, MEAN, \
    VARIANCE, STDV, SELECT_ONE, CONCAT, VALUES, VALUES_COUNT


def almost_equal(a, b, places=None, delta=None):
    if a == b:
        return True
    if delta is not None:
        if abs(a - b) <= delta:
            return True
    if places is None:
        places = 7
    if round(abs(b - a), places) == 0:
        return True
    return False



def dict_keys_equal(expected, actual):
    expected_keys = sorted(expected.keys())
    actual_keys = sorted(actual.keys())
    return expected_keys == actual_keys


# noinspection PyClassHasNoInit
class TestXFrameVersion:
    """
    Tests XFrame version
    """

    def test_version(self):
        ver = object_utils.version()
        assert type(ver) is str


# noinspection PyClassHasNoInit
class TestXFrameConstructor:
    """
    Tests XFrame constructors that create data from local sources.
    """

    def test_construct_auto_dataframe(self):
        path = 'files/test-frame-auto.csv'
        res = XFrame(path)
        assert len(res) == 3
        assert res.column_names() == ['val_int', 'val_int_signed', 'val_float', 'val_float_signed',
                                      'val_str', 'val_list', 'val_dict']
        assert res.column_types() == [int, int, float, float, str, list, dict]
        assert res[0] == {'val_int': 1, 'val_int_signed': -1, 'val_float': 1.0, 'val_float_signed': -1.0,
                          'val_str': 'a', 'val_list': ['a'], 'val_dict': {1: 'a'}}
        assert res[1] == {'val_int': 2, 'val_int_signed': -2, 'val_float': 2.0, 'val_float_signed': -2.0,
                          'val_str': 'b', 'val_list': ['b'], 'val_dict': {2: 'b'}}
        assert res[2] == {'val_int': 3, 'val_int_signed': -3, 'val_float': 3.0, 'val_float_signed': -3.0,
                          'val_str': 'c', 'val_list': ['c'], 'val_dict': {3: 'c'}}

    def test_construct_auto_str_csv(self):
        path = 'files/test-frame.csv'
        res = XFrame(path)
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_construct_auto_str_tsv(self):
        path = 'files/test-frame.tsv'
        res = XFrame(path)
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_construct_auto_str_psv(self):
        path = 'files/test-frame.psv'
        res = XFrame(path)
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_construct_auto_str_txt(self):
        # construct and XFrame given a text file
        # interpret as csv
        path = 'files/test-frame.txt'
        res = XFrame(path)
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_construct_auto_str_noext(self):
        # construct and XFrame given a text file
        # interpret as csv
        path = 'files/test-frame'
        res = XFrame(path)
        res = res.sort('id')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_construct_auto_pandas_dataframe(self):
        df = pandas.DataFrame({'id': [1, 2, 3], 'val': [10.0, 20.0, 30.0]})
        res = XFrame(df)
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, float]
        assert res[0] == {'id': 1, 'val': 10.0}
        assert res[1] == {'id': 2, 'val': 20.0}
        assert res[2] == {'id': 3, 'val': 30.0}

    def test_construct_auto_str_xframe(self):
        # construct an XFrame given a file with unrecognized file extension
        path = 'files/test-frame'
        res = XFrame(path)
        res = res.sort('id')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_construct_xarray(self):
        # construct and XFrame given an XArray
        xa = XArray([1, 2, 3])
        t = XFrame(xa)
        assert len(t) == 3
        assert t.column_names() == ['X.0']
        assert t.column_types() == [int]
        assert t[0] == {'X.0': 1}
        assert t[1] == {'X.0': 2}
        assert t[2] == {'X.0': 3}

    def test_construct_xframe(self):
        # construct an XFrame given another XFrame
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = XFrame(t)
        assert len(res) == 3
        res = res.sort('id')
        assert list(res['id']) == [1, 2, 3]
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]

    def test_construct_iteritems(self):
        # construct an XFrame from an object that has iteritems
        class MyIterItem(object):
            @staticmethod
            def iteritems():
                return iter([('id', [1, 2, 3]), ('val', ['a', 'b', 'c'])])

        t = XFrame(MyIterItem())
        assert len(t) == 3
        assert t.column_names() == ['id', 'val']
        assert t.column_types() == [int, str]
        assert t[0] == {'id': 1, 'val': 'a'}
        assert t[1] == {'id': 2, 'val': 'b'}
        assert t[2] == {'id': 3, 'val': 'c'}

    def test_construct_iteritems_bad(self):
        # construct an XFrame from an object that has iteritems
        class MyIterItem(object):
            @staticmethod
            def iteritems():
                return iter([('id', 1), ('val', 'a')])

        with pytest.raises(TypeError) as exception_info:
            _ = XFrame(MyIterItem())
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Iterator values must be iterable.'

    def test_construct_iter(self):
        # construct an XFrame from an object that has __iter__
        class MyIter(object):
            def __iter__(self):
                return iter([1, 2, 3])

        t = XFrame(MyIter())
        assert len(t) == 3
        assert t.column_names() == ['X.0']
        assert t.column_types() == [int]
        assert t[0] == {'X.0': 1}
        assert t[1] == {'X.0': 2}
        assert t[2] == {'X.0': 3}

    def test_construct_iter_bad(self):
        # construct an XFrame from an object that has __iter__
        class MyIter(object):
            def __iter__(self):
                return iter([])

        with pytest.raises(TypeError) as exception_info:
            _ = XFrame(MyIter())
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Cannot determine types.'


    def test_construct_empty(self):
        # construct an empty XFrame
        t = XFrame()
        assert len(t) == 0

    def test_construct_str_csv(self):
        # construct and XFrame given a text file
        # interpret as csv
        path = 'files/test-frame.txt'
        res = XFrame(path, format='csv')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_construct_str_xframe(self):
        # construct and XFrame given a saved xframe
        path = 'files/test-frame'
        res = XFrame(path, format='xframe')
        res = res.sort('id')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_construct_array(self):
        # construct an XFrame from an array
        t = XFrame([1, 2, 3], format='array')
        assert len(t) == 3
        assert list(t['X.0']) == [1, 2, 3]

    def test_construct_array_mixed_xarray(self):
        # construct an XFrame from an xarray and values
        xa = XArray([1, 2, 3])
        with pytest.raises(ValueError) as exception_info:
            _ = XFrame([1, 2, xa], format='array')
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Cannot create XFrame from mix of regular values and XArrays.'

    def test_construct_array_mixed_types(self):
        # construct an XFrame from
        # an array of mixed types
        with pytest.raises(TypeError) as exception_info:
            _ = XFrame([1, 2, 'a'], format='array')
        exception_message = exception_info.value.args[0]
        assert exception_message == "Infer_type_of_list: mixed types in list: <type 'str'> <type 'int'>"

    def test_construct_unknown_format(self):
        # test unknown format
        with pytest.raises(ValueError) as exception_info:
            _ = XFrame([1, 2, 'a'], format='bad-format')
        exception_message = exception_info.value.args[0]
        assert exception_message == "Unknown input type: 'bad-format'."

    def test_construct_array_empty(self):
        # construct an XFrame from an empty array
        t = XFrame([], format='array')
        assert len(t) == 0

    def test_construct_array_xarray(self):
        # construct an XFrame from an xarray
        xa1 = XArray([1, 2, 3])
        xa2 = XArray(['a', 'b', 'c'])
        t = XFrame([xa1, xa2], format='array')
        assert len(t) == 3
        assert t.column_names() == ['X.0', 'X.1']
        assert t.column_types() == [int, str]
        assert t[0] == {'X.0': 1, 'X.1': 'a'}
        assert t[1] == {'X.0': 2, 'X.1': 'b'}
        assert t[2] == {'X.0': 3, 'X.1': 'c'}

    def test_construct_dict_int(self):
        # construct an XFrame from a dict of int
        t = XFrame({'id': [1, 2, 3], 'val': [10, 20, 30]}, format='dict')
        res = XFrame(t)
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': 1, 'val': 10}
        assert res[1] == {'id': 2, 'val': 20}
        assert res[2] == {'id': 3, 'val': 30}

    def test_construct_dict_float(self):
        # construct an XFrame from a dict of float
        t = XFrame({'id': [1.0, 2.0, 3.0], 'val': [10.0, 20.0, 30.0]}, format='dict')
        res = XFrame(t)
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [float, float]
        assert res[0] == {'id': 1.0, 'val': 10.0}
        assert res[1] == {'id': 2.0, 'val': 20.0}
        assert res[2] == {'id': 3.0, 'val': 30.0}

    def test_construct_dict_str(self):
        # construct an XFrame from a dict of str
        t = XFrame({'id': ['a', 'b', 'c'], 'val': ['A', 'B', 'C']}, format='dict')
        res = XFrame(t)
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [str, str]
        assert res[0] == {'id': 'a', 'val': 'A'}
        assert res[1] == {'id': 'b', 'val': 'B'}
        assert res[2] == {'id': 'c', 'val': 'C'}

    def test_construct_dict_int_str(self):
        # construct an XFrame from a dict of int and str
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']}, format='dict')
        assert len(t) == 3
        t = t.sort('id')
        assert list(t['id']) == [1, 2, 3]
        assert t.column_names() == ['id', 'val']
        assert t.column_types() == [int, str]

    def test_construct_dict_int_str_bad_len(self):
        # construct an XFrame from a dict of int and str with different lengths
        with pytest.raises(ValueError) as exception_info:
            XFrame({'id': [1, 2, 3], 'val': ['a', 'b']})
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Cannot create XFrame from dict of lists of different lengths.'

    def test_construct_binary(self):
        # make binary file
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        path = 'tmp/frame'
        t.save(path, format='binary')  # File does not necessarily save in order
        res = XFrame(path).sort('id')  # so let's sort after we read it back
        assert len(res) == 3
        assert t.column_names() == ['id', 'val']
        assert t.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}

    def test_construct_rdd(self):
        sc = CommonSparkContext.spark_context()
        rdd = sc.parallelize([(1, 'a'), (2, 'b'), (3, 'c')])
        res = XFrame(rdd)
        assert len(res) == 3
        assert res[0] == {'X.0': 1, 'X.1': 'a'}
        assert res[1] == {'X.0': 2, 'X.1': 'b'}

    def test_construct_spark_dataframe(self):
        sc = CommonSparkContext.spark_context()
        rdd = sc.parallelize([(1, 'a'), (2, 'b'), (3, 'c')])
        fields = [StructField('id', IntegerType(), True), StructField('val', StringType(), True)]
        schema = StructType(fields)
        sqlc = CommonSparkContext.spark_sql_context()
        s_rdd = sqlc.createDataFrame(rdd, schema)
        res = XFrame(s_rdd)
        assert len(res) == 3
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}

    def test_construct_binary_not_exist(self):
        path = 'does-not-exist'
        with pytest.raises(ValueError) as exception_info:
            _ = XFrame(path)
        exception_message = exception_info.value.args[0]
        assert exception_message.startswith('Input file does not exist:')


# noinspection PyClassHasNoInit
class TestXFrameReadCsvWithErrors:
    """
    Tests XFrame read_csv_with_errors
    """

    def test_read_csv_no_errors(self):
        path = 'files/test-frame.csv'
        res, errs = XFrame.read_csv_with_errors(path)
        assert len(res) == 3
        assert errs == {}

    def test_read_csv_width_error(self):
        path = 'files/test-frame-width-err.csv'
        res, errs = XFrame.read_csv_with_errors(path)
        assert 'width' in errs
        width_errs = errs['width']
        assert isinstance(width_errs, XArray)
        assert len(width_errs) == 2
        assert width_errs[0] == '1'
        assert width_errs[1] == '2,x,y'
        assert len(res) == 2

    def test_read_csv_null_error(self):
        path = 'files/test-frame-null.csv'
        res, errs = XFrame.read_csv_with_errors(path)
        assert 'csv' in errs
        csv_errs = errs['csv']
        assert isinstance(csv_errs, XArray)
        assert len(csv_errs) == 1
        assert csv_errs[0] == '2,\x00b'
        assert len(res) == 1

    def test_read_csv_null_header_error(self):
        path = 'files/test-frame-null-header.csv'
        res, errs = XFrame.read_csv_with_errors(path)
        assert 'header' in errs
        csv_errs = errs['header']
        assert isinstance(csv_errs, XArray)
        assert len(csv_errs) == 1
        assert csv_errs[0] == 'id,\x00val'
        assert len(res) == 0

    def test_read_csv_file_not_exist(self):
        path = 'files/does-not-exist.csv'
        with pytest.raises(ValueError) as exception_info:
            XFrame.read_csv_with_errors(path)
        exception_message = exception_info.value.args[0]
        assert exception_message.startswith('Input file does not exist:')

    # Cannot figure out how to cause SystemError in csv reader.
    # But it happened one time


# noinspection PyClassHasNoInit
class TestXFrameReadCsv:
    """
    Tests XFrame read_csv
    """

    def test_read_csv(self):
        path = 'files/test-frame.csv'
        res = XFrame.read_csv(path)
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_read_csv_verbose(self):
        path = 'files/test-frame.csv'
        res = XFrame.read_csv(path)
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_read_csv_delim(self):
        path = 'files/test-frame.psv'
        res = XFrame.read_csv(path, delimiter='|')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_read_csv_no_header(self):
        path = 'files/test-frame-no-header.csv'
        res = XFrame.read_csv(path, header=False)
        assert len(res) == 3
        assert res.column_names() == ['X.0', 'X.1']
        assert res.column_types() == [int, str]
        assert res[0] == {'X.0': 1, 'X.1': 'a'}
        assert res[1] == {'X.0': 2, 'X.1': 'b'}
        assert res[2] == {'X.0': 3, 'X.1': 'c'}

    def test_read_csv_comment(self):
        path = 'files/test-frame-comment.csv'
        res = XFrame.read_csv(path, comment_char='#')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_read_csv_escape(self):
        path = 'files/test-frame-escape.csv'
        res = XFrame.read_csv(path)
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a,a'}
        assert res[1] == {'id': 2, 'val': 'b,b'}
        assert res[2] == {'id': 3, 'val': 'c,c'}

    def test_read_csv_escape_custom(self):
        path = 'files/test-frame-escape-custom.csv'
        res = XFrame.read_csv(path, escape_char='$')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a,a'}
        assert res[1] == {'id': 2, 'val': 'b,b'}
        assert res[2] == {'id': 3, 'val': 'c,c'}

    def test_read_csv_initial_space(self):
        path = 'files/test-frame-initial_space.csv'
        res = XFrame.read_csv(path, skip_initial_space=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_read_csv_hints_type(self):
        path = 'files/test-frame.csv'
        res = XFrame.read_csv(path, column_type_hints=str)
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [str, str]
        assert res[0] == {'id': '1', 'val': 'a'}
        assert res[1] == {'id': '2', 'val': 'b'}
        assert res[2] == {'id': '3', 'val': 'c'}

    def test_read_csv_hints_list(self):
        path = 'files/test-frame-extra.csv'
        res = XFrame.read_csv(path, column_type_hints=[str, str, int])
        assert len(res) == 3
        assert res.column_names() == ['id', 'val1', 'val2']
        assert res.column_types() == [str, str, int]
        assert res[0] == {'id': '1', 'val1': 'a', 'val2': 10}
        assert res[1] == {'id': '2', 'val1': 'b', 'val2': 20}
        assert res[2] == {'id': '3', 'val1': 'c', 'val2': 30}

    # noinspection PyTypeChecker
    def test_read_csv_hints_dict(self):
        path = 'files/test-frame-extra.csv'
        res = XFrame.read_csv(path, column_type_hints={'val2': int})
        assert len(res) == 3
        assert res.column_names() == ['id', 'val1', 'val2']
        assert res.column_types() == [str, str, int]
        assert res[0] == {'id': '1', 'val1': 'a', 'val2': 10}
        assert res[1] == {'id': '2', 'val1': 'b', 'val2': 20}
        assert res[2] == {'id': '3', 'val1': 'c', 'val2': 30}

    def test_read_csv_na(self):
        path = 'files/test-frame-na.csv'
        res = XFrame.read_csv(path, na_values='None')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'NA'}
        assert res[1] == {'id': None, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_read_csv_na_mult(self):
        path = 'files/test-frame-na.csv'
        res = XFrame.read_csv(path, na_values=['NA', 'None'])
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': None}
        assert res[1] == {'id': None, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_read_csv_file_not_exist(self):
        path = 'files/does-not-exist.csv'
        with pytest.raises(ValueError) as exception_info:
            _ = XFrame.read_csv(path)
        exception_message = exception_info.value.args[0]
        assert exception_message.startswith('Input file does not exist:')


# noinspection PyClassHasNoInit
class TestXFrameReadText:
    """
    Tests XFrame read_text
    """

    def test_read_text(self):
        path = 'files/test-frame-text.txt'
        res = XFrame.read_text(path)
        assert len(res) == 3
        assert res.column_names() == ['text']
        assert res.column_types() == [str]
        assert res[0] == {'text': 'This is a test'}
        assert res[1] == {'text': 'of read_text.'}
        assert res[2] == {'text': 'Here is another sentence.'}

    def test_read_text_delimited(self):
        path = 'files/test-frame-text.txt'
        res = XFrame.read_text(path, delimiter='.')
        assert len(res) == 3
        assert res.column_names() == ['text']
        assert res.column_types() == [str]
        assert res[0] == {'text': 'This is a test of read_text'}
        assert res[1] == {'text': 'Here is another sentence'}
        assert res[2] == {'text': ''}

    def test_read_text_file_not_exist(self):
        path = 'files/does-not-exist.txt'
        with pytest.raises(ValueError) as exception_info:
            XFrame.read_text(path)
        exception_message = exception_info.value.args[0]
        assert exception_message.startswith('Input file does not exist:')


# noinspection PyClassHasNoInit
class TestXFrameReadParquet:
    """
    Tests XFrame read_parquet
    """

    def test_read_parquet_str(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        path = 'tmp/frame-parquet'
        t.save(path, format='parquet')

        res = XFrame('tmp/frame-parquet.parquet')
        # results may not come back in the same order
        res = res.sort('id')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_read_parquet_bool(self):
        t = XFrame({'id': [1, 2, 3], 'val': [True, False, True]})
        path = 'tmp/frame-parquet'
        t.save(path, format='parquet')

        res = XFrame('tmp/frame-parquet.parquet')
        res = res.sort('id')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, bool]
        assert res[0] == {'id': 1, 'val': True}
        assert res[1] == {'id': 2, 'val': False}
        assert res[2] == {'id': 3, 'val': True}

    def test_read_parquet_int(self):
        t = XFrame({'id': [1, 2, 3], 'val': [10, 20, 30]})
        path = 'tmp/frame-parquet'
        t.save(path, format='parquet')

        res = XFrame('tmp/frame-parquet.parquet')
        res = res.sort('id')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': 1, 'val': 10}
        assert res[1] == {'id': 2, 'val': 20}
        assert res[2] == {'id': 3, 'val': 30}

    def test_read_parquet_float(self):
        t = XFrame({'id': [1, 2, 3], 'val': [1.0, 2.0, 3.0]})
        path = 'tmp/frame-parquet'
        t.save(path, format='parquet')

        res = XFrame('tmp/frame-parquet.parquet')
        res = res.sort('id')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, float]
        assert res[0] == {'id': 1, 'val': 1.0}
        assert res[1] == {'id': 2, 'val': 2.0}
        assert res[2] == {'id': 3, 'val': 3.0}

    def test_read_parquet_list(self):
        t = XFrame({'id': [1, 2, 3], 'val': [[1, 1], [2, 2], [3, 3]]})
        path = 'tmp/frame-parquet'
        t.save(path, format='parquet')

        res = XFrame('tmp/frame-parquet.parquet')
        res = res.sort('id')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, list]
        assert res[0] == {'id': 1, 'val': [1, 1]}
        assert res[1] == {'id': 2, 'val': [2, 2]}
        assert res[2] == {'id': 3, 'val': [3, 3]}

    def test_read_parquet_dict(self):
        t = XFrame({'id': [1, 2, 3], 'val': [{1: 1}, {2: 2}, {3: 3}]})
        path = 'tmp/frame-parquet'
        t.save(path, format='parquet')

        res = XFrame('tmp/frame-parquet.parquet')
        res = res.sort('id')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, dict]
        assert res[0] == {'id': 1, 'val': {1: 1}}
        assert res[1] == {'id': 2, 'val': {2: 2}}
        assert res[2] == {'id': 3, 'val': {3: 3}}

    def test_read_parquet_not_exist(self):
        path = 'files/does-not-exist.parquet'
        with pytest.raises(ValueError) as exception_info:
            _ = XFrame(path)
        exception_message = exception_info.value.args[0]
        assert exception_message.startswith('Input file does not exist:')


# noinspection PyClassHasNoInit
class TestXFrameFromXArray:
    """
    Tests XFrame from_xarray
    """

    # noinspection PyUnresolvedReferences
    def test_from_xarray(self):
        a = XArray([1, 2, 3])
        res = XFrame.from_xarray(a, 'id')
        assert len(res) == 3
        assert res.column_names() == ['id']
        assert res.column_types() == [int]
        assert res[0] == {'id': 1}
        assert res[1] == {'id': 2}
        assert res[2] == {'id': 3}


# noinspection SqlDialectInspection,SqlNoDataSourceInspection
# noinspection PyClassHasNoInit
class TestXFrameToSparkDataFrame:
    """
    Tests XFrame to_spark_dataframe
    """

    def test_to_spark_dataframe_str(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t.to_spark_dataframe('tmp_tbl')
        sqlc = CommonSparkContext.spark_sql_context()
        results = sqlc.sql('SELECT * FROM tmp_tbl ORDER BY id')
        assert results.count() == 3
        row = results.collect()[0]
        assert row.id == 1
        assert row.val == 'a'

    def test_to_spark_dataframe_bool(self):
        t = XFrame({'id': [1, 2, 3], 'val': [True, False, True]})
        t.to_spark_dataframe('tmp_tbl')
        sqlc = CommonSparkContext.spark_sql_context()
        results = sqlc.sql('SELECT * FROM tmp_tbl ORDER BY id')
        assert results.count() == 3
        row = results.collect()[0]
        assert row.id == 1
        assert row.val is True

    def test_to_spark_dataframe_float(self):
        t = XFrame({'id': [1, 2, 3], 'val': [1.0, 2.0, 3.0]})
        t.to_spark_dataframe('tmp_tbl')
        sqlc = CommonSparkContext.spark_sql_context()
        results = sqlc.sql('SELECT * FROM tmp_tbl ORDER BY id')
        assert results.count() == 3
        row = results.collect()[0]
        assert row.id == 1
        assert row.val == 1.0

    def test_to_spark_dataframe_int(self):
        t = XFrame({'id': [1, 2, 3], 'val': [1, 2, 3]})
        t.to_spark_dataframe('tmp_tbl')
        sqlc = CommonSparkContext.spark_sql_context()
        results = sqlc.sql('SELECT * FROM tmp_tbl ORDER BY id')
        assert results.count() == 3
        row = results.collect()[0]
        assert row.id == 1
        assert row.val == 1

    def test_to_spark_dataframe_list(self):
        t = XFrame({'id': [1, 2, 3], 'val': [[1, 1], [2, 2], [3, 3]]})
        t.to_spark_dataframe('tmp_tbl')
        sqlc = CommonSparkContext.spark_sql_context()
        results = sqlc.sql('SELECT * FROM tmp_tbl ORDER BY id')
        assert results.count() == 3
        row = results.collect()[0]
        assert row.id == 1
        assert row.val == [1, 1]

    def test_to_spark_dataframe_list_hint(self):
        t = XFrame({'id': [1, 2, 3], 'val': [[None, 1], [2, 2], [3, 3]]})
        t.to_spark_dataframe('tmp_tbl', column_type_hints={'val': 'list[int]'})
        sqlc = CommonSparkContext.spark_sql_context()
        results = sqlc.sql('SELECT * FROM tmp_tbl ORDER BY id')
        assert results.count() == 3
        row = results.collect()[1]
        assert row.id == 2
        assert row.val == [2, 2]

    def test_to_spark_dataframe_list_bad(self):
        t = XFrame({'id': [1, 2, 3], 'val': [[None, 1], [2, 2], [3, 3]]})
        with pytest.raises(TypeError) as exception_info:
            t.to_spark_dataframe('tmp_tbl')
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Element type cannot be determined.'

    def test_to_spark_dataframe_map(self):
        t = XFrame({'id': [1, 2, 3], 'val': [{'x': 1}, {'y': 2}, {'z': 3}]})
        t.to_spark_dataframe('tmp_tbl')
        sqlc = CommonSparkContext.spark_sql_context()
        results = sqlc.sql('SELECT * FROM tmp_tbl ORDER BY id')
        assert results.count() == 3
        row = results.collect()[0]
        assert row.id == 1
        assert row.val == {'x': 1}

    def test_to_spark_dataframe_map_bad(self):
        t = XFrame({'id': [1, 2, 3], 'val': [None, {'y': 2}, {'z': 3}]})
        with pytest.raises(ValueError) as exception_info:
            t.to_spark_dataframe('tmp_tbl')
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Schema type cannot be determined.'

    @pytest.mark.skip(reason='files in spark 2')
    def test_to_spark_dataframe_map_hint(self):
        t = XFrame({'id': [1, 2, 3], 'val': [{None: None}, {'y': 2}, {'z': 3}]})
        t.to_spark_dataframe('tmp_tbl', column_type_hints={'val': 'dict{str: int}'})
        sqlc = CommonSparkContext.spark_sql_context()
        results = sqlc.sql('SELECT * FROM tmp_tbl ORDER BY id')
        assert results.count() == 3
        row = results.collect()[1]
        assert row.id == 1
        assert row.val == {'y': 2}

    def test_to_spark_dataframe_str_rewrite(self):
        t = XFrame({'id': [1, 2, 3], 'val;1': ['a', 'b', 'c']})
        t.to_spark_dataframe('tmp_tbl')
        sqlc = CommonSparkContext.spark_sql_context()
        results = sqlc.sql('SELECT * FROM tmp_tbl ORDER BY id')
        assert results.count() == 3
        row = results.collect()[0]
        assert row.id == 1
        assert row.val_1 == 'a'

    def test_to_spark_dataframe_str_rename(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t.to_spark_dataframe('tmp_tbl', column_names=['id1', 'val1'])
        sqlc = CommonSparkContext.spark_sql_context()
        results = sqlc.sql('SELECT * FROM tmp_tbl ORDER BY id1')
        assert results.count() == 3
        row = results.collect()[0]
        assert row.id1 == 1
        assert row.val1 == 'a'

    def test_to_spark_dataframe_str_rename_bad_type(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            t.to_spark_dataframe('tmp_tbl', column_names='id1')

    def test_to_spark_dataframe_str_rename_bad_len(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(ValueError) as exception_info:
            t.to_spark_dataframe('tmp_tbl', column_names=['id1'])
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Column names list must match number of columns: actual: 1, expected: 2'


# noinspection PyClassHasNoInit
class TestXFrameToRdd:
    """
    Tests XFrame to_rdd
    """

    def test_to_rdd(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        rdd_val = t.to_rdd().collect()
        assert rdd_val[0] == (1, 'a')
        assert rdd_val[1] == (2, 'b')
        assert rdd_val[2] == (3, 'c')


# noinspection PyClassHasNoInit
class TestXFrameFromRdd:
    """
    Tests XFrame from_rdd with regular rdd
    """

    def test_from_rdd(self):
        sc = CommonSparkContext.spark_context()
        rdd = sc.parallelize([(1, 'a'), (2, 'b'), (3, 'c')])
        res = XFrame.from_rdd(rdd)
        assert len(res) == 3
        assert res[0] == {'X.0': 1, 'X.1': 'a'}
        assert res[1] == {'X.0': 2, 'X.1': 'b'}

    def test_from_rdd_names(self):
        sc = CommonSparkContext.spark_context()
        rdd = sc.parallelize([(1, 'a'), (2, 'b'), (3, 'c')])
        res = XFrame.from_rdd(rdd, column_names=['id', 'val'])
        assert len(res) == 3
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}

    def test_from_rdd_types(self):
        sc = CommonSparkContext.spark_context()
        rdd = sc.parallelize([(None, 'a'), (2, 'b'), (3, 'c')])
        res = XFrame.from_rdd(rdd, column_types=[int, str])
        assert len(res) == 3
        assert res.column_types() == [int, str]
        assert res[0] == {'X.0': None, 'X.1': 'a'}
        assert res[1] == {'X.0': 2, 'X.1': 'b'}

    def test_from_rdd_names_types(self):
        sc = CommonSparkContext.spark_context()
        rdd = sc.parallelize([(None, 'a'), (2, 'b'), (3, 'c')])
        res = XFrame.from_rdd(rdd, column_names=['id', 'val'], column_types=[int, str])
        assert len(res) == 3
        assert res.column_types() == [int, str]
        assert res[0] == {'id': None, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}

    def test_from_rdd_names_bad(self):
        sc = CommonSparkContext.spark_context()
        rdd = sc.parallelize([(1, 'a'), (2, 'b'), (3, 'c')])
        with pytest.raises(ValueError) as exception_info:
            XFrame.from_rdd(rdd, column_names=('id',))
        exception_message = exception_info.value.args[0]
        assert exception_message == "Length of names does not match RDD: ('id',) (1, 'a')."

    def test_from_rdd_types_bad(self):
        sc = CommonSparkContext.spark_context()
        rdd = sc.parallelize([(None, 'a'), (2, 'b'), (3, 'c')])
        with pytest.raises(ValueError) as exception_info:
            XFrame.from_rdd(rdd, column_types=(int,))
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Length of types does not match RDD.'


# noinspection PyClassHasNoInit
class TestXFrameFromSparkDataFrame:
    """
    Tests XFrame from_rdd with spark dataframe
    """

    def test_from_rdd(self):
        sc = CommonSparkContext.spark_context()
        rdd = sc.parallelize([(1, 'a'), (2, 'b'), (3, 'c')])
        fields = [StructField('id', IntegerType(), True), StructField('val', StringType(), True)]
        schema = StructType(fields)
        sqlc = CommonSparkContext.spark_sql_context()
        s_rdd = sqlc.createDataFrame(rdd, schema)

        res = XFrame.from_rdd(s_rdd)
        assert len(res) == 3
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}


# noinspection PyClassHasNoInit
class TestXFramePrintRows:
    """
    Tests XFrame print_rows
    """

    def test_print_rows(self):
        pass


# noinspection PyClassHasNoInit
class TestXFrameToStr:
    """
    Tests XFrame __str__
    """

    def test_to_str(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        s = str(t).split('\n')
        assert s[0] == '+----+-----+'
        assert s[1] == '| id | val |'
        assert s[2] == '+----+-----+'
        assert s[3] == '| 1  |  a  |'
        assert s[4] == '| 2  |  b  |'
        assert s[5] == '| 3  |  c  |'
        assert s[6] == '+----+-----+'


# noinspection PyClassHasNoInit
class TestXFrameNonzero:
    """
    Tests XFrame __nonzero__
    """

    def test_nonzero_true(self):
        # not empty, nonzero data
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        assert t is not False

    def test_nonzero_false(self):
        # empty
        t = XFrame()
        assert t is not True

    def test_empty_false(self):
        # empty, but previously not empty
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t = t.filterby([99], 'id')
        assert t is not True


# noinspection PyClassHasNoInit
class TestXFrameLen:
    """
    Tests XFrame __len__
    """

    def test_len_nonzero(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        assert len(t) == 3

    def test_len_zero(self):
        t = XFrame()
        assert len(t) == 0

        # TODO make an XFrame and then somehow delete all its rows, so the RDD
        # exists but is empty


# noinspection PyClassHasNoInit
class TestXFrameCopy:
    """
    Tests XFrame __copy__
    """

    def test_copy(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        x = copy.copy(t)
        assert len(x) == 3
        assert list(x['id']) == [1, 2, 3]
        assert x.column_names() == ['id', 'val']
        assert x.column_types() == [int, str]


# noinspection PyClassHasNoInit
class TestXFrameDtype:
    """
    Tests XFrame dtype
    """

    def test_dtype(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        dt = t.dtype()
        assert dt[0] is int
        assert dt[1] is str


# noinspection PyClassHasNoInit
class TestXFrameTableLineage:
    """
    Tests XFrame table lineage
    """

    def test_lineage(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        lineage = t.lineage()['table']
        assert len(lineage) == 1
        item = list(lineage)[0]
        assert item == 'PROGRAM'

    def test_lineage_csv(self):
        path = 'files/test-frame-auto.csv'
        res = XFrame(path)
        lineage = res.lineage()['table']
        assert len(lineage) == 1
        item = list(lineage)[0]
        filename = os.path.basename(item)
        assert filename == 'test-frame-auto.csv'

    def test_lineage_transform(self):
        path = 'files/test-frame-auto.csv'
        res = XFrame(path).transform_col('val_int', lambda row: row['val_int'] * 2)
        lineage = res.lineage()['table']
        assert len(lineage) == 1
        filename = os.path.basename(list(lineage)[0])
        assert filename == 'test-frame-auto.csv'

    def test_lineage_rdd(self):
        sc = CommonSparkContext.spark_context()
        rdd = sc.parallelize([(1, 'a'), (2, 'b'), (3, 'c')])
        res = XFrame.from_rdd(rdd)
        lineage = res.lineage()['table']
        assert len(lineage) == 1
        item = list(lineage)[0]
        assert item == 'RDD'

    def test_lineage_hive(self):
        pass

    def test_lineage_pandas_dataframe(self):
        df = pandas.DataFrame({'id': [1, 2, 3], 'val': [10.0, 20.0, 30.0]})
        res = XFrame(df)
        lineage = res.lineage()['table']
        assert len(lineage) == 1
        item = list(lineage)[0]
        assert item == 'PANDAS'

    def test_lineage_spark_dataframe(self):
        pass

    def test_lineage_program_data(self):
        res = XFrame({'id': [1, 2, 3], 'val': [10.0, 20.0, 30.0]})
        lineage = res.lineage()['table']
        assert len(lineage) == 1
        item = list(lineage)[0]
        assert item == 'PROGRAM'

    def test_lineage_append(self):
        res1 = XFrame('files/test-frame.csv')
        res2 = XFrame('files/test-frame.psv')
        res = res1.append(res2)
        lineage = res.lineage()['table']
        assert len(lineage) == 2
        basenames = set([os.path.basename(item) for item in lineage])
        assert 'test-frame.csv' in basenames
        assert 'test-frame.psv' in basenames

    def test_lineage_join(self):
        res1 = XFrame('files/test-frame.csv')
        res2 = XFrame('files/test-frame.psv').transform_col('val', lambda row: row['val'] + 'xxx')
        res = res1.join(res2, on='id').sort('id').head()
        lineage = res.lineage()['table']
        assert len(lineage) == 2
        basenames = set([os.path.basename(item) for item in lineage])
        assert 'test-frame.csv' in basenames
        assert 'test-frame.psv' in basenames

    def test_lineage_add_column(self):
        res1 = XFrame('files/test-frame.csv')
        res2 = XArray('files/test-array-int')
        res = res1.add_column(res2, 'new-col')
        lineage = res.lineage()['table']
        assert len(lineage) == 2
        basenames = set([os.path.basename(item) for item in lineage])
        assert 'test-frame.csv' in basenames
        assert 'test-array-int' in basenames

    def test_lineage_save(self):
        res = XFrame('files/test-frame.csv')
        path = 'tmp/frame'
        res.save(path, format='binary')
        with open(os.path.join(path, '_metadata')) as f:
            metadata = pickle.load(f)
            assert metadata == [['id', 'val'], [int, str]]
        with open(os.path.join(path, '_lineage')) as f:
            lineage = pickle.load(f)
            table_lineage = lineage[0]
            assert len(table_lineage) == 1
            basenames = set([os.path.basename(item) for item in table_lineage])
            assert 'test-frame.csv' in basenames

    def test_lineage_load(self):
        res = XFrame('files/test-frame.csv')
        path = 'tmp/frame'
        res.save(path, format='binary')
        res = XFrame(path)
        lineage = res.lineage()['table']
        assert len(lineage) == 1
        basenames = set([os.path.basename(item) for item in lineage])
        assert 'test-frame.csv' in basenames


# noinspection PyClassHasNoInit
class TestXFrameColumnLineage:
    """
    Tests XFrame column lineage
    """

    def test_lineage(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        lineage = t.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'val']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['val'] == {('PROGRAM', 'val')}

    def test_construct_empty(self):
        t = XFrame()
        lineage = t.lineage()['column']
        assert len(lineage) == 0

    def test_construct_auto_pandas_dataframe(self):
        df = pandas.DataFrame({'id': [1, 2, 3], 'val': [10.0, 20.0, 30.0]})
        res = XFrame(df)
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'val']
        assert lineage['id'] == {('PANDAS', 'id')}
        assert lineage['val'] == {('PANDAS', 'val')}

    def test_lineage_load(self):
        inpath = 'files/test-frame.csv'
        real_inpath = os.path.realpath(inpath)
        res = XFrame(inpath)
        path = 'tmp/frame'
        res.save(path, format='binary')
        res = XFrame(path)
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'val']
        assert lineage['id'] == {(real_inpath, 'id')}
        assert lineage['val'] == {(real_inpath, 'val')}

    def test_construct_auto_dataframe(self):
        path = 'files/test-frame-auto.csv'
        real_path = os.path.realpath(path)
        res = XFrame(path)
        lineage = res.lineage()['column']
        assert len(lineage) == 7
        assert 'val_int' in lineage
        assert 'val_str' in lineage
        assert lineage['val_int'] == {(real_path, 'val_int')}
        assert lineage['val_str'] == {(real_path, 'val_str')}

    # hive
    # TODO test

    def test_from_rdd(self):
        sc = CommonSparkContext.spark_context()
        rdd = sc.parallelize([(1, 'a'), (2, 'b'), (3, 'c')])
        res = XFrame.from_rdd(rdd, column_names=['id', 'val'])
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'val']
        assert lineage['id'] == {('RDD', 'id')}
        assert lineage['val'] == {('RDD', 'val')}

    def test_construct_auto_str_csv(self):
        path = 'files/test-frame.csv'
        real_path = os.path.realpath(path)
        res = XFrame(path)
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'val']
        assert lineage['id'] == {(real_path, 'id')}
        assert lineage['val'] == {(real_path, 'val')}

    def test_read_text(self):
        path = 'files/test-frame-text.txt'
        real_path = os.path.realpath(path)
        res = XFrame.read_text(path)
        lineage = res.lineage()['column']
        assert len(lineage) == 1
        assert sorted(lineage.keys()) == ['text']
        assert lineage['text'] == {(real_path, 'text')}

    def test_read_parquet_str(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        path = 'tmp/frame-parquet'
        t.save(path, format='parquet')
        inpath = 'tmp/frame-parquet.parquet'
        real_path = os.path.realpath(inpath)
        res = XFrame(inpath)
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'val']
        assert lineage['id'] == {(real_path, 'id')}
        assert lineage['val'] == {(real_path, 'val')}

    def test_save(self):
        t = XFrame({'id': [30, 20, 10], 'val': ['a', 'b', 'c']})
        path = 'tmp/frame'
        t.save(path, format='binary')
        lineage = t.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'val']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['val'] == {('PROGRAM', 'val')}

    def test_sample(self):
        t = XFrame({'id': [1, 2, 3, 4, 5], 'val': ['a', 'b', 'c', 'd', 'e']})
        res = t.sample(0.2, 2)
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'val']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['val'] == {('PROGRAM', 'val')}

    def test_select_column(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.select_column('id')
        lineage = res.lineage()['column']
        assert len(lineage) == 1
        assert sorted(lineage.keys()) == ['_XARRAY']
        assert lineage['_XARRAY'] == {('PROGRAM', 'id')}

    def test_select_columns(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        res = t.select_columns(['id', 'val'])
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'val']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['val'] == {('PROGRAM', 'val')}

    def test_copy(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = copy.copy(t)
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'val']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['val'] == {('PROGRAM', 'val')}

    # noinspection PyUnresolvedReferences
    def test_from_xarray(self):
        a = XArray([1, 2, 3])
        res = XFrame.from_xarray(a, 'id')
        lineage = res.lineage()['column']
        assert len(lineage) == 1
        assert lineage['id'] == {('PROGRAM', '_XARRAY')}

    def test_add_column(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta = XArray([3.0, 2.0, 1.0])
        res = tf.add_column(ta, name='another')
        lineage = res.lineage()['column']
        assert len(lineage) == 3
        assert sorted(lineage.keys()) == ['another', 'id', 'val']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['val'] == {('PROGRAM', 'val')}
        assert lineage['another'] == {('PROGRAM', '_XARRAY')}

    # add_column_in_place
    # TODO test

    def test_add_columns_array(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta1 = XArray([3.0, 2.0, 1.0])
        ta2 = XArray([30.0, 20.0, 10.0])
        res = tf.add_columns([ta1, ta2], names=['new1', 'new2'])
        lineage = res.lineage()['column']
        assert len(lineage) == 4
        assert sorted(lineage.keys()) == ['id', 'new1', 'new2', 'val']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['val'] == {('PROGRAM', 'val')}
        assert lineage['new1'] == {('PROGRAM', '_XARRAY')}
        assert lineage['new2'] == {('PROGRAM', '_XARRAY')}

    # add_columns_array_in_place
    # TODO test

    def test_add_columns_frame(self):
        tf1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        tf2 = XFrame({'new1': [3.0, 2.0, 1.0], 'new2': [30.0, 20.0, 10.0]})
        res = tf1.add_columns(tf2)
        lineage = res.lineage()['column']
        assert len(lineage) == 4
        assert sorted(lineage.keys()) == ['id', 'new1', 'new2', 'val']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['val'] == {('PROGRAM', 'val')}
        assert lineage['new1'] == {('PROGRAM', 'new1')}
        assert lineage['new2'] == {('PROGRAM', 'new2')}

    # add_columns_frame_in_place
    # TODO test

    def test_remove_column(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        res = t.remove_column('another')
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'val']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['val'] == {('PROGRAM', 'val')}

    # remove_column_in_place
    # TODO test

    def test_remove_columns(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'new1': [3.0, 2.0, 1.0], 'new2': [30.0, 20.0, 10.0]})
        res = t.remove_columns(['new1', 'new2'])
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'val']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['val'] == {('PROGRAM', 'val')}

    def test_swap_columns(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'x': [3.0, 2.0, 1.0]})
        res = t.swap_columns('val', 'x')
        lineage = res.lineage()['column']
        assert len(lineage) == 3
        assert sorted(lineage.keys()) == ['id', 'val', 'x']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['val'] == {('PROGRAM', 'val')}
        assert lineage['x'] == {('PROGRAM', 'x')}

    def test_reorder_columns(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'x': [3.0, 2.0, 1.0]})
        res = t.reorder_columns(['val', 'x', 'id'])
        lineage = res.lineage()['column']
        assert len(lineage) == 3
        assert sorted(lineage.keys()) == ['id', 'val', 'x']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['val'] == {('PROGRAM', 'val')}
        assert lineage['x'] == {('PROGRAM', 'x')}

    # add_column_const_in_place
    # TODO test
    # replace_column_const_in_place
    # TODO test
    # replace_single_column_in_place
    # TODO test

    def test_replace_column(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        a = XArray(['x', 'y', 'z'])
        res = t.replace_column('val', a)
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'val']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['val'] == {('PROGRAM', '_XARRAY')}

    # replace_selected_column_in_place
    # TODO test

    def test_flat_map(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.flat_map(['number', 'letter'],
                         lambda row: [list(six.itervalues(row)) for _ in range(0, row['id'])],
                         column_types=[int, str])
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['letter', 'number']
        assert lineage['number'] == {('PROGRAM', 'id'), ('PROGRAM', 'val')}
        assert lineage['letter'] == {('PROGRAM', 'id'), ('PROGRAM', 'val')}

    def test_flat_map_use_columns(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [10, 20, 30]})
        res = t.flat_map(['number', 'letter'],
                         lambda row: [list(six.itervalues(row)) for _ in range(0, row['id'])],
                         column_types=[int, str], use_columns=['id', 'val'])
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['letter', 'number']
        assert lineage['number'] == {('PROGRAM', 'id'), ('PROGRAM', 'val')}
        assert lineage['letter'] == {('PROGRAM', 'id'), ('PROGRAM', 'val')}

    def test_filterby_xarray(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        a = XArray([1, 3])
        res = t.filterby(a, 'id').sort('id')
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'val']
        assert lineage['id'] == {('PROGRAM', 'id'), ('PROGRAM', '_XARRAY')}
        assert lineage['val'] == {('PROGRAM', 'val')}

    def test_stack_list(self):
        t = XFrame({'id': [1, 2, 3], 'val': [['a1', 'b1', 'c1'], ['a2', 'b2'], ['a3', 'b3', 'c3', None]]})
        res = t.stack('val', 'new-val')
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'new-val']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['new-val'] == {('PROGRAM', 'val')}

    def test_stack_dict(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [{'a': 3, 'b': 2}, {'a': 2, 'c': 2}, {'c': 1, 'd': 3}, {}]})
        res = t.stack('val', ['stack-key', 'stack-val'])
        lineage = res.lineage()['column']
        assert len(lineage) == 3
        assert sorted(lineage.keys()) == ['id', 'stack-key', 'stack-val']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['stack-key'] == {('PROGRAM', 'val')}
        assert lineage['stack-val'] == {('PROGRAM', 'val')}

    def test_append(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [10, 20, 30], 'val': ['aa', 'bb', 'cc']})
        res = t1.append(t2)
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'val']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['val'] == {('PROGRAM', 'val')}

    def test_range_slice(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.range(slice(0, 2))
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'val']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['val'] == {('PROGRAM', 'val')}

    def test_dropna(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.dropna()
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'val']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['val'] == {('PROGRAM', 'val')}

    def test_add_row_number(self):
        t = XFrame({'ident': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.add_row_number()
        lineage = res.lineage()['column']
        assert len(lineage) == 3
        assert sorted(lineage.keys()) == ['id', 'ident', 'val']
        assert lineage['id'] == {('INDEX', 'id')}
        assert lineage['val'] == {('PROGRAM', 'val')}
        assert lineage['ident'] == {('PROGRAM', 'ident')}

    def test_pack_columns(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='new')
        lineage = res.lineage()['column']
        assert len(lineage) == 1
        assert sorted(lineage.keys()) == ['new']
        assert lineage['new'] == {('PROGRAM', 'id'), ('PROGRAM', 'val')}

    def test_foreach(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t.foreach(lambda row, ini: row['id'] * 2)

    def test_apply(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.apply(lambda row: row['id'] * 2)
        lineage = res.lineage()['column']
        assert len(lineage) == 1
        assert sorted(lineage.keys()) == ['_XARRAY']
        assert lineage['_XARRAY'] == {('PROGRAM', 'id'), ('PROGRAM', 'val')}

    def test_apply_with_use_columns(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [10, 20, 30]})
        res = t.apply(lambda row: row['id'] * 2, use_columns=['id', 'val'])
        lineage = res.lineage()['column']
        assert len(lineage) == 1
        assert sorted(lineage.keys()) == ['_XARRAY']
        assert lineage['_XARRAY'] == {('PROGRAM', 'id'), ('PROGRAM', 'val')}

    def test_transform_col(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.transform_col('id', lambda row: row['id'] * 2)
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'val']
        assert lineage['id'] == {('PROGRAM', 'id'), ('PROGRAM', 'val')}
        assert lineage['val'] == {('PROGRAM', 'val')}

    def test_transform_col_with_use_cols(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [10, 20, 30]})
        res = t.transform_col('id', lambda row: row['id'] * 2, use_columns=['id', 'val'])
        lineage = res.lineage()['column']
        assert len(lineage) == 3
        assert sorted(lineage.keys()) == ['another', 'id', 'val']
        assert lineage['id'] == {('PROGRAM', 'id'), ('PROGRAM', 'val')}
        assert lineage['val'] == {('PROGRAM', 'val')}

    def test_transform_cols(self):
        t = XFrame({'other': ['x', 'y', 'z'], 'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.transform_cols(['id', 'val'], lambda row: [row['id'] * 2, row['val'] + 'x'])
        lineage = res.lineage()['column']
        assert len(lineage) == 3
        assert sorted(lineage.keys()) == ['id', 'other', 'val']
        assert lineage['id'] == {('PROGRAM', 'id'), ('PROGRAM', 'val'), ('PROGRAM', 'other')}
        assert lineage['val'] == {('PROGRAM', 'id'), ('PROGRAM', 'val'), ('PROGRAM', 'other')}

    def test_transform_cols_with_use_cols(self):
        t = XFrame({'other': ['x', 'y', 'z'], 'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.transform_cols(['id', 'val'], lambda row: [row['id'] * 2, row['val'] + 'x'], use_columns=['id', 'val'])
        lineage = res.lineage()['column']
        assert len(lineage) == 3
        assert sorted(lineage.keys()) == ['id', 'other', 'val']
        assert lineage['id'] == {('PROGRAM', 'id'), ('PROGRAM', 'val')}
        assert lineage['val'] == {('PROGRAM', 'id'), ('PROGRAM', 'val')}

    def test_filterby_int_id(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.filterby(1, 'id').sort('id')
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'val']
        assert sorted(lineage.keys()) == ['id', 'val']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['val'] == {('PROGRAM', 'val')}

    def test_groupby_count(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', {'count': COUNT})
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['count', 'id']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['count'] == {('COUNT', '')}

    def test_groupby_sum(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', {'sum': SUM('another')})
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'sum']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['sum'] == {('PROGRAM', 'another')}

    def test_join(self):
        path = 'files/test-frame.csv'
        real_path = os.path.realpath(path)
        t1 = XFrame(path)
        t2 = XFrame({'id': [1, 2, 3], 'doubled': ['aa', 'bb', 'cc']})
        res = t1.join(t2).sort('id').head()
        lineage = res.lineage()['column']
        assert sorted(lineage.keys()) == ['doubled', 'id', 'val']
        assert lineage['id'] == {('PROGRAM', 'id'), (real_path, 'id')}
        assert lineage['val'] == {(real_path, 'val')}
        assert lineage['doubled'] == {('PROGRAM', 'doubled')}

    def test_sort(self):
        t = XFrame({'id': [3, 2, 1], 'val': ['c', 'b', 'a']})
        res = t.sort('id')
        lineage = res.lineage()['column']
        assert len(lineage) == 2
        assert sorted(lineage.keys()) == ['id', 'val']
        assert lineage['id'] == {('PROGRAM', 'id')}
        assert lineage['val'] == {('PROGRAM', 'val')}


# noinspection PyClassHasNoInit
class TestXFrameNumRows:
    """
    Tests XFrame num_rows
    """

    def test_num_rows(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        assert t.num_rows() == 3


# noinspection PyClassHasNoInit
class TestXFrameNumColumns:
    """
    Tests XFrame num_columns
    """

    def test_num_columns(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        assert t.num_columns() == 2


# noinspection PyClassHasNoInit
class TestXFrameColumnNames:
    """
    Tests XFrame column_names
    """

    def test_column_names(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        names = t.column_names()
        assert names == ['id', 'val']


# noinspection PyClassHasNoInit
class TestXFrameColumnTypes:
    """
    Tests XFrame column_types
    """

    def test_column_types(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        types = t.column_types()
        assert types == [int, str]


# noinspection PyClassHasNoInit
class TestXFrameSelectRows:
    """
    Tests XFrame select_rows
    """

    def test_select_rowsr(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        a = XArray([1, 0, 1])
        res = t.select_rows(a)
        assert len(res) == 2
        assert list(res['id']) == [1, 3]
        assert list(res['val']) == ['a', 'c']


# noinspection PyClassHasNoInit
class TestXFrameHead:
    """
    Tests XFrame head
    """

    def test_head(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        hd = t.head(2)
        assert len(hd) == 2
        assert list(hd['id']) == [1, 2]
        assert list(hd['val']) == ['a', 'b']


# noinspection PyClassHasNoInit
class TestXFrameTail:
    """
    Tests XFrame tail
    """

    def test_tail(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        tl = t.tail(2)
        assert len(tl) == 2
        assert list(tl['id']) == [2, 3]
        assert list(tl['val']) == ['b', 'c']


# noinspection PyClassHasNoInit
class TestXFrameToPandasDataframe:
    """
    Tests XFrame to_pandas_dataframe
    """

    # Note: with numpy and pandas, use expr == True instead of
    #  expr is True.  This is because numpy(pandas) boolean is of type
    #   <type 'numpy.bool_'> and not bool.
    def test_to_pandas_dataframe_str(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        df = t.to_pandas_dataframe()
        assert len(df) == 3
        assert df['id'][0] == 1
        assert df['id'][1] == 2
        assert df['val'][0] == 'a'

    def test_to_pandas_dataframe_bool(self):
        t = XFrame({'id': [1, 2, 3], 'val': [True, False, True]})
        df = t.to_pandas_dataframe()
        assert len(df) == 3
        assert df['id'][0] == 1
        assert df['id'][1] == 2
        assert df['val'][0] == True
        assert df['val'][1] == False

    def test_to_pandas_dataframe_float(self):
        t = XFrame({'id': [1, 2, 3], 'val': [1.0, 2.0, 3.0]})
        df = t.to_pandas_dataframe()
        assert len(df) == 3
        assert df['id'][0] == 1
        assert df['id'][1] == 2
        assert df['val'][0] == 1.0
        assert df['val'][1] == 2.0

    def test_to_pandas_dataframe_int(self):
        t = XFrame({'id': [1, 2, 3], 'val': [1, 2, 3]})
        df = t.to_pandas_dataframe()
        assert len(df) == 3
        assert df['id'][0] == 1
        assert df['id'][1] == 2
        assert df['val'][0] == 1
        assert df['val'][1] == 2

    def test_to_pandas_dataframe_list(self):
        t = XFrame({'id': [1, 2, 3], 'val': [[1, 1], [2, 2], [3, 3]]})
        df = t.to_pandas_dataframe()
        assert len(df) == 3
        assert df['id'][0] == 1
        assert df['id'][1] == 2
        assert df['val'][0] == [1, 1]
        assert df['val'][1] == [2, 2]

    def test_to_pandas_dataframe_map(self):
        t = XFrame({'id': [1, 2, 3], 'val': [{'x': 1}, {'y': 2}, {'z': 3}]})
        df = t.to_pandas_dataframe()
        assert len(df) == 3
        assert df['id'][0] == 1
        assert df['id'][1] == 2
        assert df['val'][0] == {'x': 1}
        assert df['val'][1] == {'y': 2}


# noinspection PyClassHasNoInit
class TestXFrameForeach:
    """
    Tests XFrame foreach
    """

    def test_foreach(self):
        path = 'tmp/foreach.csv'
        # truncate file
        with open(path, 'w'):
            pass
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})

        # noinspection PyUnusedLocal
        def append_to_file(row, ini):
            with open(path, 'a') as f:
                f.write('{},{}\n'.format(row['id'], row['val']))
        t.foreach(append_to_file)
        # Read back as an XFrame
        res = XFrame.read_csv(path, header=False)
        res = res.rename(['id', 'val']).sort('id')
        assert len(res) == 3
        assert res.dtype() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}

    def test_foreach_init(self):
        path = 'tmp/foreach.csv'
        # truncate file
        with open(path, 'w'):
            pass
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})

        def append_to_file(row, ini):
            with open(path, 'a') as f:
                f.write('{},{},{}\n'.format(row['id'], row['val'], ini))

        def add_to_file():
            return 99

        t.foreach(append_to_file, add_to_file)
        # Read back as an XFrame
        res = XFrame.read_csv(path, header=False)
        res = res.rename(['id', 'val', 'ini']).sort('id')
        res = res.filterby([1, 2, 3], 'id')
        assert len(res) == 3
        assert res.dtype() == [int, str, int]
        assert res[0] == {'id': 1, 'val': 'a', 'ini': 99}
        assert res[1] == {'id': 2, 'val': 'b', 'ini': 99}


# noinspection PyClassHasNoInit
class TestXFrameApply:
    """
    Tests XFrame apply
    """

    def test_apply(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.apply(lambda row: row['id'] * 2)
        assert len(res) == 3
        assert res.dtype() is int
        assert list(res) == [2, 4, 6]

    def test_apply_float(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.apply(lambda row: row['id'] * 2, dtype=float)
        assert len(res) == 3
        assert res.dtype() is float
        assert list(res) == [2.0, 4.0, 6.0]

    def test_apply_str(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.apply(lambda row: row['id'] * 2, dtype=str)
        assert len(res) == 3
        assert res.dtype() is str
        assert list(res) == ['2', '4', '6']


# noinspection PyClassHasNoInit
class TestXFrameTransformCol:
    """
    Tests XFrame transform_col
    """

    def test_transform_col_identity(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.transform_col('id')
        assert len(res) == 3
        assert res.dtype() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}

    def test_transform_col_lambda(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.transform_col('id', lambda row: row['id'] * 2)
        assert len(res) == 3
        assert res.dtype() == [int, str]
        assert res[0] == {'id': 2, 'val': 'a'}
        assert res[1] == {'id': 4, 'val': 'b'}

    def test_transform_col_type(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.transform_col('id', lambda row: 'x' * row['id'])
        assert len(res) == 3
        assert res.dtype() == [str, str]
        assert res[0] == {'id': 'x', 'val': 'a'}
        assert res[1] == {'id': 'xx', 'val': 'b'}

    def test_transform_col_cast(self):
        t = XFrame({'id': ['1', '2', '3'], 'val': ['a', 'b', 'c']})
        res = t.transform_col('id', dtype=int)
        assert len(res) == 3
        assert res.dtype() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}


# noinspection PyClassHasNoInit
class TestXFrameTransformCols:
    """
    Tests XFrame transform_cols
    """

    def test_transform_cols_identity(self):
        t = XFrame({'other': ['x', 'y', 'z'], 'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.transform_cols(['id', 'val'])
        assert len(res) == 3
        assert res.dtype() == [int, str, str]
        assert res[0] == {'other': 'x', 'id': 1, 'val': 'a'}
        assert res[1] == {'other': 'y', 'id': 2, 'val': 'b'}

    def test_transform_cols_lambda(self):
        t = XFrame({'other': ['x', 'y', 'z'], 'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.transform_cols(['id', 'val'], lambda row: [row['id'] * 2, row['val'] + 'x'])
        assert len(res) == 3
        assert res.dtype() == [int, str, str]
        assert res[0] == {'other': 'x', 'id': 2, 'val': 'ax'}
        assert res[1] == {'other': 'y', 'id': 4, 'val': 'bx'}

    def test_transform_cols_type(self):
        t = XFrame({'other': ['x', 'y', 'z'], 'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.transform_cols(['id', 'val'], lambda row: ['x' * row['id'], ord(row['val'][0])])
        assert len(res) == 3
        assert res.dtype() == [str, str, int]
        assert res[0] == {'other': 'x', 'id': 'x', 'val': 97}
        assert res[1] == {'other': 'y', 'id': 'xx', 'val': 98}

    def test_transform_cols_cast(self):
        t = XFrame({'other': ['x', 'y', 'z'], 'id': ['1', '2', '3'], 'val': [10, 20, 30]})
        res = t.transform_cols(['id', 'val'], dtypes=[int, str])
        assert len(res) == 3
        assert res.dtype() == [int, str, str]
        assert res[0] == {'other': 'x', 'id': 1, 'val': '10'}
        assert res[1] == {'other': 'y', 'id': 2, 'val': '20'}


# noinspection PyClassHasNoInit
class TestXFrameFlatMap:
    """
    Tests XFrame flat_map
    """

    def test_flat_map(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.flat_map(['number', 'letter'],
                         lambda row: [list(six.itervalues(row)) for _ in range(0, row['id'])],
                         column_types=[int, str])
        assert res.column_names() == ['number', 'letter']
        assert res.dtype() == [int, str]
        assert res[0] == {'number': 1, 'letter': 'a'}
        assert res[1] == {'number': 2, 'letter': 'b'}
        assert res[2] == {'number': 2, 'letter': 'b'}
        assert res[3] == {'number': 3, 'letter': 'c'}
        assert res[4] == {'number': 3, 'letter': 'c'}
        assert res[5] == {'number': 3, 'letter': 'c'}

    def test_flat_map_identity(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.flat_map(['number', 'letter'],
                         lambda row: [[row['id'], row['val']]],
                         column_types=[int, str])
        assert res.column_names() == ['number', 'letter']
        assert res.dtype() == [int, str]
        assert res[0] == {'number': 1, 'letter': 'a'}
        assert res[1] == {'number': 2, 'letter': 'b'}
        assert res[2] == {'number': 3, 'letter': 'c'}

    def test_flat_map_mapped(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.flat_map(['number', 'letter'],
                         lambda row: [[row['id'] * 2, row['val'] + 'x']],
                         column_types=[int, str])
        assert res.column_names() == ['number', 'letter']
        assert res.dtype() == [int, str]
        assert res[0] == {'number': 2, 'letter': 'ax'}
        assert res[1] == {'number': 4, 'letter': 'bx'}
        assert res[2] == {'number': 6, 'letter': 'cx'}

    def test_flat_map_auto(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.flat_map(['number', 'letter'],
                         lambda row: [[row['id'] * 2, row['val'] + 'x']])
        assert res.column_names() == ['number', 'letter']
        assert res.dtype() == [int, str]
        assert res[0] == {'number': 2, 'letter': 'ax'}
        assert res[1] == {'number': 4, 'letter': 'bx'}
        assert res[2] == {'number': 6, 'letter': 'cx'}

        # TODO: test auto error cases


# noinspection PyClassHasNoInit
class TestXFrameSample:
    """
    Tests XFrame sample
    """

    @pytest.mark.skip(reason='depends on number of partitions')
    def test_sample_02(self):
        t = XFrame({'id': [1, 2, 3, 4, 5], 'val': ['a', 'b', 'c', 'd', 'e']})
        res = t.sample(0.2, 2)
        assert len(res) == 1
        assert res[0] == {'id': 2, 'val': 'b'}

    @pytest.mark.skip(reason='depends on number of partitions')
    def test_sample_08(self):
        t = XFrame({'id': [1, 2, 3, 4, 5], 'val': ['a', 'b', 'c', 'd', 'e']})
        res = t.sample(0.8, 3)
        assert len(res) == 3
        assert res[0] == {'id': 2, 'val': 'b'}
        assert res[1] == {'id': 4, 'val': 'd'}
        assert res[2] == {'id': 5, 'val': 'e'}


# noinspection PyClassHasNoInit
class TestXFrameRandomSplit:
    """
    Tests XFrame random_split
    """

    @pytest.mark.skip(reason='depends on number of partitions')
    def test_random_split(self):
        t = XFrame({'id': [1, 2, 3, 4, 5], 'val': ['a', 'b', 'c', 'd', 'e']})
        res1, res2 = t.random_split(0.5, 1)
        assert len(res1) == 3
        assert res1[0] == {'id': 1, 'val': 'a'}
        assert res1[1] == {'id': 4, 'val': 'd'}
        assert res1[2] == {'id': 5, 'val': 'e'}
        assert len(res2) == 2
        assert res2[0] == {'id': 2, 'val': 'b'}
        assert res2[1] == {'id': 3, 'val': 'c'}


# noinspection PyClassHasNoInit
class TestXFrameTopk:
    """
    Tests XFrame topk
    """

    def test_topk_int(self):
        t = XFrame({'id': [10, 20, 30], 'val': ['a', 'b', 'c']})
        res = t.topk('id', 2)
        assert len(res) == 2
        # noinspection PyUnresolvedReferences
        assert (XArray([30, 20]) == res['id']).all()
        assert list(res['val']) == ['c', 'b']
        assert res.column_types() == [int, str]
        assert res.column_names() == ['id', 'val']

    def test_topk_int_reverse(self):
        t = XFrame({'id': [30, 20, 10], 'val': ['c', 'b', 'a']})
        res = t.topk('id', 2, reverse=True)
        assert len(res) == 2
        assert list(res['id']) == [10, 20]
        assert list(res['val']) == ['a', 'b']

    # noinspection PyUnresolvedReferences
    def test_topk_float(self):
        t = XFrame({'id': [10.0, 20.0, 30.0], 'val': ['a', 'b', 'c']})
        res = t.topk('id', 2)
        assert len(res) == 2
        assert (XArray([30.0, 20.0]) == res['id']).all()
        assert list(res['val']) == ['c', 'b']
        assert res.column_types() == [float, str]
        assert res.column_names() == ['id', 'val']

    def test_topk_float_reverse(self):
        t = XFrame({'id': [30.0, 20.0, 10.0], 'val': ['c', 'b', 'a']})
        res = t.topk('id', 2, reverse=True)
        assert len(res) == 2
        assert list(res['id']) == [10.0, 20.0]
        assert list(res['val']) == ['a', 'b']

    def test_topk_str(self):
        t = XFrame({'id': [30, 20, 10], 'val': ['a', 'b', 'c']})
        res = t.topk('val', 2)
        assert len(res) == 2
        assert list(res['id']) == [10, 20]
        assert list(res['val']) == ['c', 'b']
        assert res.column_types() == [int, str]
        assert res.column_names() == ['id', 'val']

    def test_topk_str_reverse(self):
        t = XFrame({'id': [10, 20, 30], 'val': ['c', 'b', 'a']})
        res = t.topk('val', 2, reverse=True)
        assert len(res) == 2
        assert list(res['id']) == [30, 20]
        assert list(res['val']) == ['a', 'b']


# noinspection PyClassHasNoInit
class TestXFrameSaveBinary:
    """
    Tests XFrame save binary format
    """

    def test_save(self):
        t = XFrame({'id': [30, 20, 10], 'val': ['a', 'b', 'c']})
        path = 'tmp/frame'
        t.save(path, format='binary')
        with open(os.path.join(path, '_metadata')) as f:
            metadata = pickle.load(f)
            assert metadata == [['id', 'val'], [int, str]]
        # TODO find some way to check the data

    def test_save_not_exist(self, tmpdir):
        path = os.path.join(str(tmpdir), 'frame')
        t = XFrame({'id': [30, 20, 10], 'val': ['a', 'b', 'c']})
        t.save(path, format='binary')
        # TODO find some way to check the data


# noinspection PyClassHasNoInit
class TestXFrameSaveCsv:
    """
    Tests XFrame save csv format
    """

    def test_save(self):
        t = XFrame({'id': [30, 20, 10], 'val': ['a', 'b', 'c']})
        path = 'tmp/frame-csv'
        t.save(path, format='csv')

        with open(path + '.csv') as f:
            heading = f.readline().rstrip()
            assert heading == 'id,val'
            assert f.readline().rstrip() == '30,a'
            assert f.readline().rstrip() == '20,b'
            assert f.readline().rstrip() == '10,c'

    def test_save_not_exist(self, tmpdir):
        path = os.path.join(str(tmpdir), 'frame')
        t = XFrame({'id': [30, 20, 10], 'val': ['a', 'b', 'c']})
        t.save(path, format='csv')
        # TODO find some way to check the data


# noinspection PyClassHasNoInit
class TestXFrameSaveParquet:
    """
    Tests XFrame save for parquet files
    """

    def test_save(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        path = 'tmp/frame-parquet'
        t.save(path, format='parquet')
        res = XFrame(path + '.parquet')
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_save_as_parquet(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        path = 'tmp/frame-parquet'
        t.save_as_parquet(path)
        res = XFrame(path, format='parquet')
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_save_rename(self):
        t = XFrame({'id col': [1, 2, 3], 'val,col': ['a', 'b', 'c']})
        path = 'tmp/frame-parquet'
        t.save(path, format='parquet')
        res = XFrame(path + '.parquet')
        assert res.column_names() == ['id_col', 'val_col']
        assert res.column_types() == [int, str]
        assert res[0] == {'id_col': 1, 'val_col': 'a'}
        assert res[1] == {'id_col': 2, 'val_col': 'b'}
        assert res[2] == {'id_col': 3, 'val_col': 'c'}

    def test_save_as_parquet_rename(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        path = 'tmp/frame-parquet'
        t.save_as_parquet(path, column_names=['id1', 'val1'])
        res = XFrame(path, format='parquet')
        assert res.column_names() == ['id1', 'val1']
        assert res.column_types() == [int, str]
        assert res[0] == {'id1': 1, 'val1': 'a'}
        assert res[1] == {'id1': 2, 'val1': 'b'}
        assert res[2] == {'id1': 3, 'val1': 'c'}

    def test_save_not_exist(self):
        t = XFrame({'id': [30, 20, 10], 'val': ['a', 'b', 'c']})
        path = 'xxx/frame'
        t.save_as_parquet(path)
        # TODO find some way to check the data


# noinspection PyClassHasNoInit
class TestXFrameSelectColumn:
    """
    Tests XFrame select_column
    """

    def test_select_column_id(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.select_column('id')
        assert list(res) == [1, 2, 3]

    def test_select_column_val(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.select_column('val')
        assert list(res) == ['a', 'b', 'c']

    def test_select_column_bad_name(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(ValueError) as exception_info:
            t.select_column('xx')
        exception_message = exception_info.value.args[0]
        assert exception_message == "Column name does not exist: 'xx'."

    # noinspection PyTypeChecker
    def test_select_column_bad_type(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            t.select_column(1)
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Invalid column_name type must be str.'


# noinspection PyClassHasNoInit
class TestXFrameSelectColumns:
    """
    Tests XFrame select_columns
    """

    def test_select_columns_id_val(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        res = t.select_columns(['id', 'val'])
        assert res[0] == {'id': 1, 'val': 'a'}

    def test_select_columns_id(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        res = t.select_columns(['id'])
        assert res[0] == {'id': 1}

    # noinspection PyTypeChecker
    def test_select_columns_not_iterable(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        with pytest.raises(TypeError) as exception_info:
            t.select_columns(1)
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Keylist must be an iterable.'

    def test_select_columns_bad_type(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        with pytest.raises(TypeError) as exception_info:
            t.select_columns(['id', 2])
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Invalid key type: must be str.'

    def test_select_columns_bad_dup(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        with pytest.raises(ValueError) as exception_info:
            t.select_columns(['id', 'id'])
        exception_message = exception_info.value.args[0]
        assert exception_message == "There are duplicate keys in key list: 'id'."


# noinspection PyClassHasNoInit
class TestXFrameAddColumn:
    """
    Tests XFrame add_column
    """

    def test_add_column_named(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta = XArray([3.0, 2.0, 1.0])
        res = tf.add_column(ta, name='another')
        assert res.column_names() == ['id', 'val', 'another']
        assert res[0] == {'id': 1, 'val': 'a', 'another': 3.0}

    def test_add_column_name_default(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta = XArray([3.0, 2.0, 1.0])
        res = tf.add_column(ta)
        assert res.column_names() == ['id', 'val', 'X.2']
        assert res[0] == {'id': 1, 'val': 'a', 'X.2': 3.0}

    def test_add_column_name_dup(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta = XArray([3.0, 2.0, 1.0])
        res = tf.add_column(ta, name='id')
        assert res.column_names() == ['id', 'val', 'id.2']
        assert res[0] == {'id': 1, 'val': 'a', 'id.2': 3.0}


# noinspection PyClassHasNoInit
class TestXFrameAddColumnsArray:
    """
    Tests XFrame add_columns where data is array of XArray
    """

    def test_add_columns_one(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta = XArray([3.0, 2.0, 1.0])
        res = tf.add_columns([ta], names=['new1'])
        assert res.column_names() == ['id', 'val', 'new1']
        assert res.column_types() == [int, str, float]
        assert res[0] == {'id': 1, 'val': 'a', 'new1': 3.0}

    def test_add_columns_two(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta1 = XArray([3.0, 2.0, 1.0])
        ta2 = XArray([30.0, 20.0, 10.0])
        res = tf.add_columns([ta1, ta2], names=['new1', 'new2'])
        assert res.column_names() == ['id', 'val', 'new1', 'new2']
        assert res.column_types() == [int, str, float, float]
        assert res[0] == {'id': 1, 'val': 'a', 'new1': 3.0, 'new2': 30.0}

    def test_add_columns_namelist_missing(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta1 = XArray([3.0, 2.0, 1.0])
        ta2 = XArray([30.0, 20.0, 10.0])
        with pytest.raises(TypeError) as exception_info:
            tf.add_columns([ta1, ta2])
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Namelist must be an iterable.'

    # noinspection PyTypeChecker
    def test_add_columns_data_not_iterable(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            tf.add_columns(1, names=[])
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Column list must be an iterable.'

    # noinspection PyTypeChecker
    def test_add_columns_namelist_not_iterable(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta1 = XArray([3.0, 2.0, 1.0])
        ta2 = XArray([30.0, 20.0, 10.0])
        with pytest.raises(TypeError) as exception_info:
            tf.add_columns([ta1, ta2], names=1)
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Namelist must be an iterable.'

    def test_add_columns_not_xarray(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta1 = XArray([3.0, 2.0, 1.0])
        ta2 = [30.0, 20.0, 10.0]
        with pytest.raises(TypeError) as exception_info:
            tf.add_columns([ta1, ta2], names=['new1', 'new2'])
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Must give column as XArray.'

    def test_add_columns_name_not_str(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta1 = XArray([3.0, 2.0, 1.0])
        ta2 = XArray([30.0, 20.0, 10.0])
        with pytest.raises(TypeError) as exception_info:
            tf.add_columns([ta1, ta2], names=['new1', 1])
        exception_message = exception_info.value.args[0]
        assert exception_message == "Invalid column name in list : must all be str."


# noinspection PyClassHasNoInit
class TestXFrameAddColumnsFrame:
    """
    Tests XFrame add_columns where data is XFrame
    """

    def test_add_columns(self):
        tf1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        tf2 = XFrame({'new1': [3.0, 2.0, 1.0], 'new2': [30.0, 20.0, 10.0]})
        res = tf1.add_columns(tf2)
        assert res.column_names() == ['id', 'val', 'new1', 'new2']
        assert res.column_types() == [int, str, float, float]
        assert res[0] == {'id': 1, 'val': 'a', 'new1': 3.0, 'new2': 30.0}

    def test_add_columns_dup_names(self):
        tf1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        tf2 = XFrame({'new1': [3.0, 2.0, 1.0], 'val': [30.0, 20.0, 10.0]})
        res = tf1.add_columns(tf2)
        assert res.column_names() == ['id', 'val', 'new1', 'val.1']
        assert res.column_types() == [int, str, float, float]
        assert res[0] == {'id': 1, 'val': 'a', 'new1': 3.0, 'val.1': 30.0}


# noinspection PyClassHasNoInit
class TestXFrameReplaceColumn:
    """
    Tests XFrame replace_column
    """

    def test_replace_column(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        a = XArray(['x', 'y', 'z'])
        res = t.replace_column('val', a)
        assert res.column_names() == ['id', 'val']
        assert res[0] == {'id': 1, 'val': 'x'}

    # noinspection PyTypeChecker
    def test_replace_column_bad_col_type(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            t.replace_column('val', ['x', 'y', 'z'])
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Must give column as XArray.'

    def test_replace_column_bad_name(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        a = XArray(['x', 'y', 'z'])
        with pytest.raises(ValueError) as exception_info:
            t.replace_column('xx', a)
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Column name must be in XFrame.'

    # noinspection PyTypeChecker
    def test_replace_column_bad_name_type(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        a = XArray(['x', 'y', 'z'])
        with pytest.raises(TypeError) as exception_info:
            t.replace_column(2, a)
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Invalid column name: must be str.'


# noinspection PyClassHasNoInit
class TestXFrameRemoveColumn:
    """
    Tests XFrame remove_column
    """

    def test_remove_column(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        res = t.remove_column('another')
        assert res[0] == {'id': 1, 'val': 'a'}
        assert len(t.column_names()) == 3
        assert len(res.column_names()) == 2

    def test_remove_column_not_found(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        with pytest.raises(KeyError) as exception_info:
            t.remove_column('xx')
        exception_message = exception_info.value.args[0]
        assert exception_message == "Cannot find column 'xx'."

    def test_remove_column_many(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'new1': [3.0, 2.0, 1.0], 'new2': [30.0, 20.0, 10.0]})
        res = t.remove_column(['new1', 'new2'])
        assert res[0] == {'id': 1, 'val': 'a'}
        assert len(t.column_names()) == 4
        assert len(res.column_names()) == 2

    def test_remove_column_many_bad_type(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        with pytest.raises(TypeError) as exception_info:
            t.remove_columns(10)
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Column_names must be an iterable.'


# noinspection PyClassHasNoInit
class TestXFrameRemoveColumns:
    """
    Tests XFrame remove_columns
    """

    def test_remove_columns(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'new1': [3.0, 2.0, 1.0], 'new2': [30.0, 20.0, 10.0]})
        res = t.remove_columns(['new1', 'new2'])
        assert res[0] == {'id': 1, 'val': 'a'}
        assert len(t.column_names()) == 4
        assert len(res.column_names()) == 2

    def test_remove_column_not_iterable(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        with pytest.raises(TypeError) as exception_info:
            t.remove_columns('xx')
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Column_names must be an iterable.'

    def test_remove_column_not_found(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        with pytest.raises(KeyError) as exception_info:
            t.remove_columns(['xx'])
        exception_message = exception_info.value.args[0]
        assert exception_message == "Cannot find column 'xx'."


# noinspection PyClassHasNoInit
class TestXFrameSwapColumns:
    """
    Tests XFrame swap_columns
    """

    def test_swap_columns(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'x': [3.0, 2.0, 1.0]})
        res = t.swap_columns('val', 'x')
        assert res.column_names() == ['id', 'x', 'val']
        assert t.column_names() == ['id', 'val', 'x']
        assert res[0] == {'id': 1, 'x': 3.0, 'val': 'a'}

    def test_swap_columns_bad_col_1(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        with pytest.raises(KeyError) as exception_info:
            t.swap_columns('xx', 'another')
        exception_message = exception_info.value.args[0]
        assert exception_message == "Cannot find column 'xx'."

    def test_swap_columns_bad_col_2(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        with pytest.raises(KeyError) as exception_info:
            t.swap_columns('val', 'xx')
        exception_message = exception_info.value.args[0]
        assert exception_message == "Cannot find column 'xx'."


# noinspection PyClassHasNoInit
class TestXFrameReorderColumns:
    """
    Tests XFrame reorder_columns
    """

    def test_reorder_columns(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'x': [3.0, 2.0, 1.0]})
        res = t.reorder_columns(['val', 'x', 'id'])
        assert res.column_names() == ['val', 'x', 'id']
        assert t.column_names() == ['id', 'val', 'x']
        assert res[0] == {'id': 1, 'x': 3.0, 'val': 'a'}

    # noinspection PyTypeChecker
    def test_reorder_columns_list_not_iterable(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'x': [3.0, 2.0, 1.0]})
        with pytest.raises(TypeError) as exception_info:
            t.reorder_columns('val')
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Keylist must be an iterable.'

    def test_reorder_columns_bad_col(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'x': [3.0, 2.0, 1.0]})
        with pytest.raises(KeyError) as exception_info:
            t.reorder_columns(['val', 'y', 'id'])
        exception_message = exception_info.value.args[0]
        assert exception_message == "Cannot find column 'y'."

    def test_reorder_columns_incomplete(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'x': [3.0, 2.0, 1.0]})
        with pytest.raises(KeyError) as exception_info:
            t.reorder_columns(['val', 'id'])
        exception_message = exception_info.value.args[0]
        assert exception_message == "Column 'x' not assigned'."


# noinspection PyClassHasNoInit
class TestXFrameRename:
    """
    Tests XFrame rename
    """

    def test_rename(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.rename({'id': 'new_id'})
        assert res.column_names() == ['new_id', 'val']
        assert t.column_names() == ['id', 'val']
        assert res[0] == {'new_id': 1, 'val': 'a'}

    # noinspection PyTypeChecker
    def test_rename_arg_not_dict(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            t.rename('id')
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Names must be a dictionary: oldname -> newname or a list of newname (str).'

    def test_rename_col_not_found(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(ValueError) as exception_info:
            t.rename({'xx': 'new_id'})
        exception_message = exception_info.value.args[0]
        assert exception_message == "Cannot find column 'xx' in the XFrame."

    def test_rename_bad_length(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(ValueError) as exception_info:
            t.rename(['id'])
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Names must be the same length as the number of columns (names: 1 columns: 2).'

    def test_rename_list(self):
        t = XFrame({'X.0': [1, 2, 3], 'X.1': ['a', 'b', 'c']})
        res = t.rename(['id', 'val'])
        assert res.column_names() == ['id', 'val']
        assert t.column_names() == ['X.0', 'X.1']
        assert res[0] == {'id': 1, 'val': 'a'}


# noinspection PyClassHasNoInit
class TestXFrameGetitem:
    """
    Tests XFrame __getitem__
    """

    def test_getitem_str(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t['id']
        assert list(res) == [1, 2, 3]

    def test_getitem_int_pos(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t[1]
        assert res == {'id': 2, 'val': 'b'}

    def test_getitem_int_neg(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t[-2]
        assert res == {'id': 2, 'val': 'b'}

    def test_getitem_int_too_low(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(IndexError) as exception_info:
            _ = t[-100]
        exception_message = exception_info.value.args[0]
        assert exception_message == 'XFrame index out of range (too low).'

    def test_getitem_int_too_high(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(IndexError) as exception_info:
            _ = t[100]
        exception_message = exception_info.value.args[0]
        assert exception_message == 'XFrame index out of range (too high).'

    def test_getitem_slice(self):
        # TODO we could test more variations of slice
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t[:2]
        assert len(res) == 2
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}

    def test_getitem_list(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'x': [1.0, 2.0, 3.0]})
        res = t[['id', 'x']]
        assert res[1] == {'id': 2, 'x': 2.0}

    def test_getitem_bad_type(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            _ = t[{'a': 1}]
        exception_message = exception_info.value.args[0]
        assert exception_message == "Invalid index type: must be XArray, 'int', 'list', slice, or 'str': (dict)."

    # TODO: need to implement
    def test_getitem_xarray(self):
        pass


# noinspection PyClassHasNoInit
class TestXFrameSetitem:
    """
    Tests XFrame __setitem__
    """

    def test_setitem_float_const(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t['x'] = 5.0
        assert t.column_names() == ['id', 'val', 'x']
        assert t[1] == {'id': 2, 'val': 'b', 'x': 5.0}

    def test_setitem_str_const_replace(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t['val'] = 'x'
        assert t.column_names() == ['id', 'val']
        assert t[1] == {'id': 2, 'val': 'x'}

    def test_setitem_list(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta1 = XArray([3.0, 2.0, 1.0])
        ta2 = XArray([30.0, 20.0, 10.0])
        tf[['new1', 'new2']] = [ta1, ta2]
        assert tf.column_names() == ['id', 'val', 'new1', 'new2']
        assert tf.column_types() == [int, str, float, float]
        assert tf[0] == {'id': 1, 'val': 'a', 'new1': 3.0, 'new2': 30.0}

    def test_setitem_str_iter(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t['x'] = [1.0, 2.0, 3.0]
        assert t.column_names() == ['id', 'val', 'x']
        assert t[1] == {'id': 2, 'val': 'b', 'x': 2.0}

    def test_setitem_str_xarray(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta = XArray([3.0, 2.0, 1.0])
        tf['new'] = ta
        assert tf.column_names() == ['id', 'val', 'new']
        assert tf.column_types() == [int, str, float]
        assert tf[0] == {'id': 1, 'val': 'a', 'new': 3.0}

    def test_setitem_str_iter_replace(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t['val'] = [1.0, 2.0, 3.0]
        assert t.column_names() == ['id', 'val']
        assert t[1] == {'id': 2, 'val': 2.0}

    def test_setitem_bad_key(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            t[{'a': 1}] = [1.0, 2.0, 3.0]
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Cannot set column with key type dict.'

    def test_setitem_str_iter_replace_one_col(self):
        t = XFrame({'val': ['a', 'b', 'c']})
        t['val'] = [1.0, 2.0, 3.0, 4.0]
        assert t.column_names() == ['val']
        assert len(t) == 4
        assert t[1] == {'val': 2.0}


# noinspection PyClassHasNoInit
class TestXFrameDelitem:
    """
    Tests XFrame __delitem__
    """

    def test_delitem(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        del t['another']
        assert t[0] == {'id': 1, 'val': 'a'}

    def test_delitem_not_found(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        with pytest.raises(KeyError) as exception_info:
            del t['xx']
        exception_message = exception_info.value.args[0]
        assert exception_message == "Cannot find column 'xx'."


# noinspection PyClassHasNoInit
class TestXFrameIsMaterialized:
    """
    Tests XFrame _is_materialized
    """

    def test_is_materialized_false(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        assert t._is_materialized() is False

    def test_is_materialized(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        len(t)
        assert t._is_materialized() is True


# noinspection PyClassHasNoInit
class TestXFrameIter:
    """
    Tests XFrame __iter__
    """

    def test_iter(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        expect_id = [1, 2, 3]
        expect_val = ['a', 'b', 'c']
        for item in zip(t, expect_id, expect_val):
            assert item[0]['id'] == item[1]
            assert item[0]['val'] == item[2]


# noinspection PyClassHasNoInit
class TestXFrameRange:
    """
    Tests XFrame range
    """

    def test_range_int_pos(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.range(1)
        assert res == {'id': 2, 'val': 'b'}

    def test_range_int_neg(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.range(-2)
        assert res == {'id': 2, 'val': 'b'}

    def test_range_int_too_low(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(IndexError) as exception_info:
            _ = t.range(-100)
        exception_message = exception_info.value.args[0]
        assert exception_message == 'XFrame index out of range (too low).'

    def test_range_int_too_high(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(IndexError) as exception_info:
            _ = t.range(100)
        exception_message = exception_info.value.args[0]
        assert exception_message == 'XFrame index out of range (too high).'

    def test_range_slice(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.range(slice(0, 2))
        assert len(res) == 2
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}

    # noinspection PyTypeChecker
    def test_range_bad_type(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            _ = t.range({'a': 1})
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Invalid argument type: must be int or slice (dict).'


# noinspection PyClassHasNoInit
class TestXFrameAppend:
    """
    Tests XFrame append
    """

    def test_append(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [10, 20, 30], 'val': ['aa', 'bb', 'cc']})
        res = t1.append(t2)
        assert len(res) == 6
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[3] == {'id': 10, 'val': 'aa'}

    # noinspection PyTypeChecker
    def test_append_bad_type(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(RuntimeError) as exception_info:
            t1.append(1)
        exception_message = exception_info.value.args[0]
        assert exception_message == 'XFrame append can only work with XFrame.'

    def test_append_both_empty(self):
        t1 = XFrame()
        t2 = XFrame()
        res = t1.append(t2)
        assert len(res) == 0

    def test_append_first_empty(self):
        t1 = XFrame()
        t2 = XFrame({'id': [10, 20, 30], 'val': ['aa', 'bb', 'cc']})
        res = t1.append(t2)
        assert len(res) == 3
        assert res[0] == {'id': 10, 'val': 'aa'}

    def test_append_second_empty(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame()
        res = t1.append(t2)
        assert len(res) == 3
        assert res[0] == {'id': 1, 'val': 'a'}

    def test_append_unequal_col_length(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [10, 20, 30], 'val': ['aa', 'bb', 'cc'], 'another': [1.0, 2.0, 3.0]})
        with pytest.raises(RuntimeError) as exception_info:
            t1.append(t2)
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Two XFrames must have the same number of columns.'

    def test_append_col_name_mismatch(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [10, 20, 30], 'xx': ['a', 'b', 'c']})
        with pytest.raises(RuntimeError) as exception_info:
            t1.append(t2)
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Column val name is not the same in two XFrames, one is val the other is xx.'

    def test_append_col_type_mismatch(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [10, 20], 'val': [1.0, 2.0]})
        with pytest.raises(RuntimeError) as exception_info:
            t1.append(t2)
        exception_message = exception_info.value.args[0]
        assert exception_message == "Column val type is not the same in two XFrames, " + \
                                    "one is <type 'str'> the other is [<type 'int'>, <type 'float'>]."


# noinspection PyClassHasNoInit
class TestXFrameGroupby:
    """
    Tests XFrame groupby
    """

    def test_groupby(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', {})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id']
        assert res.column_types() == [int]
        assert res[0] == {'id': 1}
        assert res[1] == {'id': 2}
        assert res[2] == {'id': 3}

    def test_groupby_nooperation(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id')
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id']
        assert res.column_types() == [int]
        assert res[0] == {'id': 1}
        assert res[1] == {'id': 2}
        assert res[2] == {'id': 3}

    # noinspection PyTypeChecker
    def test_groupby_bad_col_name_type(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        with pytest.raises(TypeError) as exception_info:
            t.groupby(1, {})
        exception_message = exception_info.value.args[0]
        assert exception_message == "'int' object is not iterable"

    # noinspection PyTypeChecker
    def test_groupby_bad_col_name_list_type(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        with pytest.raises(TypeError) as exception_info:
            t.groupby([1], {})
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Column name must be a string.'

    def test_groupby_bad_col_group_name(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        with pytest.raises(KeyError) as exception_info:
            t.groupby('xx', {})
        exception_message = exception_info.value.args[0]
        assert exception_message == "Column 'xx' does not exist in XFrame."

    def test_groupby_bad_group_type(self):
        t = XFrame({'id': [{1: 'a', 2: 'b'}, {3: 'c'}],
                    'val': ['a', 'b']})
        with pytest.raises(TypeError) as exception_info:
            t.groupby('id', {})
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Cannot group on a dictionary column.'

    def test_groupby_bad_agg_group_name(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        with pytest.raises(KeyError) as exception_info:
            t.groupby('id', SUM('xx'))
        exception_message = exception_info.value.args[0]
        assert exception_message == "Column 'xx' does not exist in XFrame."

    def test_groupby_bad_agg_group_type(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        with pytest.raises(TypeError) as exception_info:
            t.groupby('id', SUM(1))
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Column name must be a string.'


# noinspection PyClassHasNoInit
class TestXFrameGroupbyAggregators:
    """
    Tests XFrame groupby aggregators
    """

    def test_groupby_count(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', COUNT)
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'count']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': 1, 'count': 3}
        assert res[1] == {'id': 2, 'count': 2}
        assert res[2] == {'id': 3, 'count': 1}

    def test_groupby_count_call(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', COUNT())
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'count']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': 1, 'count': 3}
        assert res[1] == {'id': 2, 'count': 2}
        assert res[2] == {'id': 3, 'count': 1}

    def test_groupby_count_named(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', {'record-count': COUNT})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'record-count']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': 1, 'record-count': 3}
        assert res[1] == {'id': 2, 'record-count': 2}
        assert res[2] == {'id': 3, 'record-count': 1}

    def test_groupby_sum(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', {'sum': SUM('another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'sum']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': 1, 'sum': 110}
        assert res[1] == {'id': 2, 'sum': 70}
        assert res[2] == {'id': 3, 'sum': 30}

    def test_groupby_sum_def(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', SUM('another'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'sum']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': 1, 'sum': 110}
        assert res[1] == {'id': 2, 'sum': 70}
        assert res[2] == {'id': 3, 'sum': 30}

    def test_groupby_sum_sum_def(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', [SUM('another'), SUM('another')])
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'sum', 'sum.1']
        assert res.column_types() == [int, int, int]
        assert res[0] == {'id': 1, 'sum': 110, 'sum.1': 110}
        assert res[1] == {'id': 2, 'sum': 70, 'sum.1': 70}
        assert res[2] == {'id': 3, 'sum': 30, 'sum.1': 30}

    def test_groupby_sum_rename(self):
        t = XFrame({'sum': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('sum', SUM('another'))
        res = res.topk('sum', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['sum', 'sum.1']
        assert res.column_types() == [int, int]
        assert res[0] == {'sum': 1, 'sum.1': 110}
        assert res[1] == {'sum': 2, 'sum.1': 70}
        assert res[2] == {'sum': 3, 'sum.1': 30}

    def test_groupby_count_sum(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', {'count': COUNT, 'sum': SUM('another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'count', 'sum']
        assert res.column_types() == [int, int, int]
        assert res[0] == {'id': 1, 'count': 3, 'sum': 110}
        assert res[1] == {'id': 2, 'count': 2, 'sum': 70}
        assert res[2] == {'id': 3, 'count': 1, 'sum': 30}

    def test_groupby_count_sum_def(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', [COUNT, SUM('another')])
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'count', 'sum']
        assert res.column_types() == [int, int, int]
        assert res[0] == {'id': 1, 'count': 3, 'sum': 110}
        assert res[1] == {'id': 2, 'count': 2, 'sum': 70}
        assert res[2] == {'id': 3, 'count': 1, 'sum': 30}

    def test_groupby_argmax(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', ARGMAX('another', 'val'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'argmax']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'argmax': 'f'}
        assert res[1] == {'id': 2, 'argmax': 'e'}
        assert res[2] == {'id': 3, 'argmax': 'c'}

    def test_groupby_argmin(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', ARGMIN('another', 'val'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'argmin']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'argmin': 'a'}
        assert res[1] == {'id': 2, 'argmin': 'b'}
        assert res[2] == {'id': 3, 'argmin': 'c'}

    def test_groupby_max(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', MAX('another'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'max']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': 1, 'max': 60}
        assert res[1] == {'id': 2, 'max': 50}
        assert res[2] == {'id': 3, 'max': 30}

    def test_groupby_max_float(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', MAX('another'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'max']
        assert res.column_types() == [int, float]
        assert res[0] == {'id': 1, 'max': 60.0}
        assert res[1] == {'id': 2, 'max': 50.0}
        assert res[2] == {'id': 3, 'max': 30.0}

    def test_groupby_max_str(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', MAX('val'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'max']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'max': 'f'}
        assert res[1] == {'id': 2, 'max': 'e'}
        assert res[2] == {'id': 3, 'max': 'c'}

    def test_groupby_min(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', MIN('another'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'min']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': 1, 'min': 10}
        assert res[1] == {'id': 2, 'min': 20}
        assert res[2] == {'id': 3, 'min': 30}

    def test_groupby_min_float(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', MIN('another'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'min']
        assert res.column_types() == [int, float]
        assert res[0] == {'id': 1, 'min': 10.0}
        assert res[1] == {'id': 2, 'min': 20.0}
        assert res[2] == {'id': 3, 'min': 30.0}

    def test_groupby_min_str(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', MIN('val'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'min']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'min': 'a'}
        assert res[1] == {'id': 2, 'min': 'b'}
        assert res[2] == {'id': 3, 'min': 'c'}

    def test_groupby_mean(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', MEAN('another'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'mean']
        assert res.column_types() == [int, float]
        assert res[0] == {'id': 1, 'mean': 110.0 / 3.0}
        assert res[1] == {'id': 2, 'mean': 70.0 / 2.0}
        assert res[2] == {'id': 3, 'mean': 30}

    def test_groupby_variance(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', VARIANCE('another'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'variance']
        assert res.column_types() == [int, float]
        assert almost_equal(res[0]['variance'], 3800.0 / 9.0)
        assert almost_equal(res[1]['variance'], 225.0)
        assert res[2] == {'id': 3, 'variance': 0.0}

    def test_groupby_stdv(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', STDV('another'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'stdv']
        assert res.column_types() == [int, float]
        assert almost_equal(res[0]['stdv'], math.sqrt(3800.0 / 9.0))
        assert almost_equal(res[1]['stdv'], math.sqrt(225.0))
        assert res[2] == {'id': 3, 'stdv': 0.0}

    def test_groupby_select_one(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', SELECT_ONE('another'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'select-one']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': 1, 'select-one': 60}
        assert res[1] == {'id': 2, 'select-one': 50}
        assert res[2] == {'id': 3, 'select-one': 30}

    def test_groupby_select_one_float(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', SELECT_ONE('another'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'select-one']
        assert res.column_types() == [int, float]
        assert res[0] == {'id': 1, 'select-one': 60.0}
        assert res[1] == {'id': 2, 'select-one': 50.0}
        assert res[2] == {'id': 3, 'select-one': 30.0}

    def test_groupby_select_one_str(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', SELECT_ONE('val'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'select-one']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'select-one': 'f'}
        assert res[1] == {'id': 2, 'select-one': 'e'}
        assert res[2] == {'id': 3, 'select-one': 'c'}

    def test_groupby_concat_list(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', CONCAT('another'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'concat']
        assert res.column_types() == [int, list]
        assert res[0] == {'id': 1, 'concat': [10, 40, 60]}
        assert res[1] == {'id': 2, 'concat': [20, 50]}
        assert res[2] == {'id': 3, 'concat': [30]}

    def test_groupby_concat_dict(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', CONCAT('val', 'another'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'concat']
        assert res.column_types() == [int, dict]
        assert res[0] == {'id': 1, 'concat': {'a': 10, 'd': 40, 'f': 60}}
        assert res[1] == {'id': 2, 'concat': {'b': 20, 'e': 50}}
        assert res[2] == {'id': 3, 'concat': {'c': 30}}

    def test_groupby_values_list(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
                    'another': [10, 20, 30, 40, 50, 60, 10]})
        res = t.groupby('id', VALUES('another'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'values']
        assert res.column_types() == [int, list]
        assert dict_keys_equal(res[0], {'id': 1, 'values': [10, 40, 60]})
        assert dict_keys_equal(res[2], {'id': 2, 'values': [20, 50]})
        assert res[2] == {'id': 3, 'values': [30]}

    def test_groupby_values_count_list(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
                    'another': [10, 20, 30, 40, 50, 60, 10]})
        res = t.groupby('id', VALUES_COUNT('another'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'values-count']
        assert res.column_types() == [int, dict]
        assert dict_keys_equal(res[0], {'id': 1, 'values-count': {10: 2, 40: 1, 60: 1}})
        assert dict_keys_equal(res[2], {'id': 2, 'values-count': {20: 1, 50: 1}})
        assert res[2] == {'id': 3, 'values-count': {30: 1}}

    def test_groupby_quantile(self):
        # not implemented
        pass


# noinspection PyClassHasNoInit
class TestXFrameGroupbyAggregatorsWithMissingValues:
    """
    Tests XFrame groupby aggregators with missing values
    """

    # noinspection PyTypeChecker
    def test_groupby_count(self):
        t = XFrame({'id': [1, 2, 3, None, None, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', COUNT)
        res = res.topk('id', reverse=True)
        assert len(res) == 4
        assert res.column_names() == ['id', 'count']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': None, 'count': 2}
        assert res[1] == {'id': 1, 'count': 2}
        assert res[2] == {'id': 2, 'count': 1}
        assert res[3] == {'id': 3, 'count': 1}

    def test_groupby_sum(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, None, None, 60]})
        res = t.groupby('id', SUM('another'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'sum']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': 1, 'sum': 70}
        assert res[1] == {'id': 2, 'sum': 20}
        assert res[2] == {'id': 3, 'sum': 30}

    def test_groupby_argmax(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, None, None]})
        res = t.groupby('id', {'argmax': ARGMAX('another', 'val')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'argmax']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'argmax': 'd'}
        assert res[1] == {'id': 2, 'argmax': 'b'}
        assert res[2] == {'id': 3, 'argmax': 'c'}

    def test_groupby_argmin(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, None, 30, 40, 50, 60]})
        res = t.groupby('id', {'argmin': ARGMIN('another', 'val')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'argmin']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'argmin': 'a'}
        assert res[1] == {'id': 2, 'argmin': 'e'}
        assert res[2] == {'id': 3, 'argmin': 'c'}

    def test_groupby_max(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, None, None, 60]})
        res = t.groupby('id', {'max': MAX('another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'max']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': 1, 'max': 60}
        assert res[1] == {'id': 2, 'max': 20}
        assert res[2] == {'id': 3, 'max': 30}

    def test_groupby_max_float(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10.0, 20.0, 30.0, float('nan'), float('nan'), 60.0]})
        res = t.groupby('id', {'max': MAX('another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'max']
        assert res.column_types() == [int, float]
        assert res[0] == {'id': 1, 'max': 60.0}
        assert res[1] == {'id': 2, 'max': 20.0}
        assert res[2] == {'id': 3, 'max': 30.0}

    def test_groupby_max_str(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', None, None, 'f'],
                    'another': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', {'max': MAX('val')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'max']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'max': 'f'}
        assert res[1] == {'id': 2, 'max': 'b'}
        assert res[2] == {'id': 3, 'max': 'c'}

    def test_groupby_min(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [None, None, 30, 40, 50, 60]})
        res = t.groupby('id', {'min': MIN('another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'min']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': 1, 'min': 40}
        assert res[1] == {'id': 2, 'min': 50}
        assert res[2] == {'id': 3, 'min': 30}

    def test_groupby_min_float(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [None, None, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', {'min': MIN('another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'min']
        assert res.column_types() == [int, float]
        assert res[0] == {'id': 1, 'min': 40.0}
        assert res[1] == {'id': 2, 'min': 50.0}
        assert res[2] == {'id': 3, 'min': 30.0}

    def test_groupby_min_str(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': [None, None, 'c', 'd', 'e', 'f'],
                    'another': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', {'min': MIN('val')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'min']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'min': 'd'}
        assert res[1] == {'id': 2, 'min': 'e'}
        assert res[2] == {'id': 3, 'min': 'c'}

    def test_groupby_mean(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, None, None, 60]})
        res = t.groupby('id', {'mean': MEAN('another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'mean']
        assert res.column_types() == [int, float]
        assert res[0] == {'id': 1, 'mean': 70.0 / 2.0}
        assert res[1] == {'id': 2, 'mean': 20.0}
        assert res[2] == {'id': 3, 'mean': 30.0}

    def test_groupby_variance(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, None, None, 60]})
        res = t.groupby('id', {'variance': VARIANCE('another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'variance']
        assert res.column_types() == [int, float]
        assert almost_equal(res[0]['variance'], 2500.0 / 4.0)
        assert almost_equal(res[1]['variance'], 0.0)
        assert res[2] == {'id': 3, 'variance': 0.0}

    def test_groupby_stdv(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, None, None, 60]})
        res = t.groupby('id', {'stdv': STDV('another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'stdv']
        assert res.column_types() == [int, float]
        assert almost_equal(res[0]['stdv'], math.sqrt(2500.0 / 4.0))
        assert almost_equal(res[1]['stdv'], 0.0)
        assert res[2] == {'id': 3, 'stdv': 0.0}

    def test_groupby_select_one(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, None, None, 60]})
        res = t.groupby('id', {'select_one': SELECT_ONE('another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'select_one']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': 1, 'select_one': 60}
        assert res[1] == {'id': 2, 'select_one': 20}
        assert res[2] == {'id': 3, 'select_one': 30}

    def test_groupby_concat_list(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, None, None, 60]})
        res = t.groupby('id', {'concat': CONCAT('another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'concat']
        assert res.column_types() == [int, list]
        assert res[0] == {'id': 1, 'concat': [10, 60]}
        assert res[1] == {'id': 2, 'concat': [20]}
        assert res[2] == {'id': 3, 'concat': [30]}

    def test_groupby_concat_dict(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', None, None, 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', {'concat': CONCAT('val', 'another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'concat']
        assert res.column_types() == [int, dict]
        assert res[0] == {'id': 1, 'concat': {'a': 10, 'f': 60}}
        assert res[1] == {'id': 2, 'concat': {'b': 20}}
        assert res[2] == {'id': 3, 'concat': {'c': 30}}


# noinspection PyClassHasNoInit
class TestXFrameGroupbyAggregatorsEmpty:
    """
    Tests XFrame groupby aggregators with missing values
    """

    # noinspection PyTypeChecker
    def test_groupby_count(self):
        t = XFrame({'id': [1, 2, None, None, None, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', COUNT)
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'count']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': None, 'count': 3}
        assert res[1] == {'id': 1, 'count': 2}
        assert res[2] == {'id': 2, 'count': 1}

    def test_groupby_sum(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, None, None, None, 60]})
        res = t.groupby('id', SUM('another'))
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'sum']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': 1, 'sum': 70}
        assert res[1] == {'id': 2, 'sum': 20}
        assert res[2] == {'id': 3, 'sum': 0}

    def test_groupby_argmax(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, None, 40, None, None]})
        res = t.groupby('id', {'argmax': ARGMAX('another', 'val')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'argmax']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'argmax': 'd'}
        assert res[1] == {'id': 2, 'argmax': 'b'}
        assert res[2] == {'id': 3, 'argmax': None}

    def test_groupby_argmin(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, None, None, 40, 50, 60]})
        res = t.groupby('id', {'argmin': ARGMIN('another', 'val')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'argmin']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'argmin': 'a'}
        assert res[1] == {'id': 2, 'argmin': 'e'}
        assert res[2] == {'id': 3, 'argmin': None}

    def test_groupby_max(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, None, None, None, 60]})
        res = t.groupby('id', {'max': MAX('another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'max']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': 1, 'max': 60}
        assert res[1] == {'id': 2, 'max': 20}
        assert res[2] == {'id': 3, 'max': None}

    def test_groupby_max_float(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10.0, 20.0, None, float('nan'), float('nan'), 60.0]})
        res = t.groupby('id', {'max': MAX('another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'max']
        assert res.column_types() == [int, float]
        assert res[0] == {'id': 1, 'max': 60.0}
        assert res[1] == {'id': 2, 'max': 20.0}
        assert res[2] == {'id': 3, 'max': None}

    def test_groupby_max_str(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', None, None, None, 'f'],
                    'another': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', {'max': MAX('val')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'max']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'max': 'f'}
        assert res[1] == {'id': 2, 'max': 'b'}
        assert res[2] == {'id': 3, 'max': None}

    def test_groupby_min(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [None, None, None, 40, 50, 60]})
        res = t.groupby('id', {'min': MIN('another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'min']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': 1, 'min': 40}
        assert res[1] == {'id': 2, 'min': 50}
        assert res[2] == {'id': 3, 'min': None}

    def test_groupby_min_float(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [None, None, None, 40.0, 50.0, 60.0]})
        res = t.groupby('id', {'min': MIN('another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'min']
        assert res.column_types() == [int, float]
        assert res[0] == {'id': 1, 'min': 40.0}
        assert res[1] == {'id': 2, 'min': 50.0}
        assert res[2] == {'id': 3, 'min': None}

    def test_groupby_min_str(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': [None, None, None, 'd', 'e', 'f'],
                    'another': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', {'min': MIN('val')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'min']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'min': 'd'}
        assert res[1] == {'id': 2, 'min': 'e'}
        assert res[2] == {'id': 3, 'min': None}

    def test_groupby_mean(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, None, None, None, 60]})
        res = t.groupby('id', {'mean': MEAN('another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'mean']
        assert res.column_types() == [int, float]
        assert res[0] == {'id': 1, 'mean': 70.0 / 2.0}
        assert res[1] == {'id': 2, 'mean': 20.0}
        assert res[2] == {'id': 3, 'mean': None}

    def test_groupby_variance(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, None, None, None, 60]})
        res = t.groupby('id', {'variance': VARIANCE('another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'variance']
        assert res.column_types() == [int, float]
        assert almost_equal(res[0]['variance'], 2500.0 / 4.0)
        assert almost_equal(res[1]['variance'], 0.0)
        assert res[2] == {'id': 3, 'variance': None}

    def test_groupby_stdv(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, None, None, None, 60]})
        res = t.groupby('id', {'stdv': STDV('another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'stdv']
        assert res.column_types() == [int, float]
        assert almost_equal(res[0]['stdv'], math.sqrt(2500.0 / 4.0))
        assert almost_equal(res[1]['stdv'], math.sqrt(0.0))
        assert res[2] == {'id': 3, 'stdv': None}

    def test_groupby_select_one(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, None, None, None, 60]})
        res = t.groupby('id', {'select_one': SELECT_ONE('another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'select_one']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': 1, 'select_one': 60}
        assert res[1] == {'id': 2, 'select_one': 20}
        assert res[2] == {'id': 3, 'select_one': None}

    def test_groupby_concat_list(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, None, None, None, 60]})
        res = t.groupby('id', {'concat': CONCAT('another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'concat']
        assert res.column_types() == [int, list]
        assert res[0] == {'id': 1, 'concat': [10, 60]}
        assert res[1] == {'id': 2, 'concat': [20]}
        assert res[2] == {'id': 3, 'concat': []}

    def test_groupby_concat_dict(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', None, None, None, 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', {'concat': CONCAT('val', 'another')})
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'concat']
        assert res.column_types() == [int, dict]
        assert res[0] == {'id': 1, 'concat': {'a': 10, 'f': 60}}
        assert res[1] == {'id': 2, 'concat': {'b': 20}}
        assert res[2] == {'id': 3, 'concat': {}}


# noinspection PyClassHasNoInit
class TestXFrameJoin:
    """
    Tests XFrame join
    """

    def test_join(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 3], 'doubled': ['aa', 'bb', 'cc']})
        res = t1.join(t2).sort('id').head()
        assert len(res) == 3
        assert res.column_names() == ['id', 'val', 'doubled']
        assert res.column_types() == [int, str, str]
        assert res[0] == {'id': 1, 'val': 'a', 'doubled': 'aa'}
        assert res[1] == {'id': 2, 'val': 'b', 'doubled': 'bb'}
        assert res[2] == {'id': 3, 'val': 'c', 'doubled': 'cc'}

    def test_join_rename(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 3], 'val': ['aa', 'bb', 'cc']})
        res = t1.join(t2, on='id').sort('id').head()
        assert len(res) == 3
        assert res.column_names() == ['id', 'val', 'val.1']
        assert res.column_types() == [int, str, str]
        assert res[0] == {'id': 1, 'val': 'a', 'val.1': 'aa'}
        assert res[1] == {'id': 2, 'val': 'b', 'val.1': 'bb'}
        assert res[2] == {'id': 3, 'val': 'c', 'val.1': 'cc'}

    def test_join_compound_key(self):
        t1 = XFrame({'id1': [1, 2, 3], 'id2': [10, 20, 30], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id1': [1, 2, 3], 'id2': [10, 20, 30], 'doubled': ['aa', 'bb', 'cc']})
        res = t1.join(t2).sort('id1').head()
        assert len(res) == 3
        assert res.column_names() == ['id1', 'id2', 'val', 'doubled']
        assert res.column_types() == [int, int, str, str]
        assert res[0] == {'id1': 1, 'id2': 10, 'val': 'a', 'doubled': 'aa'}
        assert res[1] == {'id1': 2, 'id2': 20, 'val': 'b', 'doubled': 'bb'}
        assert res[2] == {'id1': 3, 'id2': 30, 'val': 'c', 'doubled': 'cc'}

    def test_join_dict_key(self):
        t1 = XFrame({'id1': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id2': [1, 2, 3], 'doubled': ['aa', 'bb', 'cc']})
        res = t1.join(t2, on={'id1': 'id2'}).sort('id1').head()
        assert len(res) == 3
        assert res.column_names() == ['id1', 'val', 'doubled']
        assert res.column_types() == [int, str, str]
        assert res[0] == {'id1': 1, 'val': 'a', 'doubled': 'aa'}
        assert res[1] == {'id1': 2, 'val': 'b', 'doubled': 'bb'}
        assert res[2] == {'id1': 3, 'val': 'c', 'doubled': 'cc'}

    def test_join_partial(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 4], 'doubled': ['aa', 'bb', 'cc']})
        res = t1.join(t2).sort('id').head()
        assert len(res) == 2
        assert res.column_names() == ['id', 'val', 'doubled']
        assert res.column_types() == [int, str, str]
        assert res[0] == {'id': 1, 'val': 'a', 'doubled': 'aa'}
        assert res[1] == {'id': 2, 'val': 'b', 'doubled': 'bb'}

    def test_join_empty(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [4, 5, 6], 'doubled': ['aa', 'bb', 'cc']})
        res = t1.join(t2).head()
        assert len(res) == 0
        assert res.column_names() == ['id', 'val', 'doubled']
        assert res.column_types() == [int, str, str]

    def test_join_on_val(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [10, 20, 30], 'val': ['a', 'b', 'c']})
        res = t1.join(t2, on='val').sort('id').head()
        assert len(res) == 3
        assert res.column_names() == ['id', 'val', 'id.1']
        assert res.column_types() == [int, str, int]
        assert res[0] == {'id': 1, 'val': 'a', 'id.1': 10}
        assert res[1] == {'id': 2, 'val': 'b', 'id.1': 20}
        assert res[2] == {'id': 3, 'val': 'c', 'id.1': 30}

    def test_join_inner(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 4], 'doubled': ['aa', 'bb', 'cc']})
        res = t1.join(t2, how='inner').sort('id').head()
        assert len(res) == 2
        assert res.column_names() == ['id', 'val', 'doubled']
        assert res.column_types() == [int, str, str]
        assert res[0] == {'id': 1, 'val': 'a', 'doubled': 'aa'}
        assert res[1] == {'id': 2, 'val': 'b', 'doubled': 'bb'}

    def test_join_left(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 4], 'doubled': ['aa', 'bb', 'cc']})
        res = t1.join(t2, how='left').sort('id').head()
        assert len(res) == 3
        assert res.column_names() == ['id', 'val', 'doubled']
        assert res.column_types() == [int, str, str]
        assert res[0] == {'id': 1, 'val': 'a', 'doubled': 'aa'}
        assert res[1] == {'id': 2, 'val': 'b', 'doubled': 'bb'}
        assert res[2] == {'id': 3, 'val': 'c', 'doubled': None}

    def test_join_right(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 4], 'doubled': ['aa', 'bb', 'dd']})
        res = t1.join(t2, how='right').sort('id').head()
        assert len(res) == 3
        assert res.column_names() == ['id', 'val', 'doubled']
        assert res.column_types() == [int, str, str]
        assert res[0] == {'id': 1, 'val': 'a', 'doubled': 'aa'}
        assert res[1] == {'id': 2, 'val': 'b', 'doubled': 'bb'}
        assert res[2] == {'id': 4, 'val': None, 'doubled': 'dd'}

    def test_join_full(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 4], 'doubled': ['aa', 'bb', 'dd']})
        res = t1.join(t2, how='full').sort('id').head()
        assert len(res) == 4
        assert res.column_names() == ['id', 'val', 'doubled']
        assert res.column_types() == [int, str, str]
        assert res[0] == {'id': 1, 'val': 'a', 'doubled': 'aa'}
        assert res[1] == {'id': 2, 'val': 'b', 'doubled': 'bb'}
        assert res[2] == {'id': 3, 'val': 'c', 'doubled': None}
        assert res[3] == {'id': 4, 'val': None, 'doubled': 'dd'}

    def test_join_cartesian(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [10, 20, 30], 'doubled': ['aa', 'bb', 'cc']})
        res = t1.join(t2, how='cartesian').sort(['id', 'id.1']).head()
        assert len(res) == 9
        assert res.column_names() == ['id', 'val', 'doubled', 'id.1']
        assert res.column_types() == [int, str, str, int]
        assert res[0] == {'id': 1, 'val': 'a', 'doubled': 'aa', 'id.1': 10}
        assert res[1] == {'id': 1, 'val': 'a', 'doubled': 'bb', 'id.1': 20}
        assert res[3] == {'id': 2, 'val': 'b', 'doubled': 'aa', 'id.1': 10}
        assert res[8] == {'id': 3, 'val': 'c', 'doubled': 'cc', 'id.1': 30}

    def test_join_bad_how(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 3], 'doubled': ['aa', 'bb', 'cc']})
        with pytest.raises(ValueError) as exception_info:
            t1.join(t2, how='xx')
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Invalid join type.'

    # noinspection PyTypeChecker
    def test_join_bad_right(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            t1.join([1, 2, 3])
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Can only join two XFrames.'

    def test_join_bad_on_list(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 3], 'doubled': ['aa', 'bb', 'cc']})
        with pytest.raises(TypeError) as exception_info:
            t1.join(t2, on=['id', 1])
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Join keys must each be a str.'

    # noinspection PyTypeChecker
    def test_join_bad_on_type(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 3], 'doubled': ['aa', 'bb', 'cc']})
        with pytest.raises(TypeError) as exception_info:
            t1.join(t2, on=1)
        exception_message = exception_info.value.args[0]
        assert exception_message == "Must pass a 'str', 'list', or 'dict' of join keys."

    def test_join_bad_on_col_name(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 3], 'doubled': ['aa', 'bb', 'cc']})
        with pytest.raises(ValueError) as exception_info:
            t1.join(t2, on='xx')
        exception_message = exception_info.value.args[0]
        assert exception_message == "Key 'xx' is not a column name."


# noinspection PyClassHasNoInit
class TestXFrameSplitDatetime:
    """
    Tests XFrame split_datetime
    """

    def test_split_datetime(self):
        t = XFrame({'id': [1, 2, 3], 'val': [datetime(2011, 1, 1),
                                             datetime(2012, 2, 2),
                                             datetime(2013, 3, 3)]})
        res = t.split_datetime('val')
        assert len(res) == 3
        assert res.column_names() == ['id',
                                      'val.year', 'val.month', 'val.day',
                                      'val.hour', 'val.minute', 'val.second']
        assert res.column_types() == [int, int, int, int, int, int, int]
        assert list(res['id']) == [1, 2, 3]
        assert list(res['val.year']) == [2011, 2012, 2013]
        assert list(res['val.month']) == [1, 2, 3]
        assert list(res['val.day']) == [1, 2, 3]
        assert list(res['val.hour']) == [0, 0, 0]
        assert list(res['val.minute']) == [0, 0, 0]
        assert list(res['val.second']) == [0, 0, 0]

    # noinspection PyTypeChecker
    def test_split_datetime_col_conflict(self):
        t = XFrame({'id': [1, 2, 3],
                    'val.year': ['x', 'y', 'z'],
                    'val': [datetime(2011, 1, 1),
                            datetime(2012, 2, 2),
                            datetime(2013, 3, 3)]})
        res = t.split_datetime('val', limit='year')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val.year', 'val.year.1']
        assert res.column_types() == [int, str, int]
        assert list(res['id']) == [1, 2, 3]
        assert list(res['val.year']) == ['x', 'y', 'z']
        assert list(res['val.year.1']) == [2011, 2012, 2013]

    def test_split_datetime_bad_col(self):
        t = XFrame({'id': [1, 2, 3], 'val': [datetime(2011, 1, 1),
                                             datetime(2011, 2, 2),
                                             datetime(2011, 3, 3)]})
        with pytest.raises(KeyError) as exception_info:
            t.split_datetime('xx')
        exception_message = exception_info.value.args[0]
        assert exception_message == "Column 'xx' does not exist in current XFrame."


# noinspection PyClassHasNoInit
class TestXFrameFilterby:
    """
    Tests XFrame filterby
    """

    def test_filterby_int_id(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.filterby(1, 'id').sort('id')
        assert len(res) == 1
        assert res[0] == {'id': 1, 'val': 'a'}

    def test_filterby_str_id(self):
        t = XFrame({'id': ['qaz', 'wsx', 'edc', 'rfv'], 'val': ['a', 'b', 'c', 'd']})
        res = t.filterby('qaz', 'id').sort('id')
        assert len(res) == 1
        assert res[0] == {'id': 'qaz', 'val': 'a'}

    def test_filterby_object_id(self):
        t = XFrame({'id': [datetime(2016, 2, 1, 0, 0),
                           datetime(2016, 2, 2, 0, 0),
                           datetime(2016, 2, 3, 0, 0),
                           datetime(2016, 2, 4, 0, 0)],
                    'val': ['a', 'b', 'c', 'd']})
        res = t.filterby(datetime(2016, 2, 1, 0, 0), 'id').sort('id')
        assert len(res) == 1
        assert res[0] == {'id': datetime(2016, 2, 1, 0, 0), 'val': 'a'}

    def test_filterby_list_id(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.filterby([1, 3], 'id').sort('id')
        assert len(res) == 2
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 3, 'val': 'c'}

    def test_filterby_tuple_id(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.filterby((1, 3), 'id').sort('id')
        assert len(res) == 2
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 3, 'val': 'c'}

    def test_filterby_iterable_id(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.filterby(range(3), 'id').sort('id')
        assert len(res) == 2
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}

    def test_filterby_set_id(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.filterby({1, 3}, 'id').sort('id')
        assert len(res) == 2
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 3, 'val': 'c'}

    def test_filterby_list_val(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.filterby(['a', 'b'], 'val').sort('id')
        assert len(res) == 2
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert list(res['id']) == [1, 2]

    def test_filterby_xarray(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        a = XArray([1, 3])
        res = t.filterby(a, 'id').sort('id')
        assert len(res) == 2
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 3, 'val': 'c'}
        assert list(res['id']) == [1, 3]

    def test_filterby_function(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.filterby(lambda x: x != 2, 'id').sort('id')
        assert len(res) == 2
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 3, 'val': 'c'}
        assert list(res['id']) == [1, 3]

    def test_filterby_function_exclude(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.filterby(lambda x: x == 2, 'id', exclude=True).sort('id')
        assert len(res) == 2
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 3, 'val': 'c'}
        assert list(res['id']) == [1, 3]

    def test_filterby_function_row(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.filterby(lambda row: row['id'] != 2, None).sort('id')
        assert len(res) == 2
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 3, 'val': 'c'}
        assert list(res['id']) == [1, 3]

    def test_filterby_list_exclude(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.filterby([1, 3], 'id', exclude=True).sort('id')
        assert len(res) == 2
        assert res[0] == {'id': 2, 'val': 'b'}
        assert res[1] == {'id': 4, 'val': 'd'}
        assert list(res['id']) == [2, 4]

    def test_filterby_bad_column_type_list(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(TypeError) as exception_info:
            t.filterby([1, 3], 'val')
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Value type (int) does not match column type (str).'

    def test_filterby_xarray_exclude(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        a = XArray([1, 3])
        res = t.filterby(a, 'id', exclude=True).sort('id')
        assert len(res) == 2
        assert res[0] == {'id': 2, 'val': 'b'}
        assert res[1] == {'id': 4, 'val': 'd'}
        assert list(res['id']) == [2, 4]

    # noinspection PyTypeChecker
    def test_filterby_bad_column_name_type(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(TypeError) as exception_info:
            t.filterby([1, 3], 1)
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Column_name must be a string.'

    def test_filterby_bad_column_name(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(KeyError) as exception_info:
            t.filterby([1, 3], 'xx')
        exception_message = exception_info.value.args[0]
        assert exception_message == "Column 'xx' not in XFrame."

    def test_filterby_bad_column_type_xarray(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        a = XArray([1, 3])
        with pytest.raises(TypeError) as exception_info:
            t.filterby(a, 'val')
        exception_message = exception_info.value.args[0]
        assert exception_message == "Type of given values ('<type 'int'>') does not match " + \
                                    "type of column 'val' ('<type 'str'>') in XFrame."

    def test_filterby_bad_list_empty(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(ValueError) as exception_info:
            t.filterby([], 'id').sort('id')
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Value list is empty.'

    def test_filterby_bad_xarray_empty(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        a = XArray([])
        with pytest.raises(TypeError) as exception_info:
            t.filterby(a, 'val')
        exception_message = exception_info.value.args[0]
        assert exception_message == "Type of given values ('None') does not match " + \
                                    "type of column 'val' ('<type 'str'>') in XFrame."


# noinspection PyClassHasNoInit
class TestXFramePackColumnsList:
    """
    Tests XFrame pack_columns into list
    """

    def test_pack_columns(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='new')
        assert len(res) == 4
        assert res.num_columns() == 1
        assert res.column_names() == ['new']
        assert res.column_types() == [list]
        assert res[0] == {'new': [1, 'a']}
        assert res[1] == {'new': [2, 'b']}

    def test_pack_columns_all(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.pack_columns(new_column_name='new')
        assert len(res) == 4
        assert res.num_columns() == 1
        assert res.column_names() == ['new']
        assert res.column_types() == [list]
        assert res[0] == {'new': [1, 'a']}
        assert res[1] == {'new': [2, 'b']}

    def test_pack_columns_prefix(self):
        t = XFrame({'x.id': [1, 2, 3, 4], 'x.val': ['a', 'b', 'c', 'd'], 'another': [10, 20, 30, 40]})
        res = t.pack_columns(column_prefix='x', new_column_name='new')
        assert len(res) == 4
        assert res.num_columns() == 2
        assert res.column_names() == ['another', 'new']
        assert res.column_types() == [int, list]
        assert res[0] == {'another': 10, 'new': [1, 'a']}
        assert res[1] == {'another': 20, 'new': [2, 'b']}

    def test_pack_columns_rest(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd'], 'another': [10, 20, 30, 40]})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='new')
        assert len(res) == 4
        assert res.num_columns() == 2
        assert res.column_names() == ['another', 'new']
        assert res.column_types() == [int, list]
        assert res[0] == {'another': 10, 'new': [1, 'a']}
        assert res[1] == {'another': 20, 'new': [2, 'b']}

    def test_pack_columns_na(self):
        t = XFrame({'id': [1, 2, None, 4], 'val': ['a', 'b', 'c', None]})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='new', fill_na='x')
        assert len(res) == 4
        assert res.num_columns() == 1
        assert res.column_names() == ['new']
        assert res.column_types() == [list]
        assert res[0] == {'new': [1, 'a']}
        assert res[1] == {'new': [2, 'b']}
        assert res[2] == {'new': ['x', 'c']}
        assert res[3] == {'new': [4, 'x']}

    def test_pack_columns_fill_na(self):
        t = XFrame({'id': [1, 2, None, 4], 'val': ['a', 'b', 'c', None]})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='new', fill_na=99)
        assert len(res) == 4
        assert res.num_columns() == 1
        assert res.column_names() == ['new']
        assert res.column_types() == [list]
        assert res[0] == {'new': [1, 'a']}
        assert res[1] == {'new': [2, 'b']}
        assert res[2] == {'new': [99, 'c']}
        assert res[3] == {'new': [4, 99]}

    def test_pack_columns_def_new_name(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.pack_columns(columns=['id', 'val'])
        assert len(res) == 4
        assert res.num_columns() == 1
        assert res.column_names() == ['X.0']
        assert res.column_types() == [list]
        assert res[0] == {'X.0': [1, 'a']}
        assert res[1] == {'X.0': [2, 'b']}

    def test_pack_columns_prefix_def_new_name(self):
        t = XFrame({'x.id': [1, 2, 3, 4], 'x.val': ['a', 'b', 'c', 'd'], 'another': [10, 20, 30, 40]})
        res = t.pack_columns(column_prefix='x')
        assert len(res) == 4
        assert res.num_columns() == 2
        assert res.column_names() == ['another', 'x']
        assert res.column_types() == [int, list]
        assert res[0] == {'another': 10, 'x': [1, 'a']}
        assert res[1] == {'another': 20, 'x': [2, 'b']}

    # noinspection PyTypeChecker
    def test_pack_columns_bad_col_spec(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(ValueError) as exception_info:
            t.pack_columns(columns='id', column_prefix='val')
        exception_message = exception_info.value.args[0]
        assert exception_message == "'Columns' and 'column_prefix' parameter cannot be given at the same time."

    # noinspection PyTypeChecker
    def test_pack_columns_bad_col_prefix_type(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(TypeError) as exception_info:
            t.pack_columns(column_prefix=1)
        exception_message = exception_info.value.args[0]
        assert exception_message == "'Column_prefix' must be a string. Found 'int': 1."

    def test_pack_columns_bad_col_prefix(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(ValueError) as exception_info:
            t.pack_columns(column_prefix='xx')
        exception_message = exception_info.value.args[0]
        assert exception_message == "There are no column starts with prefix 'xx'."

    def test_pack_columns_bad_cols(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(ValueError) as exception_info:
            t.pack_columns(columns=['xx'])
        exception_message = exception_info.value.args[0]
        assert exception_message == "Current XFrame has no column called 'xx'."

    def test_pack_columns_bad_cols_dup(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(ValueError) as exception_info:
            t.pack_columns(columns=['id', 'id'])
        exception_message = exception_info.value.args[0]
        assert exception_message == 'There are duplicate column names in columns parameter.'

    def test_pack_columns_bad_cols_single(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(ValueError) as exception_info:
            t.pack_columns(columns=['id'])
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Please provide at least two columns to pack.'

    # noinspection PyTypeChecker
    def test_pack_columns_bad_dtype(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(ValueError) as exception_info:
            t.pack_columns(columns=['id', 'val'], dtype=int)
        exception_message = exception_info.value.args[0]
        assert exception_message == "Resulting dtype has to be one of 'dict', 'array.array', 'list', or 'tuple' type."

    # noinspection PyTypeChecker
    def test_pack_columns_bad_new_col_name_type(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(TypeError) as exception_info:
            t.pack_columns(columns=['id', 'val'], new_column_name=1)
        exception_message = exception_info.value.args[0]
        assert exception_message == "'New_column_name' must be a string. Found 'int': 1."

    def test_pack_columns_bad_new_col_name_dup_rest(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd'], 'another': [11, 12, 13, 14]})
        with pytest.raises(KeyError) as exception_info:
            t.pack_columns(columns=['id', 'val'], new_column_name='another')
        exception_message = exception_info.value.args[0]
        assert exception_message == "Current XFrame already contains a column name 'another'."

    def test_pack_columns_good_new_col_name_dup_key(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='id')
        assert res.column_names() == ['id']
        assert res[0] == {'id': [1, 'a']}
        assert res[1] == {'id': [2, 'b']}


class TestXFramePackColumnsTuple:
    """
    Tests XFrame pack_columns into tuple
    """

    def test_pack_columns(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='new', dtype=tuple)
        assert len(res) == 4
        assert res.num_columns() == 1
        assert res.column_names() == ['new']
        assert res.column_types() == [tuple]
        assert res[0] == {'new': (1, 'a')}
        assert res[1] == {'new': (2, 'b')}

# noinspection PyClassHasNoInit
class TestXFramePackColumnsDict:
    """
    Tests XFrame pack_columns into dict
    """

    def test_pack_columns(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='new', dtype=dict)
        assert len(res) == 4
        assert res.num_columns() == 1
        assert res.dtype() == [dict]
        assert res.column_names() == ['new']
        assert res[0] == {'new': {'id': 1, 'val': 'a'}}
        assert res[1] == {'new': {'id': 2, 'val': 'b'}}

    def test_pack_columns_prefix(self):
        t = XFrame({'x.id': [1, 2, 3, 4], 'x.val': ['a', 'b', 'c', 'd'], 'another': [10, 20, 30, 40]})
        res = t.pack_columns(column_prefix='x', dtype=dict)
        assert len(res) == 4
        assert res.num_columns() == 2
        assert res.dtype() == [int, dict]
        assert res.column_names() == ['another', 'x']
        assert res[0] == {'another': 10, 'x': {'id': 1, 'val': 'a'}}
        assert res[1] == {'another': 20, 'x': {'id': 2, 'val': 'b'}}

    def test_pack_columns_prefix_named(self):
        t = XFrame({'x.id': [1, 2, 3, 4], 'x.val': ['a', 'b', 'c', 'd'], 'another': [10, 20, 30, 40]})
        res = t.pack_columns(column_prefix='x', dtype=dict, new_column_name='new')
        assert len(res) == 4
        assert res.num_columns() == 2
        assert res.dtype() == [int, dict]
        assert res.column_names() == ['another', 'new']
        assert res[0] == {'another': 10, 'new': {'id': 1, 'val': 'a'}}
        assert res[1] == {'another': 20, 'new': {'id': 2, 'val': 'b'}}

    def test_pack_columns_prefix_no_remove(self):
        t = XFrame({'x.id': [1, 2, 3, 4], 'x.val': ['a', 'b', 'c', 'd'], 'another': [10, 20, 30, 40]})
        res = t.pack_columns(column_prefix='x', dtype=dict, remove_prefix=False)
        assert len(res) == 4
        assert res.num_columns() == 2
        assert res.dtype() == [int, dict]
        assert res.column_names() == ['another', 'x']
        assert res[0] == {'another': 10, 'x': {'x.id': 1, 'x.val': 'a'}}
        assert res[1] == {'another': 20, 'x': {'x.id': 2, 'x.val': 'b'}}

    def test_pack_columns_drop_missing(self):
        t = XFrame({'id': [1, 2, None, 4], 'val': ['a', 'b', 'c', None]})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='new', dtype=dict)
        assert len(res) == 4
        assert res.num_columns() == 1
        assert res.dtype() == [dict]
        assert res.column_names() == ['new']
        assert res[0] == {'new': {'id': 1, 'val': 'a'}}
        assert res[1] == {'new': {'id': 2, 'val': 'b'}}
        assert res[2] == {'new': {'val': 'c'}}
        assert res[3] == {'new': {'id': 4}}

    def test_pack_columns_fill_na(self):
        t = XFrame({'id': [1, 2, None, 4], 'val': ['a', 'b', 'c', None]})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='new', dtype=dict, fill_na=99)
        assert len(res) == 4
        assert res.num_columns() == 1
        assert res.dtype() == [dict]
        assert res.column_names() == ['new']
        assert res[0] == {'new': {'id': 1, 'val': 'a'}}
        assert res[1] == {'new': {'id': 2, 'val': 'b'}}
        assert res[2] == {'new': {'id': 99, 'val': 'c'}}
        assert res[3] == {'new': {'id': 4, 'val': 99}}


# noinspection PyClassHasNoInit
class TestXFramePackColumnsArray:
    """
    Tests XFrame pack_columns into array
    """

    def test_pack_columns(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [10, 20, 30, 40]})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='new', dtype=array.array)
        assert len(res) == 4
        assert res.num_columns() == 1
        assert res.dtype() == [array.array]
        assert res.column_names() == ['new']
        assert res[0] == {'new': array.array('d', [1.0, 10.0])}
        assert res[1] == {'new': array.array('d', [2.0, 20.0])}

    def test_pack_columns_bad_fill_na_not_numeric(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [10, 20, 30, 40]})
        with pytest.raises(ValueError) as exception_info:
            t.pack_columns(columns=['id', 'val'], new_column_name='new', dtype=array.array, fill_na='a')
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Fill_na value for array needs to be numeric type.'

    def test_pack_columns_bad_not_numeric(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(TypeError) as exception_info:
            t.pack_columns(columns=['id', 'val'], new_column_name='new', dtype=array.array)
        exception_message = exception_info.value.args[0]
        assert exception_message == "Column 'val' type is not numeric, cannot pack into array type."

        # TODO list


# noinspection PyClassHasNoInit
class TestXFrameUnpackList:
    """
    Tests XFrame unpack where the unpacked column contains a list
    """

    def test_unpack(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [[10, 'a'], [20, 'b'], [30, 'c'], [40, 'd']]})
        res = t.unpack('val')
        assert len(res) == 4
        assert res.column_names() == ['id', 'val.0', 'val.1']
        assert res.column_types() == [int, int, str]
        assert res[0] == {'id': 1, 'val.0': 10, 'val.1': 'a'}
        assert res[1] == {'id': 2, 'val.0': 20, 'val.1': 'b'}

    def test_unpack_prefix(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [[10, 'a'], [20, 'b'], [30, 'c'], [40, 'd']]})
        res = t.unpack('val', column_name_prefix='x')
        assert len(res) == 4
        assert res.column_names() == ['id', 'x.0', 'x.1']
        assert res.column_types() == [int, int, str]
        assert res[0] == {'id': 1, 'x.0': 10, 'x.1': 'a'}
        assert res[1] == {'id': 2, 'x.0': 20, 'x.1': 'b'}

    def test_unpack_types(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [[10, 'a'], [20, 'b'], [30, 'c'], [40, 'd']]})
        res = t.unpack('val', column_types=[str, str])
        assert len(res) == 4
        assert res.column_names() == ['id', 'val.0', 'val.1']
        assert res.column_types() == [int, str, str]
        assert res[0] == {'id': 1, 'val.0': '10', 'val.1': 'a'}
        assert res[1] == {'id': 2, 'val.0': '20', 'val.1': 'b'}

    def test_unpack_na_value(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [[10, 'a'], [20, 'b'], [None, 'c'], [40, None]]})
        res = t.unpack('val', na_value=99)
        assert len(res) == 4
        assert res.column_names() == ['id', 'val.0', 'val.1']
        assert res.column_types() == [int, int, str]
        assert res[0] == {'id': 1, 'val.0': 10, 'val.1': 'a'}
        assert res[1] == {'id': 2, 'val.0': 20, 'val.1': 'b'}
        assert res[2] == {'id': 3, 'val.0': 99, 'val.1': 'c'}
        assert res[3] == {'id': 4, 'val.0': 40, 'val.1': '99'}

    def test_unpack_limit(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [[10, 'a'], [20, 'b'], [30, 'c'], [40, 'd']]})
        res = t.unpack('val', limit=[1])
        assert len(res) == 4
        assert res.column_names() == ['id', 'val.1']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val.1': 'a'}
        assert res[1] == {'id': 2, 'val.1': 'b'}


# noinspection PyClassHasNoInit
class TestXFrameUnpackDict:
    """
    Tests XFrame unpack where the unpacked column contains a dict
    """

    def test_unpack(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [{'a': 1}, {'b': 2}, {'c': 3}, {'d': 4}]})
        res = t.unpack('val')
        assert len(res) == 4
        assert res.column_names() == ['id', 'val.a', 'val.c', 'val.b', 'val.d']
        assert res.column_types() == [int, int, int, int, int]
        assert res[0] == {'id': 1, 'val.a': 1, 'val.c': None, 'val.b': None, 'val.d': None}
        assert res[1] == {'id': 2, 'val.a': None, 'val.c': None, 'val.b': 2, 'val.d': None}

    def test_unpack_mult(self):
        t = XFrame({'id': [1, 2, 3], 'val': [{'a': 1}, {'b': 2}, {'a': 1, 'b': 2}]})
        res = t.unpack('val')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val.a', 'val.b']
        assert res.column_types() == [int, int, int]
        assert res[0] == {'id': 1, 'val.a': 1, 'val.b': None}
        assert res[1] == {'id': 2, 'val.a': None, 'val.b': 2}
        assert res[2] == {'id': 3, 'val.a': 1, 'val.b': 2}

    def test_unpack_prefix(self):
        t = XFrame({'id': [1, 2, 3], 'val': [{'a': 1}, {'b': 2}, {'a': 1, 'b': 2}]})
        res = t.unpack('val', column_name_prefix='x')
        assert len(res) == 3
        assert res.column_names() == ['id', 'x.a', 'x.b']
        assert res.column_types() == [int, int, int]
        assert res[0] == {'id': 1, 'x.a': 1, 'x.b': None}
        assert res[1] == {'id': 2, 'x.a': None, 'x.b': 2}
        assert res[2] == {'id': 3, 'x.a': 1, 'x.b': 2}

    def test_unpack_types(self):
        t = XFrame({'id': [1, 2, 3], 'val': [{'a': 1}, {'b': 2}, {'a': 1, 'b': 2}]})
        res = t.unpack('val', column_types=[str, str], limit=['a', 'b'])
        assert len(res) == 3
        assert res.column_names() == ['id', 'val.a', 'val.b']
        assert res.column_types() == [int, str, str]
        assert res[0] == {'id': 1, 'val.a': '1', 'val.b': None}
        assert res[1] == {'id': 2, 'val.a': None, 'val.b': '2'}
        assert res[2] == {'id': 3, 'val.a': '1', 'val.b': '2'}

    def test_unpack_na_value(self):
        t = XFrame({'id': [1, 2, 3], 'val': [{'a': 1}, {'b': 2}, {'a': 1, 'b': 2}]})
        res = t.unpack('val', na_value=99)
        assert len(res) == 3
        assert res.column_names() == ['id', 'val.a', 'val.b']
        assert res.column_types() == [int, int, int]
        assert res[0] == {'id': 1, 'val.a': 1, 'val.b': 99}
        assert res[1] == {'id': 2, 'val.a': 99, 'val.b': 2}
        assert res[2] == {'id': 3, 'val.a': 1, 'val.b': 2}

    def test_unpack_limit(self):
        t = XFrame({'id': [1, 2, 3], 'val': [{'a': 1}, {'b': 2}, {'a': 1, 'b': 2}]})
        res = t.unpack('val', limit=['b'])
        assert len(res) == 3
        assert res.column_names() == ['id', 'val.b']
        assert res.column_types() == [int, int]
        assert res[0] == {'id': 1, 'val.b': None}
        assert res[1] == {'id': 2, 'val.b': 2}
        assert res[2] == {'id': 3, 'val.b': 2}

    def test_unpack_bad_types_no_limit(self):
        t = XFrame({'id': [1, 2, 3], 'val': [{'a': 1}, {'b': 2}, {'a': 1, 'b': 2}]})
        with pytest.raises(ValueError) as exception_info:
            t.unpack('val', column_types=[str, str])
        exception_message = exception_info.value.args[0]
        assert exception_message == "If 'column_types' is given, 'limit' has to be provided to unpack dict type."


# TODO unpack array

# noinspection PyClassHasNoInit
class TestXFrameStackList:
    """
    Tests XFrame stack where column is a list
    """

    def test_stack_list(self):
        t = XFrame({'id': [1, 2, 3], 'val': [['a1', 'b1', 'c1'], ['a2', 'b2'], ['a3', 'b3', 'c3', None]]})
        res = t.stack('val')
        assert len(res) == 9
        assert res.column_names() == ['id', 'X']
        assert res[0] == {'id': 1, 'X': 'a1'}
        assert res[8] == {'id': 3, 'X': None}

    def test_stack_list_drop_na(self):
        t = XFrame({'id': [1, 2, 3], 'val': [['a1', 'b1', 'c1'], ['a2', 'b2'], ['a3', 'b3', 'c3', None]]})
        res = t.stack('val', drop_na=True)
        assert len(res) == 8
        assert res.column_names() == ['id', 'X']
        assert res[0] == {'id': 1, 'X': 'a1'}
        assert res[7] == {'id': 3, 'X': 'c3'}

    def test_stack_name(self):
        t = XFrame({'id': [1, 2, 3], 'val': [['a1', 'b1', 'c1'], ['a2', 'b2'], ['a3', 'b3', 'c3', None]]})
        res = t.stack('val', new_column_name='flat_val')
        assert len(res) == 9
        assert res.column_names() == ['id', 'flat_val']

    def test_stack_bad_col_name(self):
        t = XFrame({'id': [1, 2, 3], 'val': [['a1', 'b1', 'c1'], ['a2', 'b2'], ['a3', 'b3', 'c3', None]]})
        with pytest.raises(ValueError) as exception_info:
            t.stack('xx')
        exception_message = exception_info.value.args[0]
        assert exception_message == "Cannot find column 'xx' in the XFrame."

    def test_stack_bad_col_value(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            t.stack('val')
        exception_message = exception_info.value.args[0]
        assert exception_message == "Stack is only supported for column of 'dict', 'list', or 'array' type."

    # noinspection PyTypeChecker
    def test_stack_bad_new_col_name_type(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            t.stack('val', new_column_name=1)
        exception_message = exception_info.value.args[0]
        assert exception_message == "Stack is only supported for column of 'dict', 'list', or 'array' type."

    def test_stack_new_col_name_dup_ok(self):
        t = XFrame({'id': [1, 2, 3], 'val': [['a1', 'b1', 'c1'], ['a2', 'b2'], ['a3', 'b3', 'c3', None]]})
        res = t.stack('val', new_column_name='val')
        assert res.column_names() == ['id', 'val']

    def test_stack_bad_new_col_name_dup(self):
        t = XFrame({'id': [1, 2, 3], 'val': [['a1', 'b1', 'c1'], ['a2', 'b2'], ['a3', 'b3', 'c3', None]]})
        with pytest.raises(ValueError) as exception_info:
            t.stack('val', new_column_name='id')
        exception_message = exception_info.value.args[0]
        assert exception_message == "Column with name 'id' already exists, pick a new column name."

    def test_stack_bad_no_data(self):
        t = XFrame({'id': [1, 2, 3], 'val': [['a1', 'b1', 'c1'], ['a2', 'b2'], ['a3', 'b3', 'c3', None]]})
        t = t.head(0)
        with pytest.raises(ValueError) as exception_info:
            t.stack('val', new_column_name='val')
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Cannot infer column type because there are not enough rows.'


# noinspection PyClassHasNoInit
class TestXFrameStackDict:
    """
    Tests XFrame stack where column is a dict
    """

    def test_stack_dict(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [{'a': 3, 'b': 2}, {'a': 2, 'c': 2}, {'c': 1, 'd': 3}, {}]})
        res = t.stack('val')
        assert len(res) == 7
        assert res.column_names() == ['id', 'K', 'V']
        assert res[0] == {'id': 1, 'K': 'a', 'V': 3}
        assert res[6] == {'id': 4, 'K': None, 'V': None}

    def test_stack_names(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [{'a': 3, 'b': 2}, {'a': 2, 'c': 2}, {'c': 1, 'd': 3}, {}]})
        res = t.stack('val', ['new_k', 'new_v'])
        assert len(res) == 7
        assert res.column_names() == ['id', 'new_k', 'new_v']
        assert res[0] == {'id': 1, 'new_k': 'a', 'new_v': 3}
        assert res[6] == {'id': 4, 'new_k': None, 'new_v': None}

    def test_stack_dropna(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [{'a': 3, 'b': 2}, {'a': 2, 'c': 2}, {'c': 1, 'd': 3}, {}]})
        res = t.stack('val', drop_na=True)
        assert len(res) == 6
        assert res.column_names() == ['id', 'K', 'V']
        assert res[0] == {'id': 1, 'K': 'a', 'V': 3}
        assert res[5] == {'id': 3, 'K': 'd', 'V': 3}

    def test_stack_bad_col_name(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [{'a': 3, 'b': 2}, {'a': 2, 'c': 2}, {'c': 1, 'd': 3}, {}]})
        with pytest.raises(ValueError) as exception_info:
            t.stack('xx')
        exception_message = exception_info.value.args[0]
        assert exception_message == "Cannot find column 'xx' in the XFrame."

    # noinspection PyTypeChecker
    def test_stack_bad_new_col_name_type(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [{'a': 3, 'b': 2}, {'a': 2, 'c': 2}, {'c': 1, 'd': 3}, {}]})
        with pytest.raises(TypeError) as exception_info:
            t.stack('val', new_column_name=1)
        exception_message = exception_info.value.args[0]
        assert exception_message == "'New_column_name' has to be a 'list' to stack 'dict' type. Found 'int': 1"

    def test_stack_bad_new_col_name_len(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [{'a': 3, 'b': 2}, {'a': 2, 'c': 2}, {'c': 1, 'd': 3}, {}]})
        with pytest.raises(TypeError) as exception_info:
            t.stack('val', new_column_name=['a'])
        exception_message = exception_info.value.args[0]
        assert exception_message == "'New_column_name' must have length of two."

    def test_stack_bad_new_col_name_dup(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [{'a': 3, 'b': 2}, {'a': 2, 'c': 2}, {'c': 1, 'd': 3}, {}]})
        with pytest.raises(ValueError) as exception_info:
            t.stack('val', new_column_name=['id', 'xx'])
        exception_message = exception_info.value.args[0]
        assert exception_message == "Column with name 'id' already exists, pick a new column name."

    def test_stack_bad_no_data(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [{'a': 3, 'b': 2}, {'a': 2, 'c': 2}, {'c': 1, 'd': 3}, {}]})
        t = t.head(0)
        with pytest.raises(ValueError) as exception_info:
            t.stack('val', new_column_name=['k', 'v'])
        exception_message = exception_info.value.args[0]
        assert exception_message == 'Cannot infer column type because there are not enough rows.'


# noinspection PyClassHasNoInit
class TestXFrameUnstackList:
    """
    Tests XFrame unstack where unstack column is list
    """

    def test_unstack(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1, 3], 'val': ['a1', 'b1', 'c1', 'a2', 'b2', 'a3', 'c3']})
        res = t.unstack('val')
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'unstack']
        assert res.column_types() == [int, list]
        assert res[0] == {'id': 1, 'unstack': ['a1', 'a2', 'a3']}
        assert res[1] == {'id': 2, 'unstack': ['b1', 'b2']}
        assert res[2] == {'id': 3, 'unstack': ['c1', 'c3']}

    def test_unstack_name(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1, 3], 'val': ['a1', 'b1', 'c1', 'a2', 'b2', 'a3', 'c3']})
        res = t.unstack('val', new_column_name='vals')
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'vals']
        assert res.column_types() == [int, list]
        assert res[0] == {'id': 1, 'vals': ['a1', 'a2', 'a3']}
        assert res[1] == {'id': 2, 'vals': ['b1', 'b2']}
        assert res[2] == {'id': 3, 'vals': ['c1', 'c3']}


# noinspection PyClassHasNoInit
class TestXFrameUnstackDict:
    """
    Tests XFrame unstack where unstack column is dict
    """

    # untested -- test after groupby
    def test_unstack(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1, 3],
                    'key': ['ka1', 'kb1', 'kc1', 'ka2', 'kb2', 'ka3', 'kc3'],
                    'val': ['a1', 'b1', 'c1', 'a2', 'b2', 'a3', 'c3']})
        res = t.unstack(['key', 'val'])
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'unstack']
        assert res.column_types() == [int, dict]
        assert res[0] == {'id': 1, 'unstack': {'ka1': 'a1', 'ka2': 'a2', 'ka3': 'a3'}}
        assert res[1] == {'id': 2, 'unstack': {'kb1': 'b1', 'kb2': 'b2'}}
        assert res[2] == {'id': 3, 'unstack': {'kc1': 'c1', 'kc3': 'c3'}}

    def test_unstack_name(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1, 3],
                    'key': ['ka1', 'kb1', 'kc1', 'ka2', 'kb2', 'ka3', 'kc3'],
                    'val': ['a1', 'b1', 'c1', 'a2', 'b2', 'a3', 'c3']})
        res = t.unstack(['key', 'val'], new_column_name='vals')
        res = res.topk('id', reverse=True)
        assert len(res) == 3
        assert res.column_names() == ['id', 'vals']
        assert res.column_types() == [int, dict]
        assert res[0] == {'id': 1, 'vals': {'ka1': 'a1', 'ka2': 'a2', 'ka3': 'a3'}}
        assert res[1] == {'id': 2, 'vals': {'kb1': 'b1', 'kb2': 'b2'}}
        assert res[2] == {'id': 3, 'vals': {'kc1': 'c1', 'kc3': 'c3'}}


# noinspection PyClassHasNoInit
class TestXFrameUnique:
    """
    Tests XFrame unique
    """

    def test_unique_noop(self):
        t = XFrame({'id': [3, 2, 1], 'val': ['c', 'b', 'a']})
        res = t.unique()
        assert len(res) == 3
        assert sorted(list(res['id'])) == [1, 2, 3]
        assert sorted(list(res['val'])) == ['a', 'b', 'c']

    def test_unique(self):
        t = XFrame({'id': [3, 2, 1, 1], 'val': ['c', 'b', 'a', 'a']})
        res = t.unique()
        assert len(res) == 3
        assert sorted(list(res['id'])) == [1, 2, 3]
        assert sorted(list(res['val'])) == ['a', 'b', 'c']

    def test_unique_part(self):
        t = XFrame({'id': [3, 2, 1, 1], 'val': ['c', 'b', 'a', 'x']})
        res = t.unique()
        assert len(res) == 4
        assert sorted(list(res['id'])) == [1, 1, 2, 3]
        assert sorted(list(res['val'])) == ['a', 'b', 'c', 'x']


# noinspection PyClassHasNoInit
class TestXFrameSort:
    """
    Tests XFrame sort
    """

    def test_sort(self):
        t = XFrame({'id': [3, 2, 1], 'val': ['c', 'b', 'a']})
        res = t.sort('id')
        assert len(res) == 3
        assert list(res['id']) == [1, 2, 3]
        assert list(res['val']) == ['a', 'b', 'c']

    def test_sort_descending(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.sort('id', ascending=False)
        assert len(res) == 3
        assert list(res['id']) == [3, 2, 1]
        assert list(res['val']) == ['c', 'b', 'a']

    def test_sort_multi_col(self):
        t = XFrame({'id': [3, 2, 1, 1], 'val': ['c', 'b', 'b', 'a']})
        res = t.sort(['id', 'val'])
        assert len(res) == 4
        assert list(res['id']) == [1, 1, 2, 3]
        assert list(res['val']) == ['a', 'b', 'b', 'c']

    def test_sort_multi_col_asc_desc(self):
        t = XFrame({'id': [3, 2, 1, 1], 'val': ['c', 'b', 'b', 'a']})
        res = t.sort([('id', True), ('val', False)])
        assert len(res) == 4
        assert list(res['id']) == [1, 1, 2, 3]
        assert list(res['val']) == ['b', 'a', 'b', 'c']


# noinspection PyClassHasNoInit
class TestXFrameDropna:
    """
    Tests XFrame dropna
    """

    def test_dropna_no_drop(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.dropna()
        assert len(res) == 3
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_dropna_none(self):
        t = XFrame({'id': [1, None, 3], 'val': ['a', 'b', 'c']})
        res = t.dropna()
        assert len(res) == 2
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 3, 'val': 'c'}

    def test_dropna_nan(self):
        t = XFrame({'id': [1.0, float('nan'), 3.0], 'val': ['a', 'b', 'c']})
        res = t.dropna()
        assert len(res) == 2
        assert res[0] == {'id': 1.0, 'val': 'a'}
        assert res[1] == {'id': 3.0, 'val': 'c'}

    def test_dropna_float_none(self):
        t = XFrame({'id': [1.0, None, 3.0], 'val': ['a', 'b', 'c']})
        res = t.dropna()
        assert len(res) == 2
        assert res[0] == {'id': 1.0, 'val': 'a'}
        assert res[1] == {'id': 3.0, 'val': 'c'}

    def test_dropna_empty_list(self):
        t = XFrame({'id': [1, None, 3], 'val': ['a', 'b', 'c']})
        res = t.dropna(columns=[])
        assert len(res) == 3
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': None, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_dropna_any(self):
        t = XFrame({'id': [1, None, None], 'val': ['a', None, 'c']})
        res = t.dropna()
        assert len(res) == 1
        assert res[0] == {'id': 1, 'val': 'a'}

    def test_dropna_all(self):
        t = XFrame({'id': [1, None, None], 'val': ['a', None, 'c']})
        res = t.dropna(how='all')
        assert len(res) == 2
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': None, 'val': 'c'}

    def test_dropna_col_val(self):
        t = XFrame({'id': [1, None, None], 'val': ['a', None, 'c']})
        res = t.dropna(columns='val')
        assert len(res) == 2
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': None, 'val': 'c'}

    def test_dropna_col_id(self):
        t = XFrame({'id': [1, 2, None], 'val': ['a', None, 'c']})
        res = t.dropna(columns='id')
        assert len(res) == 2
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': None}

    # noinspection PyTypeChecker
    def test_dropna_bad_col_arg(self):
        t = XFrame({'id': [1, 2, None], 'val': ['a', None, 'c']})
        with pytest.raises(TypeError) as exception_info:
            t.dropna(columns=1)
        exception_message = exception_info.value.args[0]
        assert exception_message == "Must give columns as a 'list', 'str', or 'None'."

    def test_dropna_bad_col_name_in_list(self):
        t = XFrame({'id': [1, 2, None], 'val': ['a', None, 'c']})
        with pytest.raises(TypeError) as exception_info:
            t.dropna(columns=['id', 2])
        exception_message = exception_info.value.args[0]
        assert exception_message == "All columns must be of 'str' type."

    def test_dropna_bad_how(self):
        t = XFrame({'id': [1, 2, None], 'val': ['a', None, 'c']})
        with pytest.raises(ValueError) as exception_info:
            t.dropna(how='xx')
        exception_message = exception_info.value.args[0]
        assert exception_message == "Must specify 'any' or 'all'."

        # TODO drop_missing


# noinspection PyClassHasNoInit
class TestXFrameDropnaSplit:
    """
    Tests XFrame dropna_split
    """

    def test_dropna_split_no_drop(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res1, res2 = t.dropna_split()
        assert len(res1) == 3
        assert res1[0] == {'id': 1, 'val': 'a'}
        assert res1[1] == {'id': 2, 'val': 'b'}
        assert res1[2] == {'id': 3, 'val': 'c'}
        assert len(res2) == 0

    def test_dropna_split_none(self):
        t = XFrame({'id': [1, None, 3], 'val': ['a', 'b', 'c']})
        res1, res2 = t.dropna_split()
        assert len(res1) == 2
        assert res1[0] == {'id': 1, 'val': 'a'}
        assert res1[1] == {'id': 3, 'val': 'c'}
        assert len(res2) == 1
        assert res2[0] == {'id': None, 'val': 'b'}

    def test_dropna_split_all(self):
        t = XFrame({'id': [1, None, None], 'val': ['a', None, 'c']})
        res1, res2 = t.dropna_split(how='all')
        assert len(res1) == 2
        assert res1[0] == {'id': 1, 'val': 'a'}
        assert res1[1] == {'id': None, 'val': 'c'}
        assert len(res2) == 1
        assert res2[0] == {'id': None, 'val': None}


# noinspection PyClassHasNoInit
class TestXFrameFillna:
    """
    Tests XFrame fillna
    """

    def test_fillna(self):
        t = XFrame({'id': [1, None, None], 'val': ['a', 'b', 'c']})
        res = t.fillna('id', 0)
        assert len(res) == 3
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 0, 'val': 'b'}
        assert res[2] == {'id': 0, 'val': 'c'}

    def test_fillna_bad_col_name(self):
        t = XFrame({'id': [1, None, None], 'val': ['a', 'b', 'c']})
        with pytest.raises(ValueError) as exception_info:
            t.fillna('xx', 0)
        exception_message = exception_info.value.args[0]
        assert exception_message == "Column name does not exist: 'xx'."

    # noinspection PyTypeChecker
    def test_fillna_bad_arg_type(self):
        t = XFrame({'id': [1, None, None], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            t.fillna(1, 0)
        exception_message = exception_info.value.args[0]
        assert exception_message == "Must give column name as a 'str'. Found 'int': 1."


# noinspection PyClassHasNoInit
class TestXFrameAddRowNumber:
    """
    Tests XFrame add_row_number
    """

    def test_add_row_number(self):
        t = XFrame({'ident': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.add_row_number()
        assert res.column_names() == ['id', 'ident', 'val']
        assert res[0] == {'id': 0, 'ident': 1, 'val': 'a'}
        assert res[1] == {'id': 1, 'ident': 2, 'val': 'b'}
        assert res[2] == {'id': 2, 'ident': 3, 'val': 'c'}

    def test_add_row_number_start(self):
        t = XFrame({'ident': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.add_row_number(start=10)
        assert res.column_names() == ['id', 'ident', 'val']
        assert res[0] == {'id': 10, 'ident': 1, 'val': 'a'}
        assert res[1] == {'id': 11, 'ident': 2, 'val': 'b'}
        assert res[2] == {'id': 12, 'ident': 3, 'val': 'c'}

    def test_add_row_number_name(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.add_row_number(column_name='row_number')
        assert res.column_names() == ['row_number', 'id', 'val']
        assert res[0] == {'row_number': 0, 'id': 1, 'val': 'a'}
        assert res[1] == {'row_number': 1, 'id': 2, 'val': 'b'}
        assert res[2] == {'row_number': 2, 'id': 3, 'val': 'c'}


# noinspection PyClassHasNoInit
class TestXFrameShape:
    """
    Tests XFrame shape
    """

    def test_shape(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        assert t.shape == (3, 2)

    def test_shape_empty(self):
        t = XFrame()
        assert t.shape == (0, 0)


# noinspection SqlNoDataSourceInspection,SqlDialectInspection
# noinspection PyClassHasNoInit
class TestXFrameSql:
    """
    Tests XFrame sql
    """

    def test_sql(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.sql("SELECT * FROM xframe WHERE id > 1 ORDER BY id")

        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 2, 'val': 'b'}
        assert res[1] == {'id': 3, 'val': 'c'}

    def test_sql_name(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.sql("SELECT * FROM tmp_tbl WHERE id > 1 ORDER BY id", table_name='tmp_tbl')
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 2, 'val': 'b'}
        assert res[1] == {'id': 3, 'val': 'c'}
