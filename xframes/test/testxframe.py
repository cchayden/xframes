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
        assert 3 == len(res)
        assert ['val_int', 'val_int_signed', 'val_float', 'val_float_signed',
                'val_str', 'val_list', 'val_dict'] == res.column_names()
        assert [int, int, float, float, str, list, dict] == res.column_types()
        assert {'val_int': 1, 'val_int_signed': -1, 'val_float': 1.0, 'val_float_signed': -1.0,
                'val_str': 'a', 'val_list': ['a'], 'val_dict': {1: 'a'}} == res[0]
        assert {'val_int': 2, 'val_int_signed': -2, 'val_float': 2.0, 'val_float_signed': -2.0,
                'val_str': 'b', 'val_list': ['b'], 'val_dict': {2: 'b'}} == res[1]
        assert {'val_int': 3, 'val_int_signed': -3, 'val_float': 3.0, 'val_float_signed': -3.0,
                'val_str': 'c', 'val_list': ['c'], 'val_dict': {3: 'c'}} == res[2]

    def test_construct_auto_str_csv(self):
        path = 'files/test-frame.csv'
        res = XFrame(path)
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]
        assert {'id': 3, 'val': 'c'} == res[2]

    def test_construct_auto_str_tsv(self):
        path = 'files/test-frame.tsv'
        res = XFrame(path)
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]
        assert {'id': 3, 'val': 'c'} == res[2]

    def test_construct_auto_str_psv(self):
        path = 'files/test-frame.psv'
        res = XFrame(path)
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]
        assert {'id': 3, 'val': 'c'} == res[2]

    def test_construct_auto_str_txt(self):
        # construct and XFrame given a text file
        # interpret as csv
        path = 'files/test-frame.txt'
        res = XFrame(path)
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]
        assert {'id': 3, 'val': 'c'} == res[2]

    def test_construct_auto_str_noext(self):
        # construct and XFrame given a text file
        # interpret as csv
        path = 'files/test-frame'
        res = XFrame(path)
        res = res.sort('id')
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]
        assert {'id': 3, 'val': 'c'} == res[2]

    def test_construct_auto_pandas_dataframe(self):
        df = pandas.DataFrame({'id': [1, 2, 3], 'val': [10.0, 20.0, 30.0]})
        res = XFrame(df)
        assert 3 ==len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, float] == res.column_types()
        assert {'id': 1, 'val': 10.0} == res[0]
        assert {'id': 2, 'val': 20.0} == res[1]
        assert {'id': 3, 'val': 30.0} == res[2]

    def test_construct_auto_str_xframe(self):
        # construct an XFrame given a file with unrecognized file extension
        path = 'files/test-frame'
        res = XFrame(path)
        res = res.sort('id')
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]
        assert {'id': 3, 'val': 'c'} == res[2]

    def test_construct_xarray(self):
        # construct and XFrame given an XArray
        xa = XArray([1, 2, 3])
        t = XFrame(xa)
        assert 3 == len(t)
        assert ['X.0'] == t.column_names()
        assert [int] == t.column_types()
        assert {'X.0': 1} == t[0]
        assert {'X.0': 2} == t[1]
        assert {'X.0': 3} == t[2]

    def test_construct_xframe(self):
        # construct an XFrame given another XFrame
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = XFrame(t)
        assert 3 == len(res)
        res = res.sort('id')
        assert [1, 2, 3] == list(res['id'])
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()

    def test_construct_iteritems(self):
        # construct an XFrame from an object that has iteritems
        class MyIterItem(object):
            @staticmethod
            def items():
                return iter([('id', [1, 2, 3]), ('val', ['a', 'b', 'c'])])

        t = XFrame(MyIterItem())
        assert 3 == len(t)
        assert ['id', 'val'] == t.column_names()
        assert t.column_types() == [int, str]
        assert {'id': 1, 'val': 'a'} == t[0]
        assert {'id': 2, 'val': 'b'} == t[1]
        assert {'id': 3, 'val': 'c'} == t[2]

    def test_construct_iteritems_bad(self):
        # construct an XFrame from an object that has iteritems
        class MyIterItem(object):
            @staticmethod
            def items():
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
        assert 3 == len(t)
        assert ['X.0'] == t.column_names()
        assert [int] == t.column_types()
        assert {'X.0': 1} == t[0]
        assert {'X.0': 2} == t[1]
        assert {'X.0': 3} == t[2]

    def test_construct_iter_bad(self):
        # construct an XFrame from an object that has __iter__
        class MyIter(object):
            def __iter__(self):
                return iter([])

        with pytest.raises(TypeError) as exception_info:
            _ = XFrame(MyIter())
        exception_message = exception_info.value.args[0]
        assert 'Cannot determine types.' == exception_message

    def test_construct_empty(self):
        # construct an empty XFrame
        t = XFrame()
        assert 0 == len(t)

    def test_construct_str_csv(self):
        # construct and XFrame given a text file
        # interpret as csv
        path = 'files/test-frame.txt'
        res = XFrame(path, format='csv')
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]
        assert {'id': 3, 'val': 'c'} == res[2]

#    def test_save_test_frame(self):
#        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
#        t.save('files/test-frame-1', format='binary')

    def test_construct_str_xframe(self):
        # construct and XFrame given a saved xframe
        path = 'files/test-frame'
        res = XFrame(path, format='xframe')
        res = res.sort('id')
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]
        assert {'id': 3, 'val': 'c'} == res[2]

    def test_construct_array(self):
        # construct an XFrame from an array
        t = XFrame([1, 2, 3], format='array')
        assert 3 == len(t)
        assert [1, 2, 3] == list(t['X.0'])

    def test_construct_array_mixed_xarray(self):
        # construct an XFrame from an xarray and values
        xa = XArray([1, 2, 3])
        with pytest.raises(ValueError) as exception_info:
            _ = XFrame([1, 2, xa], format='array')
        exception_message = exception_info.value.args[0]
        assert 'Cannot create XFrame from mix of regular values and XArrays.' == exception_message

    def test_construct_array_mixed_types(self):
        # construct an XFrame from
        # an array of mixed types
        with pytest.raises(TypeError) as exception_info:
            _ = XFrame([1, 2, 'a'], format='array')
        exception_message = exception_info.value.args[0]
        assert "Infer_type_of_list: mixed types in list: <class 'str'> <class 'int'>" == exception_message

    def test_construct_unknown_format(self):
        # test unknown format
        with pytest.raises(ValueError) as exception_info:
            _ = XFrame([1, 2, 'a'], format='bad-format')
        exception_message = exception_info.value.args[0]
        assert "Unknown input type: 'bad-format'." == exception_message

    def test_construct_array_empty(self):
        # construct an XFrame from an empty array
        t = XFrame([], format='array')
        assert 0 == len(t)

    def test_construct_array_xarray(self):
        # construct an XFrame from an xarray
        xa1 = XArray([1, 2, 3])
        xa2 = XArray(['a', 'b', 'c'])
        t = XFrame([xa1, xa2], format='array')
        assert 3 == len(t)
        assert ['X.0', 'X.1'] == t.column_names()
        assert [int, str] == t.column_types()
        assert {'X.0': 1, 'X.1': 'a'} == t[0]
        assert {'X.0': 2, 'X.1': 'b'} == t[1]
        assert {'X.0': 3, 'X.1': 'c'} == t[2]

    def test_construct_dict_int(self):
        # construct an XFrame from a dict of int
        t = XFrame({'id': [1, 2, 3], 'val': [10, 20, 30]}, format='dict')
        res = XFrame(t)
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, int] == res.column_types()
        assert {'id': 1, 'val': 10} == res[0]
        assert {'id': 2, 'val': 20} == res[1]
        assert {'id': 3, 'val': 30} == res[2]

    def test_construct_dict_float(self):
        # construct an XFrame from a dict of float
        t = XFrame({'id': [1.0, 2.0, 3.0], 'val': [10.0, 20.0, 30.0]}, format='dict')
        res = XFrame(t)
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [float, float] == res.column_types()
        assert {'id': 1.0, 'val': 10.0} == res[0]
        assert {'id': 2.0, 'val': 20.0} == res[1]
        assert {'id': 3.0, 'val': 30.0} == res[2]

    def test_construct_dict_str(self):
        # construct an XFrame from a dict of str
        t = XFrame({'id': ['a', 'b', 'c'], 'val': ['A', 'B', 'C']}, format='dict')
        res = XFrame(t)
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [str, str] == res.column_types()
        assert {'id': 'a', 'val': 'A'} == res[0]
        assert {'id': 'b', 'val': 'B'} == res[1]
        assert {'id': 'c', 'val': 'C'} == res[2]

    def test_construct_dict_int_str(self):
        # construct an XFrame from a dict of int and str
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']}, format='dict')
        assert 3 == len(t)
        t = t.sort('id')
        assert [1, 2, 3] == list(t['id'])
        assert ['id', 'val'] == t.column_names()
        assert [int, str] == t.column_types()

    def test_construct_dict_int_str_bad_len(self):
        # construct an XFrame from a dict of int and str with different lengths
        with pytest.raises(ValueError) as exception_info:
            XFrame({'id': [1, 2, 3], 'val': ['a', 'b']})
        exception_message = exception_info.value.args[0]
        assert 'Cannot create XFrame from dict of lists of different lengths.' == exception_message

    def test_construct_binary(self):
        # make binary file
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        path = 'tmp/frame'
        t.save(path, format='binary')  # File does not necessarily save in order
        res = XFrame(path).sort('id')  # so let's sort after we read it back
        assert 3 == len(res)
        assert ['id', 'val'] == t.column_names()
        assert [int, str] == t.column_types()
        assert {'id': 1, 'val': 'a'} == res[0]

    def test_construct_rdd(self):
        sc = CommonSparkContext.spark_context()
        rdd = sc.parallelize([(1, 'a'), (2, 'b'), (3, 'c')])
        res = XFrame(rdd)
        assert 3 == len(res)
        assert {'X.0': 1, 'X.1': 'a'} == res[0]
        assert {'X.0': 2, 'X.1': 'b'} == res[1]

    def test_construct_spark_dataframe(self):
        sc = CommonSparkContext.spark_context()
        rdd = sc.parallelize([(1, 'a'), (2, 'b'), (3, 'c')])
        fields = [StructField('id', IntegerType(), True), StructField('val', StringType(), True)]
        schema = StructType(fields)
        sqlc = CommonSparkContext.spark_sql_context()
        s_rdd = sqlc.createDataFrame(rdd, schema)
        res = XFrame(s_rdd)
        assert 3 == len(res)
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]

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
        assert 3 == len(res)
        assert errs == {}

    def test_read_csv_width_error(self):
        path = 'files/test-frame-width-err.csv'
        res, errs = XFrame.read_csv_with_errors(path)
        assert 'width' in errs
        width_errs = errs['width']
        assert isinstance(width_errs, XArray)
        assert 2 == len(width_errs)
        assert '1' == width_errs[0]
        assert '2,x,y' == width_errs[1]
        assert 2 == len(res)

    def test_read_csv_null_error(self):
        path = 'files/test-frame-null.csv'
        res, errs = XFrame.read_csv_with_errors(path)
        assert 'csv' in errs
        csv_errs = errs['csv']
        assert isinstance(csv_errs, XArray)
        assert 1 == len(csv_errs)
        assert '2,\x00b' == csv_errs[0]
        assert 1 == len(res)

    def test_read_csv_null_header_error(self):
        path = 'files/test-frame-null-header.csv'
        res, errs = XFrame.read_csv_with_errors(path)
        assert 'header' in errs
        csv_errs = errs['header']
        assert isinstance(csv_errs, XArray)
        assert 1 == len(csv_errs)
        assert 'id,\x00val' == csv_errs[0]
        assert 0 == len(res)

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
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]
        assert {'id': 3, 'val': 'c'} == res[2]

    def test_read_csv_verbose(self):
        path = 'files/test-frame.csv'
        res = XFrame.read_csv(path)
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]
        assert {'id': 3, 'val': 'c'} == res[2]

    def test_read_csv_delim(self):
        path = 'files/test-frame.psv'
        res = XFrame.read_csv(path, delimiter='|')
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]
        assert {'id': 3, 'val': 'c'} == res[2]

    def test_read_csv_no_header(self):
        path = 'files/test-frame-no-header.csv'
        res = XFrame.read_csv(path, header=False)
        assert 3 == len(res)
        assert ['X.0', 'X.1'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'X.0': 1, 'X.1': 'a'} == res[0]
        assert {'X.0': 2, 'X.1': 'b'} == res[1]
        assert {'X.0': 3, 'X.1': 'c'} == res[2]

    def test_read_csv_comment(self):
        path = 'files/test-frame-comment.csv'
        res = XFrame.read_csv(path, comment_char='#')
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]
        assert {'id': 3, 'val': 'c'} == res[2]

    def test_read_csv_escape(self):
        path = 'files/test-frame-escape.csv'
        res = XFrame.read_csv(path)
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'a,a'} == res[0]
        assert {'id': 2, 'val': 'b,b'} == res[1]
        assert {'id': 3, 'val': 'c,c'} == res[2]

    def test_read_csv_escape_custom(self):
        path = 'files/test-frame-escape-custom.csv'
        res = XFrame.read_csv(path, escape_char='$')
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'a,a'} == res[0]
        assert {'id': 2, 'val': 'b,b'} == res[1]
        assert {'id': 3, 'val': 'c,c'} == res[2]

    def test_read_csv_initial_space(self):
        path = 'files/test-frame-initial_space.csv'
        res = XFrame.read_csv(path, skip_initial_space=True)
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]
        assert {'id': 3, 'val': 'c'} == res[2]

    def test_read_csv_hints_type(self):
        path = 'files/test-frame.csv'
        res = XFrame.read_csv(path, column_type_hints=str)
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [str, str] == res.column_types()
        assert {'id': '1', 'val': 'a'} == res[0]
        assert {'id': '2', 'val': 'b'} == res[1]
        assert {'id': '3', 'val': 'c'} == res[2]

    def test_read_csv_hints_list(self):
        path = 'files/test-frame-extra.csv'
        res = XFrame.read_csv(path, column_type_hints=[str, str, int])
        assert 3 == len(res)
        assert ['id', 'val1', 'val2'] == res.column_names()
        assert [str, str, int] == res.column_types()
        assert {'id': '1', 'val1': 'a', 'val2': 10} == res[0]
        assert {'id': '2', 'val1': 'b', 'val2': 20} == res[1]
        assert {'id': '3', 'val1': 'c', 'val2': 30} == res[2]

    # noinspection PyTypeChecker
    def test_read_csv_hints_dict(self):
        path = 'files/test-frame-extra.csv'
        res = XFrame.read_csv(path, column_type_hints={'val2': int})
        assert 3 == len(res)
        assert ['id', 'val1', 'val2'] == res.column_names()
        assert [str, str, int] == res.column_types()
        assert {'id': '1', 'val1': 'a', 'val2': 10} == res[0]
        assert {'id': '2', 'val1': 'b', 'val2': 20} == res[1]
        assert {'id': '3', 'val1': 'c', 'val2': 30} == res[2]

    def test_read_csv_na(self):
        path = 'files/test-frame-na.csv'
        res = XFrame.read_csv(path, na_values='None')
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'NA'} == res[0]
        assert {'id': None, 'val': 'b'} == res[1]
        assert {'id': 3, 'val': 'c'} == res[2]

    def test_read_csv_na_mult(self):
        path = 'files/test-frame-na.csv'
        res = XFrame.read_csv(path, na_values=['NA', 'None'])
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': None} == res[0]
        assert {'id': None, 'val': 'b'} == res[1]
        assert {'id': 3, 'val': 'c'} == res[2]

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
        assert 3 == len(res)
        assert ['text'] == res.column_names()
        assert [str] == res.column_types()
        assert {'text': 'This is a test'} == res[0]
        assert {'text': 'of read_text.'} == res[1]
        assert {'text': 'Here is another sentence.'} == res[2]

    def test_read_text_delimited(self):
        path = 'files/test-frame-text.txt'
        res = XFrame.read_text(path, delimiter='.')
        assert 3 == len(res)
        assert ['text'] == res.column_names()
        assert [str] == res.column_types()
        assert {'text': 'This is a test of read_text'} == res[0]
        assert {'text': 'Here is another sentence'} == res[1]
        assert {'text': ''} == res[2]

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
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]
        assert {'id': 3, 'val': 'c'} == res[2]

    def test_read_parquet_bool(self):
        t = XFrame({'id': [1, 2, 3], 'val': [True, False, True]})
        path = 'tmp/frame-parquet'
        t.save(path, format='parquet')

        res = XFrame('tmp/frame-parquet.parquet')
        res = res.sort('id')
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, bool] == res.column_types()
        assert {'id': 1, 'val': True} == res[0]
        assert {'id': 2, 'val': False} == res[1]
        assert {'id': 3, 'val': True} == res[2]

    def test_read_parquet_int(self):
        t = XFrame({'id': [1, 2, 3], 'val': [10, 20, 30]})
        path = 'tmp/frame-parquet'
        t.save(path, format='parquet')

        res = XFrame('tmp/frame-parquet.parquet')
        res = res.sort('id')
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, int] == res.column_types()
        assert {'id': 1, 'val': 10} == res[0]
        assert {'id': 2, 'val': 20} == res[1]
        assert {'id': 3, 'val': 30} == res[2]

    def test_read_parquet_float(self):
        t = XFrame({'id': [1, 2, 3], 'val': [1.0, 2.0, 3.0]})
        path = 'tmp/frame-parquet'
        t.save(path, format='parquet')

        res = XFrame('tmp/frame-parquet.parquet')
        res = res.sort('id')
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, float] == res.column_types()
        assert {'id': 1, 'val': 1.0} == res[0]
        assert {'id': 2, 'val': 2.0} == res[1]
        assert {'id': 3, 'val': 3.0} == res[2]

    def test_read_parquet_list(self):
        t = XFrame({'id': [1, 2, 3], 'val': [[1, 1], [2, 2], [3, 3]]})
        path = 'tmp/frame-parquet'
        t.save(path, format='parquet')

        res = XFrame('tmp/frame-parquet.parquet')
        res = res.sort('id')
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, list] == res.column_types()
        assert {'id': 1, 'val': [1, 1]} == res[0]
        assert {'id': 2, 'val': [2, 2]} == res[1]
        assert {'id': 3, 'val': [3, 3]} == res[2]

    def test_read_parquet_dict(self):
        t = XFrame({'id': [1, 2, 3], 'val': [{1: 1}, {2: 2}, {3: 3}]})
        path = 'tmp/frame-parquet'
        t.save(path, format='parquet')

        res = XFrame('tmp/frame-parquet.parquet')
        res = res.sort('id')
        assert 3 == len(res)
        assert ['id', 'val'] == res.column_names()
        assert [int, dict] == res.column_types()
        assert {'id': 1, 'val': {1: 1}} == res[0]
        assert {'id': 2, 'val': {2: 2}} == res[1]
        assert {'id': 3, 'val': {3: 3}} == res[2]

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
        assert 3 == len(res)
        assert ['id'] == res.column_names()
        assert [int] == res.column_types()
        assert {'id': 1} == res[0]
        assert {'id': 2} == res[1]
        assert {'id': 3} == res[2]


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
        assert 3 == results.count()
        row = results.collect()[0]
        assert 1 == row.id
        assert 'a' == row.val

    def test_to_spark_dataframe_bool(self):
        t = XFrame({'id': [1, 2, 3], 'val': [True, False, True]})
        t.to_spark_dataframe('tmp_tbl')
        sqlc = CommonSparkContext.spark_sql_context()
        results = sqlc.sql('SELECT * FROM tmp_tbl ORDER BY id')
        assert 3 == results.count()
        row = results.collect()[0]
        assert 1 == row.id
        assert row.val is True

    def test_to_spark_dataframe_float(self):
        t = XFrame({'id': [1, 2, 3], 'val': [1.0, 2.0, 3.0]})
        t.to_spark_dataframe('tmp_tbl')
        sqlc = CommonSparkContext.spark_sql_context()
        results = sqlc.sql('SELECT * FROM tmp_tbl ORDER BY id')
        assert 3 == results.count()
        row = results.collect()[0]
        assert 1 == row.id
        assert 1.0 == row.val

    def test_to_spark_dataframe_int(self):
        t = XFrame({'id': [1, 2, 3], 'val': [1, 2, 3]})
        t.to_spark_dataframe('tmp_tbl')
        sqlc = CommonSparkContext.spark_sql_context()
        results = sqlc.sql('SELECT * FROM tmp_tbl ORDER BY id')
        assert 3 == results.count()
        row = results.collect()[0]
        assert 1 == row.id
        assert 1 == row.val

    def test_to_spark_dataframe_list(self):
        t = XFrame({'id': [1, 2, 3], 'val': [[1, 1], [2, 2], [3, 3]]})
        t.to_spark_dataframe('tmp_tbl')
        sqlc = CommonSparkContext.spark_sql_context()
        results = sqlc.sql('SELECT * FROM tmp_tbl ORDER BY id')
        assert 3 == results.count()
        row = results.collect()[0]
        assert 1 == row.id
        assert [1, 1] == row.val

    def test_to_spark_dataframe_list_hint(self):
        t = XFrame({'id': [1, 2, 3], 'val': [[None, 1], [2, 2], [3, 3]]})
        t.to_spark_dataframe('tmp_tbl', column_type_hints={'val': 'list[int]'})
        sqlc = CommonSparkContext.spark_sql_context()
        results = sqlc.sql('SELECT * FROM tmp_tbl ORDER BY id')
        assert 3 == results.count()
        row = results.collect()[1]
        assert 2 == row.id
        assert [2, 2] == row.val

    def test_to_spark_dataframe_list_bad(self):
        t = XFrame({'id': [1, 2, 3], 'val': [[None, 1], [2, 2], [3, 3]]})
        with pytest.raises(TypeError) as exception_info:
            t.to_spark_dataframe('tmp_tbl')
        exception_message = exception_info.value.args[0]
        assert 'Element type cannot be determined.' == exception_message

    def test_to_spark_dataframe_map(self):
        t = XFrame({'id': [1, 2, 3], 'val': [{'x': 1}, {'y': 2}, {'z': 3}]})
        t.to_spark_dataframe('tmp_tbl')
        sqlc = CommonSparkContext.spark_sql_context()
        results = sqlc.sql('SELECT * FROM tmp_tbl ORDER BY id')
        assert 3 == results.count()
        row = results.collect()[0]
        assert 1 == row.id
        assert {'x': 1} == row.val

    def test_to_spark_dataframe_map_bad(self):
        t = XFrame({'id': [1, 2, 3], 'val': [None, {'y': 2}, {'z': 3}]})
        with pytest.raises(ValueError) as exception_info:
            t.to_spark_dataframe('tmp_tbl')
        exception_message = exception_info.value.args[0]
        assert 'Schema type cannot be determined.' == exception_message

    @pytest.mark.skip(reason='files in spark 2')
    def test_to_spark_dataframe_map_hint(self):
        t = XFrame({'id': [1, 2, 3], 'val': [{None: None}, {'y': 2}, {'z': 3}]})
        t.to_spark_dataframe('tmp_tbl', column_type_hints={'val': 'dict{str: int}'})
        sqlc = CommonSparkContext.spark_sql_context()
        results = sqlc.sql('SELECT * FROM tmp_tbl ORDER BY id')
        assert 3 == results.count()
        row = results.collect()[1]
        assert 1 == row.id
        assert {'y': 2} == row.val

    def test_to_spark_dataframe_str_rewrite(self):
        t = XFrame({'id': [1, 2, 3], 'val;1': ['a', 'b', 'c']})
        t.to_spark_dataframe('tmp_tbl')
        sqlc = CommonSparkContext.spark_sql_context()
        results = sqlc.sql('SELECT * FROM tmp_tbl ORDER BY id')
        assert 3 == results.count()
        row = results.collect()[0]
        assert 1 == row.id
        assert 'a' == row.val_1

    def test_to_spark_dataframe_str_rename(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t.to_spark_dataframe('tmp_tbl', column_names=['id1', 'val1'])
        sqlc = CommonSparkContext.spark_sql_context()
        results = sqlc.sql('SELECT * FROM tmp_tbl ORDER BY id1')
        assert 3 == results.count()
        row = results.collect()[0]
        assert 1 == row.id1
        assert 'a' == row.val1

    def test_to_spark_dataframe_str_rename_bad_type(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            t.to_spark_dataframe('tmp_tbl', column_names='id1')

    def test_to_spark_dataframe_str_rename_bad_len(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(ValueError) as exception_info:
            t.to_spark_dataframe('tmp_tbl', column_names=['id1'])
        exception_message = exception_info.value.args[0]
        assert 'Column names list must match number of columns: actual: 1, expected: 2' == exception_message


# noinspection PyClassHasNoInit
class TestXFrameToRdd:
    """
    Tests XFrame to_rdd
    """

    def test_to_rdd(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        rdd_val = t.to_rdd().collect()
        assert (1, 'a') == rdd_val[0]
        assert (2, 'b') == rdd_val[1]
        assert (3, 'c') == rdd_val[2]


# noinspection PyClassHasNoInit
class TestXFrameFromRdd:
    """
    Tests XFrame from_rdd with regular rdd
    """

    def test_from_rdd(self):
        sc = CommonSparkContext.spark_context()
        rdd = sc.parallelize([(1, 'a'), (2, 'b'), (3, 'c')])
        res = XFrame.from_rdd(rdd)
        assert 3 == len(res)
        assert {'X.0': 1, 'X.1': 'a'} == res[0]
        assert {'X.0': 2, 'X.1': 'b'} == res[1]

    def test_from_rdd_names(self):
        sc = CommonSparkContext.spark_context()
        rdd = sc.parallelize([(1, 'a'), (2, 'b'), (3, 'c')])
        res = XFrame.from_rdd(rdd, column_names=['id', 'val'])
        assert 3 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]

    def test_from_rdd_types(self):
        sc = CommonSparkContext.spark_context()
        rdd = sc.parallelize([(None, 'a'), (2, 'b'), (3, 'c')])
        res = XFrame.from_rdd(rdd, column_types=[int, str])
        assert 3 == len(res)
        assert [int, str] == res.column_types()
        assert {'X.0': None, 'X.1': 'a'} == res[0]
        assert {'X.0': 2, 'X.1': 'b'} == res[1]

    def test_from_rdd_names_types(self):
        sc = CommonSparkContext.spark_context()
        rdd = sc.parallelize([(None, 'a'), (2, 'b'), (3, 'c')])
        res = XFrame.from_rdd(rdd, column_names=['id', 'val'], column_types=[int, str])
        assert 3 == len(res)
        assert [int, str] == res.column_types()
        assert {'id': None, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]

    def test_from_rdd_names_bad(self):
        sc = CommonSparkContext.spark_context()
        rdd = sc.parallelize([(1, 'a'), (2, 'b'), (3, 'c')])
        with pytest.raises(ValueError) as exception_info:
            XFrame.from_rdd(rdd, column_names=('id',))
        exception_message = exception_info.value.args[0]
        assert "Length of names does not match RDD: ('id',) (1, 'a')." == exception_message

    def test_from_rdd_types_bad(self):
        sc = CommonSparkContext.spark_context()
        rdd = sc.parallelize([(None, 'a'), (2, 'b'), (3, 'c')])
        with pytest.raises(ValueError) as exception_info:
            XFrame.from_rdd(rdd, column_types=(int,))
        exception_message = exception_info.value.args[0]
        assert 'Length of types does not match RDD.' == exception_message


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
        assert 3 == len(res)
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]


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
        assert '+----+-----+' == s[0]
        assert '| id | val |' == s[1]
        assert '+----+-----+' == s[2]
        assert '| 1  |  a  |' == s[3]
        assert '| 2  |  b  |' == s[4]
        assert '| 3  |  c  |' == s[5]
        assert '+----+-----+' == s[6]


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
        assert 3 == len(t)

    def test_len_zero(self):
        t = XFrame()
        assert 0 == len(t)

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
        assert 3 == len(x)
        assert [1, 2, 3] == list(x['id'])
        assert ['id', 'val'] == x.column_names()
        assert [int, str] == x.column_types()


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
class TestXFrameNumRows:
    """
    Tests XFrame num_rows
    """

    def test_num_rows(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        assert 3 == t.num_rows()


# noinspection PyClassHasNoInit
class TestXFrameNumColumns:
    """
    Tests XFrame num_columns
    """

    def test_num_columns(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        assert 2 == t.num_columns()


# noinspection PyClassHasNoInit
class TestXFrameColumnNames:
    """
    Tests XFrame column_names
    """

    def test_column_names(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        names = t.column_names()
        assert ['id', 'val'] == names


# noinspection PyClassHasNoInit
class TestXFrameColumnTypes:
    """
    Tests XFrame column_types
    """

    def test_column_types(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        types = t.column_types()
        assert [int, str] == types


# noinspection PyClassHasNoInit
class TestXFrameSelectRows:
    """
    Tests XFrame select_rows
    """

    def test_select_rowsr(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        a = XArray([1, 0, 1])
        res = t.select_rows(a)
        assert 2 == len(res)
        assert [1, 3] == list(res['id'])
        assert ['a', 'c'] == list(res['val'])


# noinspection PyClassHasNoInit
class TestXFrameHead:
    """
    Tests XFrame head
    """

    def test_head(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        hd = t.head(2)
        assert 2 == len(hd)
        assert [1, 2] == list(hd['id'])
        assert ['a', 'b'] == list(hd['val'])


# noinspection PyClassHasNoInit
class TestXFrameTail:
    """
    Tests XFrame tail
    """

    def test_tail(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        tl = t.tail(2)
        assert 2 == len(tl)
        assert [2, 3] == list(tl['id'])
        assert ['b', 'c'] == list(tl['val'])


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
        assert 3 == len(df)
        assert 1 == df['id'][0]
        assert 2 == df['id'][1]
        assert 'a' == df['val'][0]

    def test_to_pandas_dataframe_bool(self):
        t = XFrame({'id': [1, 2, 3], 'val': [True, False, True]})
        df = t.to_pandas_dataframe()
        assert 3 == len(df)
        assert 1 == df['id'][0]
        assert 2 == df['id'][1]
        assert True == df['val'][0]
        assert False == df['val'][1]

    def test_to_pandas_dataframe_float(self):
        t = XFrame({'id': [1, 2, 3], 'val': [1.0, 2.0, 3.0]})
        df = t.to_pandas_dataframe()
        assert 3 == len(df)
        assert 1 == df['id'][0]
        assert 2 == df['id'][1]
        assert 1.0 == df['val'][0]
        assert 2.0 == df['val'][1]

    def test_to_pandas_dataframe_int(self):
        t = XFrame({'id': [1, 2, 3], 'val': [1, 2, 3]})
        df = t.to_pandas_dataframe()
        assert 3 == len(df)
        assert 1 == df['id'][0]
        assert 2 == df['id'][1]
        assert 1 == df['val'][0]
        assert 2 == df['val'][1]

    def test_to_pandas_dataframe_list(self):
        t = XFrame({'id': [1, 2, 3], 'val': [[1, 1], [2, 2], [3, 3]]})
        df = t.to_pandas_dataframe()
        assert 3 == len(df)
        assert 1 == df['id'][0]
        assert 2 == df['id'][1]
        assert [1, 1] == df['val'][0]
        assert [2, 2] == df['val'][1]

    def test_to_pandas_dataframe_map(self):
        t = XFrame({'id': [1, 2, 3], 'val': [{'x': 1}, {'y': 2}, {'z': 3}]})
        df = t.to_pandas_dataframe()
        assert 3 == len(df)
        assert 1 == df['id'][0]
        assert 2 == df['id'][1]
        assert {'x': 1} == df['val'][0]
        assert {'y': 2} == df['val'][1]


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
        assert 3 == len(res)
        assert [int, str] == res.dtype()
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]

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
        assert 3 == len(res)
        assert [int, str, int] == res.dtype()
        assert {'id': 1, 'val': 'a', 'ini': 99} == res[0]
        assert {'id': 2, 'val': 'b', 'ini': 99} == res[1]


# noinspection PyClassHasNoInit
class TestXFrameApply:
    """
    Tests XFrame apply
    """

    def test_apply(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.apply(lambda row: row['id'] * 2)
        assert 3 == len(res)
        assert res.dtype() is int
        assert [2, 4, 6] == list(res)

    def test_apply_float(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.apply(lambda row: row['id'] * 2, dtype=float)
        assert 3 == len(res)
        assert res.dtype() is float
        assert [2.0, 4.0, 6.0] == list(res)

    def test_apply_str(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.apply(lambda row: row['id'] * 2, dtype=str)
        assert 3 == len(res)
        assert res.dtype() is str
        assert ['2', '4', '6'] == list(res)


# noinspection PyClassHasNoInit
class TestXFrameTransformCol:
    """
    Tests XFrame transform_col
    """

    def test_transform_col_identity(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.transform_col('id')
        assert 3 == len(res)
        assert [int, str] == res.dtype()
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]

    def test_transform_col_lambda(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.transform_col('id', lambda row: row['id'] * 2)
        assert 3 == len(res)
        assert [int, str] == res.dtype()
        assert {'id': 2, 'val': 'a'} == res[0]
        assert {'id': 4, 'val': 'b'} == res[1]

    def test_transform_col_type(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.transform_col('id', lambda row: 'x' * row['id'])
        assert 3 == len(res)
        assert [str, str] == res.dtype()
        assert {'id': 'x', 'val': 'a'} == res[0]
        assert {'id': 'xx', 'val': 'b'} == res[1]

    def test_transform_col_cast(self):
        t = XFrame({'id': ['1', '2', '3'], 'val': ['a', 'b', 'c']})
        res = t.transform_col('id', dtype=int)
        assert 3 == len(res)
        assert [int, str] == res.dtype()
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]


# noinspection PyClassHasNoInit
class TestXFrameTransformCols:
    """
    Tests XFrame transform_cols
    """

    def test_transform_cols_identity(self):
        t = XFrame({'other': ['x', 'y', 'z'], 'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.transform_cols(['id', 'val'])
        assert 3 == len(res)
        assert [int, str, str] == res.dtype()
        assert {'other': 'x', 'id': 1, 'val': 'a'} == res[0]
        assert {'other': 'y', 'id': 2, 'val': 'b'} == res[1]

    def test_transform_cols_lambda(self):
        t = XFrame({'other': ['x', 'y', 'z'], 'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.transform_cols(['id', 'val'], lambda row: [row['id'] * 2, row['val'] + 'x'])
        assert 3 == len(res)
        assert [int, str, str] == res.dtype()
        assert {'other': 'x', 'id': 2, 'val': 'ax'} == res[0]
        assert {'other': 'y', 'id': 4, 'val': 'bx'} == res[1]

    def test_transform_cols_type(self):
        t = XFrame({'other': ['x', 'y', 'z'], 'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.transform_cols(['id', 'val'], lambda row: ['x' * row['id'], ord(row['val'][0])])
        assert 3 == len(res)
        assert [str, str, int] == res.dtype()
        assert {'other': 'x', 'id': 'x', 'val': 97} == res[0]
        assert {'other': 'y', 'id': 'xx', 'val': 98} == res[1]

    def test_transform_cols_cast(self):
        t = XFrame({'other': ['x', 'y', 'z'], 'id': ['1', '2', '3'], 'val': [10, 20, 30]})
        res = t.transform_cols(['id', 'val'], dtypes=[int, str])
        assert 3 == len(res)
        assert [int, str, str] == res.dtype()
        assert {'other': 'x', 'id': 1, 'val': '10'} == res[0]
        assert {'other': 'y', 'id': 2, 'val': '20'} == res[1]


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
        assert ['number', 'letter'] == res.column_names()
        assert [int, str] == res.dtype()
        assert {'number': 1, 'letter': 'a'} == res[0]
        assert {'number': 2, 'letter': 'b'} == res[1]
        assert {'number': 2, 'letter': 'b'} == res[2]
        assert {'number': 3, 'letter': 'c'} == res[3]
        assert {'number': 3, 'letter': 'c'} == res[4]
        assert {'number': 3, 'letter': 'c'} == res[5]

    def test_flat_map_identity(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.flat_map(['number', 'letter'],
                         lambda row: [[row['id'], row['val']]],
                         column_types=[int, str])
        assert 3 == len(res)
        assert ['number', 'letter'] == res.column_names()
        assert [int, str] == res.dtype()
        assert {'number': 1, 'letter': 'a'} == res[0]
        assert {'number': 2, 'letter': 'b'} == res[1]
        assert {'number': 3, 'letter': 'c'} == res[2]

    def test_flat_map_mapped(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.flat_map(['number', 'letter'],
                         lambda row: [[row['id'] * 2, row['val'] + 'x']],
                         column_types=[int, str])
        assert 3 == len(res)
        assert ['number', 'letter'] == res.column_names()
        assert [int, str] == res.dtype()
        assert {'number': 2, 'letter': 'ax'} == res[0]
        assert {'number': 4, 'letter': 'bx'} == res[1]
        assert {'number': 6, 'letter': 'cx'} == res[2]

    def test_flat_map_auto(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.flat_map(['number', 'letter'],
                         lambda row: [[row['id'] * 2, row['val'] + 'x']])
        assert 3 == len(res)
        assert ['number', 'letter'] == res.column_names()
        assert [int, str] == res.dtype()
        assert {'number': 2, 'letter': 'ax'} == res[0]
        assert {'number': 4, 'letter': 'bx'} == res[1]
        assert {'number': 6, 'letter': 'cx'} == res[2]

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
        assert 1 == len(res)
        assert {'id': 2, 'val': 'b'} == res[0]

    @pytest.mark.skip(reason='depends on number of partitions')
    def test_sample_08(self):
        t = XFrame({'id': [1, 2, 3, 4, 5], 'val': ['a', 'b', 'c', 'd', 'e']})
        res = t.sample(0.8, 3)
        assert 3 == len(res)
        assert {'id': 2, 'val': 'b'} == res[0]
        assert {'id': 4, 'val': 'd'} == res[1]
        assert {'id': 5, 'val': 'e'} == res[2]


# noinspection PyClassHasNoInit
class TestXFrameRandomSplit:
    """
    Tests XFrame random_split
    """

    @pytest.mark.skip(reason='depends on number of partitions')
    def test_random_split(self):
        t = XFrame({'id': [1, 2, 3, 4, 5], 'val': ['a', 'b', 'c', 'd', 'e']})
        res1, res2 = t.random_split(0.5, 1)
        assert 3 == len(res1)
        assert {'id': 1, 'val': 'a'} ==res1[0]
        assert {'id': 4, 'val': 'd'} == res1[1]
        assert {'id': 5, 'val': 'e'} == res1[2]
        assert 2 == len(res2)
        assert {'id': 2, 'val': 'b'} ==res2[0]
        assert {'id': 3, 'val': 'c'} == res2[1]


# noinspection PyClassHasNoInit
class TestXFrameTopk:
    """
    Tests XFrame topk
    """

    def test_topk_int(self):
        t = XFrame({'id': [10, 20, 30], 'val': ['a', 'b', 'c']})
        res = t.topk('id', 2)
        assert 2 == len(res)
        # noinspection PyUnresolvedReferences
        assert XArray([30, 20]) == res['id'].all()
        assert ['c', 'b'] == list(res['val'])
        assert [int, str] == res.column_types()
        assert ['id', 'val'] == res.column_names()

    def test_topk_int_reverse(self):
        t = XFrame({'id': [30, 20, 10], 'val': ['c', 'b', 'a']})
        res = t.topk('id', 2, reverse=True)
        assert 2 == len(res)
        assert [10, 20] == list(res['id'])
        assert ['a', 'b'] == list(res['val'])

    def test_topk_float(self):
        t = XFrame({'id': [10.0, 20.0, 30.0], 'val': ['a', 'b', 'c']})
        res = t.topk('id', 2)
        assert 2 == len(res)
        assert XArray([30.0, 20.0]) == res['id'].all()
        assert ['c', 'b'] == list(res['val'])
        assert [float, str] == res.column_types()
        assert ['id', 'val'] == res.column_names()

    def test_topk_float_reverse(self):
        t = XFrame({'id': [30.0, 20.0, 10.0], 'val': ['c', 'b', 'a']})
        res = t.topk('id', 2, reverse=True)
        assert 2 == len(res)
        assert [10.0, 20.0] == list(res['id'])
        assert ['a', 'b'] == list(res['val'])

    def test_topk_str(self):
        t = XFrame({'id': [30, 20, 10], 'val': ['a', 'b', 'c']})
        res = t.topk('val', 2)
        assert 2 == len(res)
        assert [10, 20] == list(res['id'])
        assert ['c', 'b'] == list(res['val'])
        assert [int, str] == res.column_types()
        assert ['id', 'val'] == res.column_names()

    def test_topk_str_reverse(self):
        t = XFrame({'id': [10, 20, 30], 'val': ['c', 'b', 'a']})
        res = t.topk('val', 2, reverse=True)
        assert 2 == len(res)
        assert [30, 20] == list(res['id'])
        assert ['a', 'b'] == list(res['val'])


# noinspection PyClassHasNoInit
class TestXFrameSaveBinary:
    """
    Tests XFrame save binary format
    """

    def test_save(self):
        t = XFrame({'id': [30, 20, 10], 'val': ['a', 'b', 'c']})
        path = 'tmp/frame'
        t.save(path, format='binary')
        with open(os.path.join(path, '_metadata'), 'rb') as f:
            metadata = pickle.load(f)
            assert [['id', 'val'], [int, str]] == metadata
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
            assert 'id,val' == heading
            assert '30,a' == f.readline().rstrip()
            assert '20,b' == f.readline().rstrip()
            assert '10,c' == f.readline().rstrip()

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
        res_sort = res.sort('id')
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'a'} == res_sort[0]
        assert {'id': 2, 'val': 'b'} == res_sort[1]
        assert {'id': 3, 'val': 'c'} == res_sort[2]

    def test_save_as_parquet(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        path = 'tmp/frame-parquet'
        t.save_as_parquet(path)
        res = XFrame(path, format='parquet')
        res_sort = res.sort('id')
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val': 'a'} == res_sort[0]
        assert {'id': 2, 'val': 'b'} == res_sort[1]
        assert {'id': 3, 'val': 'c'} == res_sort[2]

    def test_save_rename(self):
        t = XFrame({'id col': [1, 2, 3], 'val,col': ['a', 'b', 'c']})
        path = 'tmp/frame-parquet'
        t.save(path, format='parquet')
        res = XFrame(path + '.parquet')
        res_sort = res.sort('id_col')
        assert ['id_col', 'val_col'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id_col': 1, 'val_col': 'a'} == res_sort[0]
        assert {'id_col': 2, 'val_col': 'b'} == res_sort[1]
        assert {'id_col': 3, 'val_col': 'c'} == res_sort[2]

    def test_save_as_parquet_rename(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        path = 'tmp/frame-parquet'
        t.save_as_parquet(path, column_names=['id1', 'val1'])
        res = XFrame(path, format='parquet')
        res_sort = res.sort('id1')
        assert ['id1', 'val1'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id1': 1, 'val1': 'a'} == res_sort[0]
        assert {'id1': 2, 'val1': 'b'} == res_sort[1]
        assert {'id1': 3, 'val1': 'c'} == res_sort[2]

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
        assert [1, 2, 3] == list(res)

    def test_select_column_val(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.select_column('val')
        assert ['a', 'b', 'c'] == list(res)

    def test_select_column_bad_name(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(ValueError) as exception_info:
            t.select_column('xx')
        exception_message = exception_info.value.args[0]
        assert "Column name does not exist: 'xx'." == exception_message

    # noinspection PyTypeChecker
    def test_select_column_bad_type(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            t.select_column(1)
        exception_message = exception_info.value.args[0]
        assert 'Invalid column_name type must be str.' == exception_message


# noinspection PyClassHasNoInit
class TestXFrameSelectColumns:
    """
    Tests XFrame select_columns
    """

    def test_select_columns_id_val(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        res = t.select_columns(['id', 'val'])
        assert {'id': 1, 'val': 'a'} == res[0]

    def test_select_columns_id(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        res = t.select_columns(['id'])
        assert {'id': 1} == res[0]

    # noinspection PyTypeChecker
    def test_select_columns_not_iterable(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        with pytest.raises(TypeError) as exception_info:
            t.select_columns(1)
        exception_message = exception_info.value.args[0]
        assert 'Keylist must be an iterable.' == exception_message

    def test_select_columns_bad_type(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        with pytest.raises(TypeError) as exception_info:
            t.select_columns(['id', 2])
        exception_message = exception_info.value.args[0]
        assert 'Invalid key type: must be str.' == exception_message

    def test_select_columns_bad_dup(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        with pytest.raises(ValueError) as exception_info:
            t.select_columns(['id', 'id'])
        exception_message = exception_info.value.args[0]
        assert "There are duplicate keys in key list: 'id'." == exception_message


# noinspection PyClassHasNoInit
class TestXFrameAddColumn:
    """
    Tests XFrame add_column
    """

    def test_add_column_named(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta = XArray([3.0, 2.0, 1.0])
        res = tf.add_column(ta, name='another')
        assert ['id', 'val', 'another'] == res.column_names()
        assert {'id': 1, 'val': 'a', 'another': 3.0} == res[0]

    def test_add_column_name_default(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta = XArray([3.0, 2.0, 1.0])
        res = tf.add_column(ta)
        assert ['id', 'val', 'X.2'] == res.column_names()
        assert {'id': 1, 'val': 'a', 'X.2': 3.0} == res[0]

    def test_add_column_name_dup(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta = XArray([3.0, 2.0, 1.0])
        res = tf.add_column(ta, name='id')
        assert ['id', 'val', 'id.2'] == res.column_names()
        assert {'id': 1, 'val': 'a', 'id.2': 3.0} == res[0]


# noinspection PyClassHasNoInit
class TestXFrameAddColumnsArray:
    """
    Tests XFrame add_columns where data is array of XArray
    """

    def test_add_columns_one(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta = XArray([3.0, 2.0, 1.0])
        res = tf.add_columns([ta], names=['new1'])
        assert ['id', 'val', 'new1'] == res.column_names()
        assert [int, str, float] == res.column_types()
        assert {'id': 1, 'val': 'a', 'new1': 3.0} == res[0]

    def test_add_columns_two(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta1 = XArray([3.0, 2.0, 1.0])
        ta2 = XArray([30.0, 20.0, 10.0])
        res = tf.add_columns([ta1, ta2], names=['new1', 'new2'])
        assert ['id', 'val', 'new1', 'new2'] == res.column_names()
        assert [int, str, float, float] == res.column_types()
        assert {'id': 1, 'val': 'a', 'new1': 3.0, 'new2': 30.0} == res[0]

    def test_add_columns_namelist_missing(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta1 = XArray([3.0, 2.0, 1.0])
        ta2 = XArray([30.0, 20.0, 10.0])
        with pytest.raises(TypeError) as exception_info:
            tf.add_columns([ta1, ta2])
        exception_message = exception_info.value.args[0]
        assert 'Namelist must be an iterable.' == exception_message

    # noinspection PyTypeChecker
    def test_add_columns_data_not_iterable(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            tf.add_columns(1, names=[])
        exception_message = exception_info.value.args[0]
        assert 'Column list must be an iterable.' == exception_message

    # noinspection PyTypeChecker
    def test_add_columns_namelist_not_iterable(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta1 = XArray([3.0, 2.0, 1.0])
        ta2 = XArray([30.0, 20.0, 10.0])
        with pytest.raises(TypeError) as exception_info:
            tf.add_columns([ta1, ta2], names=1)
        exception_message = exception_info.value.args[0]
        assert 'Namelist must be an iterable.' == exception_message

    def test_add_columns_not_xarray(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta1 = XArray([3.0, 2.0, 1.0])
        ta2 = [30.0, 20.0, 10.0]
        with pytest.raises(TypeError) as exception_info:
            tf.add_columns([ta1, ta2], names=['new1', 'new2'])
        exception_message = exception_info.value.args[0]
        assert 'Must give column as XArray.' == exception_message

    def test_add_columns_name_not_str(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta1 = XArray([3.0, 2.0, 1.0])
        ta2 = XArray([30.0, 20.0, 10.0])
        with pytest.raises(TypeError) as exception_info:
            tf.add_columns([ta1, ta2], names=['new1', 1])
        exception_message = exception_info.value.args[0]
        assert "Invalid column name in list : must all be str." == exception_message


# noinspection PyClassHasNoInit
class TestXFrameAddColumnsFrame:
    """
    Tests XFrame add_columns where data is XFrame
    """

    def test_add_columns(self):
        tf1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        tf2 = XFrame({'new1': [3.0, 2.0, 1.0], 'new2': [30.0, 20.0, 10.0]})
        res = tf1.add_columns(tf2)
        assert ['id', 'val', 'new1', 'new2'] == res.column_names()
        assert [int, str, float, float] == res.column_types()
        assert {'id': 1, 'val': 'a', 'new1': 3.0, 'new2': 30.0} == res[0]

    def test_add_columns_dup_names(self):
        tf1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        tf2 = XFrame({'new1': [3.0, 2.0, 1.0], 'val': [30.0, 20.0, 10.0]})
        res = tf1.add_columns(tf2)
        assert ['id', 'val', 'new1', 'val.1'] == res.column_names()
        assert [int, str, float, float] == res.column_types()
        assert {'id': 1, 'val': 'a', 'new1': 3.0, 'val.1': 30.0} == res[0]


# noinspection PyClassHasNoInit
class TestXFrameReplaceColumn:
    """
    Tests XFrame replace_column
    """

    def test_replace_column(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        a = XArray(['x', 'y', 'z'])
        res = t.replace_column('val', a)
        assert ['id', 'val'] == res.column_names()
        assert {'id': 1, 'val': 'x'} == res[0]

    # noinspection PyTypeChecker
    def test_replace_column_bad_col_type(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            t.replace_column('val', ['x', 'y', 'z'])
        exception_message = exception_info.value.args[0]
        assert 'Must give column as XArray.' == exception_message

    def test_replace_column_bad_name(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        a = XArray(['x', 'y', 'z'])
        with pytest.raises(ValueError) as exception_info:
            t.replace_column('xx', a)
        exception_message = exception_info.value.args[0]
        assert 'Column name must be in XFrame.' ==exception_message

    # noinspection PyTypeChecker
    def test_replace_column_bad_name_type(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        a = XArray(['x', 'y', 'z'])
        with pytest.raises(TypeError) as exception_info:
            t.replace_column(2, a)
        exception_message = exception_info.value.args[0]
        assert 'Invalid column name: must be str.' == exception_message


# noinspection PyClassHasNoInit
class TestXFrameRemoveColumn:
    """
    Tests XFrame remove_column
    """

    def test_remove_column(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        res = t.remove_column('another')
        assert res[0] == {'id': 1, 'val': 'a'}
        assert 3 == len(t.column_names())
        assert 2 == len(res.column_names())

    def test_remove_column_not_found(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        with pytest.raises(KeyError) as exception_info:
            t.remove_column('xx')
        exception_message = exception_info.value.args[0]
        assert "Cannot find column 'xx'." == exception_message

    def test_remove_column_many(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'new1': [3.0, 2.0, 1.0], 'new2': [30.0, 20.0, 10.0]})
        res = t.remove_column(['new1', 'new2'])
        assert {'id': 1, 'val': 'a'} ==res[0]
        assert 4 == len(t.column_names())
        assert 2 == len(res.column_names())

    def test_remove_column_many_bad_type(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        with pytest.raises(TypeError) as exception_info:
            t.remove_columns(10)
        exception_message = exception_info.value.args[0]
        assert 'Column_names must be an iterable.' == exception_message


# noinspection PyClassHasNoInit
class TestXFrameRemoveColumns:
    """
    Tests XFrame remove_columns
    """

    def test_remove_columns(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'new1': [3.0, 2.0, 1.0], 'new2': [30.0, 20.0, 10.0]})
        res = t.remove_columns(['new1', 'new2'])
        assert {'id': 1, 'val': 'a'} == res[0]
        assert 4 == len(t.column_names())
        assert 2 == len(res.column_names())

    def test_remove_column_not_iterable(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        with pytest.raises(TypeError) as exception_info:
            t.remove_columns(3)
        exception_message = exception_info.value.args[0]
        assert 'Column_names must be an iterable.' == exception_message

    def test_remove_column_not_found(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        with pytest.raises(KeyError) as exception_info:
            t.remove_columns(['xx'])
        exception_message = exception_info.value.args[0]
        assert "Cannot find column 'xx'." == exception_message


# noinspection PyClassHasNoInit
class TestXFrameSwapColumns:
    """
    Tests XFrame swap_columns
    """

    def test_swap_columns(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'x': [3.0, 2.0, 1.0]})
        res = t.swap_columns('val', 'x')
        assert ['id', 'x', 'val'] == res.column_names()
        assert ['id', 'val', 'x'] == t.column_names()
        assert {'id': 1, 'x': 3.0, 'val': 'a'} == res[0]

    def test_swap_columns_bad_col_1(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        with pytest.raises(KeyError) as exception_info:
            t.swap_columns('xx', 'another')
        exception_message = exception_info.value.args[0]
        assert "Cannot find column 'xx'." == exception_message

    def test_swap_columns_bad_col_2(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        with pytest.raises(KeyError) as exception_info:
            t.swap_columns('val', 'xx')
        exception_message = exception_info.value.args[0]
        assert "Cannot find column 'xx'." == exception_message


# noinspection PyClassHasNoInit
class TestXFrameReorderColumns:
    """
    Tests XFrame reorder_columns
    """

    def test_reorder_columns(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'x': [3.0, 2.0, 1.0]})
        res = t.reorder_columns(['val', 'x', 'id'])
        assert ['val', 'x', 'id'] == res.column_names()
        assert ['id', 'val', 'x'] == t.column_names()
        assert {'id': 1, 'x': 3.0, 'val': 'a'} == res[0]

    # noinspection PyTypeChecker
    def test_reorder_columns_list_not_iterable(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'x': [3.0, 2.0, 1.0]})
        with pytest.raises(TypeError) as exception_info:
            t.reorder_columns(3)
        exception_message = exception_info.value.args[0]
        assert 'Keylist must be an iterable.' == exception_message

    def test_reorder_columns_bad_col(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'x': [3.0, 2.0, 1.0]})
        with pytest.raises(KeyError) as exception_info:
            t.reorder_columns(['val', 'y', 'id'])
        exception_message = exception_info.value.args[0]
        assert "Cannot find column 'y'." == exception_message

    def test_reorder_columns_incomplete(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'x': [3.0, 2.0, 1.0]})
        with pytest.raises(KeyError) as exception_info:
            t.reorder_columns(['val', 'id'])
        exception_message = exception_info.value.args[0]
        assert "Column 'x' not assigned'." == exception_message


# noinspection PyClassHasNoInit
class TestXFrameRename:
    """
    Tests XFrame rename
    """

    def test_rename(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.rename({'id': 'new_id'})
        assert ['new_id', 'val'] == res.column_names()
        assert ['id', 'val'] == t.column_names()
        assert {'new_id': 1, 'val': 'a'} == res[0]

    # noinspection PyTypeChecker
    def test_rename_arg_not_dict(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            t.rename('id')
        exception_message = exception_info.value.args[0]
        assert 'Names must be a dictionary: oldname -> newname or a list of newname (str).' == exception_message

    def test_rename_col_not_found(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(ValueError) as exception_info:
            t.rename({'xx': 'new_id'})
        exception_message = exception_info.value.args[0]
        assert "Cannot find column 'xx' in the XFrame." == exception_message

    def test_rename_bad_length(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(ValueError) as exception_info:
            t.rename(['id'])
        exception_message = exception_info.value.args[0]
        assert 'Names must be the same length as the number of columns (names: 1 columns: 2).' == exception_message

    def test_rename_list(self):
        t = XFrame({'X.0': [1, 2, 3], 'X.1': ['a', 'b', 'c']})
        res = t.rename(['id', 'val'])
        assert ['id', 'val'] == res.column_names()
        assert ['X.0', 'X.1'] == t.column_names()
        assert {'id': 1, 'val': 'a'} == res[0]


# noinspection PyClassHasNoInit
class TestXFrameGetitem:
    """
    Tests XFrame __getitem__
    """

    def test_getitem_str(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t['id']
        assert [1, 2, 3] == list(res)

    def test_getitem_int_pos(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t[1]
        assert {'id': 2, 'val': 'b'} == res

    def test_getitem_int_neg(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t[-2]
        assert {'id': 2, 'val': 'b'} == res

    def test_getitem_int_too_low(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(IndexError) as exception_info:
            _ = t[-100]
        exception_message = exception_info.value.args[0]
        assert 'XFrame index out of range (too low).' == exception_message

    def test_getitem_int_too_high(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(IndexError) as exception_info:
            _ = t[100]
        exception_message = exception_info.value.args[0]
        assert 'XFrame index out of range (too high).' == exception_message

    def test_getitem_slice(self):
        # TODO we could test more variations of slice
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t[:2]
        assert 2 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]

    def test_getitem_list(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'x': [1.0, 2.0, 3.0]})
        res = t[['id', 'x']]
        assert {'id': 2, 'x': 2.0} == res[1]

    def test_getitem_bad_type(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            _ = t[{'a': 1}]
        exception_message = exception_info.value.args[0]
        assert "Invalid index type: must be XArray, 'int', 'list', slice, or 'str': (dict)." == exception_message

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
        assert ['id', 'val', 'x'] == t.column_names()
        assert {'id': 2, 'val': 'b', 'x': 5.0} == t[1]

    def test_setitem_str_const_replace(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t['val'] = 'x'
        assert ['id', 'val'] == t.column_names()
        assert {'id': 1, 'val': 'x'} == t[0]
        assert {'id': 2, 'val': 'x'} == t[1]
        assert {'id': 3, 'val': 'x'} == t[2]

    def test_setitem_list(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta1 = XArray([3.0, 2.0, 1.0])
        ta2 = XArray([30.0, 20.0, 10.0])
        tf[['new1', 'new2']] = [ta1, ta2]
        assert ['id', 'val', 'new1', 'new2'] == tf.column_names()
        assert [int, str, float, float] == tf.column_types()
        assert {'id': 1, 'val': 'a', 'new1': 3.0, 'new2': 30.0} == tf[0]

    def test_setitem_str_iter(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t['x'] = [1.0, 2.0, 3.0]
        assert ['id', 'val', 'x'] == t.column_names()
        assert {'id': 2, 'val': 'b', 'x': 2.0} == t[1]

    def test_setitem_str_xarray(self):
        tf = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        ta = XArray([3.0, 2.0, 1.0])
        tf['new'] = ta
        assert ['id', 'val', 'new'] == tf.column_names()
        assert [int, str, float] == tf.column_types()
        assert {'id': 1, 'val': 'a', 'new': 3.0} == tf[0]

    def test_setitem_str_iter_replace(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t['val'] = [1.0, 2.0, 3.0]
        assert ['id', 'val'] == t.column_names()
        assert {'id': 1, 'val': 1.0} == t[0]
        assert {'id': 2, 'val': 2.0} == t[1]
        assert {'id': 3, 'val': 3.0} == t[2]

    def test_setitem_bad_key(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            t[{'a': 1}] = [1.0, 2.0, 3.0]
        exception_message = exception_info.value.args[0]
        assert 'Cannot set column with key type dict.' == exception_message

    def test_setitem_str_iter_replace_one_col(self):
        t = XFrame({'val': ['a', 'b', 'c']})
        t['val'] = [1.0, 2.0, 3.0, 4.0]
        assert ['val'] ==  t.column_names()
        assert 4 == len(t)
        assert {'val': 2.0} == t[1]


# noinspection PyClassHasNoInit
class TestXFrameDelitem:
    """
    Tests XFrame __delitem__
    """

    def test_delitem(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        del t['another']
        assert {'id': 1, 'val': 'a'} == t[0]

    def test_delitem_not_found(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c'], 'another': [3.0, 2.0, 1.0]})
        with pytest.raises(KeyError) as exception_info:
            del t['xx']
        exception_message = exception_info.value.args[0]
        assert "Cannot find column 'xx'." == exception_message


# noinspection PyClassHasNoInit
class TestXFrameIsMaterialized:
    """
    Tests XFrame _is_materialized
    """

    def test_is_materialized_false(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        assert False is t._is_materialized()

    def test_is_materialized(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        len(t)
        assert True is t._is_materialized()


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
        assert {'id': 2, 'val': 'b'} == res

    def test_range_int_neg(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.range(-2)
        assert {'id': 2, 'val': 'b'} == res

    def test_range_int_too_low(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(IndexError) as exception_info:
            _ = t.range(-100)
        exception_message = exception_info.value.args[0]
        assert 'XFrame index out of range (too low).' == exception_message

    def test_range_int_too_high(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(IndexError) as exception_info:
            _ = t.range(100)
        exception_message = exception_info.value.args[0]
        assert 'XFrame index out of range (too high).' == exception_message

    def test_range_slice(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.range(slice(0, 2))
        assert 2 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]

    # noinspection PyTypeChecker
    def test_range_bad_type(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            _ = t.range({'a': 1})
        exception_message = exception_info.value.args[0]
        assert 'Invalid argument type: must be int or slice (dict).' == exception_message


# noinspection PyClassHasNoInit
class TestXFrameAppend:
    """
    Tests XFrame append
    """

    def test_append(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [10, 20, 30], 'val': ['aa', 'bb', 'cc']})
        res = t1.append(t2)
        assert 6 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 10, 'val': 'aa'} == res[3]

    # noinspection PyTypeChecker
    def test_append_bad_type(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(RuntimeError) as exception_info:
            t1.append(1)
        exception_message = exception_info.value.args[0]
        assert 'XFrame append can only work with XFrame.' == exception_message

    def test_append_both_empty(self):
        t1 = XFrame()
        t2 = XFrame()
        res = t1.append(t2)
        assert 0 == len(res)

    def test_append_first_empty(self):
        t1 = XFrame()
        t2 = XFrame({'id': [10, 20, 30], 'val': ['aa', 'bb', 'cc']})
        res = t1.append(t2)
        assert 3 == len(res)
        assert {'id': 10, 'val': 'aa'} == res[0]

    def test_append_second_empty(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame()
        res = t1.append(t2)
        assert 3 == len(res)
        assert {'id': 1, 'val': 'a'} ==res[0]

    def test_append_unequal_col_length(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [10, 20, 30], 'val': ['aa', 'bb', 'cc'], 'another': [1.0, 2.0, 3.0]})
        with pytest.raises(RuntimeError) as exception_info:
            t1.append(t2)
        exception_message = exception_info.value.args[0]
        assert 'Two XFrames must have the same number of columns.' == exception_message

    def test_append_col_name_mismatch(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [10, 20, 30], 'xx': ['a', 'b', 'c']})
        with pytest.raises(RuntimeError) as exception_info:
            t1.append(t2)
        exception_message = exception_info.value.args[0]
        assert 'Column val name is not the same in two XFrames, one is val the other is xx.' == exception_message

    def test_append_col_type_mismatch(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [10, 20], 'val': [1.0, 2.0]})
        with pytest.raises(RuntimeError) as exception_info:
            t1.append(t2)
        exception_message = exception_info.value.args[0]
        assert "Column val type is not the same in two XFrames, " \
               "one is <class 'str'> the other is [<class 'int'>, <class 'float'>]." == exception_message


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
        assert 3 == len(res)
        assert ['id'] == res.column_names()
        assert [int] == res.column_types()
        assert {'id': 1} == res[0]
        assert {'id': 2} == res[1]
        assert {'id': 3} == res[2]

    def test_groupby_nooperation(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id')
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id'] == res.column_names()
        assert [int] == res.column_types()
        assert {'id': 1} == res[0]
        assert {'id': 2} == res[1]
        assert {'id': 3} == res[2]

    # noinspection PyTypeChecker
    def test_groupby_bad_col_name_type(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        with pytest.raises(TypeError) as exception_info:
            t.groupby(1, {})
        exception_message = exception_info.value.args[0]
        assert "'int' object is not iterable" == exception_message

    # noinspection PyTypeChecker
    def test_groupby_bad_col_name_list_type(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        with pytest.raises(TypeError) as exception_info:
            t.groupby([1], {})
        exception_message = exception_info.value.args[0]
        assert 'Column name must be a string.' == exception_message

    def test_groupby_bad_col_group_name(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        with pytest.raises(KeyError) as exception_info:
            t.groupby('xx', {})
        exception_message = exception_info.value.args[0]
        assert "Column 'xx' does not exist in XFrame." ==exception_message

    def test_groupby_bad_group_type(self):
        t = XFrame({'id': [{1: 'a', 2: 'b'}, {3: 'c'}],
                    'val': ['a', 'b']})
        with pytest.raises(TypeError) as exception_info:
            t.groupby('id', {})
        exception_message = exception_info.value.args[0]
        assert 'Cannot group on a dictionary column.' == exception_message

    def test_groupby_bad_agg_group_name(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        with pytest.raises(KeyError) as exception_info:
            t.groupby('id', SUM('xx'))
        exception_message = exception_info.value.args[0]
        assert "Column 'xx' does not exist in XFrame." == exception_message

    def test_groupby_bad_agg_group_type(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        with pytest.raises(TypeError) as exception_info:
            t.groupby('id', SUM(1))
        exception_message = exception_info.value.args[0]
        assert 'Column name must be a string.' == exception_message


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
        assert 3 == len(res)
        assert ['id', 'count'] == res.column_names()
        assert [int, int] == res.column_types()
        assert {'id': 1, 'count': 3} == res[0]
        assert {'id': 2, 'count': 2} == res[1]
        assert {'id': 3, 'count': 1} == res[2]

    def test_groupby_count_call(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', COUNT())
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'count'] == res.column_names()
        assert [int, int] == res.column_types()
        assert {'id': 1, 'count': 3} == res[0]
        assert {'id': 2, 'count': 2} == res[1]
        assert {'id': 3, 'count': 1} == res[2]

    def test_groupby_count_named(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', {'record-count': COUNT})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'record-count'] == res.column_names()
        assert [int, int] == res.column_types()
        assert {'id': 1, 'record-count': 3} == res[0]
        assert {'id': 2, 'record-count': 2} ==res[1]
        assert {'id': 3, 'record-count': 1} == res[2]

    def test_groupby_sum(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', {'sum': SUM('another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'sum'] == res.column_names()
        assert [int, int] == res.column_types()
        assert {'id': 1, 'sum': 110} == res[0]
        assert {'id': 2, 'sum': 70} == res[1]
        assert {'id': 3, 'sum': 30} == res[2]

    def test_groupby_sum_def(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', SUM('another'))
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'sum'] == res.column_names()
        assert [int, int] == res.column_types()
        assert {'id': 1, 'sum': 110} == res[0]
        assert {'id': 2, 'sum': 70} == res[1]
        assert {'id': 3, 'sum': 30} == res[2]

    def test_groupby_sum_sum_def(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', [SUM('another'), SUM('another')])
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'sum', 'sum.1'] == res.column_names()
        assert [int, int, int] == res.column_types()
        assert {'id': 1, 'sum': 110, 'sum.1': 110} == res[0]
        assert {'id': 2, 'sum': 70, 'sum.1': 70} == res[1]
        assert {'id': 3, 'sum': 30, 'sum.1': 30} == res[2]

    def test_groupby_sum_rename(self):
        t = XFrame({'sum': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('sum', SUM('another'))
        res = res.topk('sum', reverse=True)
        assert 3 == len(res)
        assert ['sum', 'sum.1'] == res.column_names()
        assert [int, int] == res.column_types()
        assert {'sum': 1, 'sum.1': 110} == res[0]
        assert {'sum': 2, 'sum.1': 70} == res[1]
        assert {'sum': 3, 'sum.1': 30} == res[2]

    def test_groupby_count_sum(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', {'count': COUNT, 'sum': SUM('another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['count', 'id', 'sum'] == sorted(res.column_names())
        assert [int, int, int] == res.column_types()
        assert {'id': 1, 'count': 3, 'sum': 110} == res[0]
        assert {'id': 2, 'count': 2, 'sum': 70} == res[1]
        assert {'id': 3, 'count': 1, 'sum': 30} == res[2]

    def test_groupby_count_sum_def(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', [COUNT, SUM('another')])
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'count', 'sum'] == res.column_names()
        assert [int, int, int] == res.column_types()
        assert {'id': 1, 'count': 3, 'sum': 110} == res[0]
        assert {'id': 2, 'count': 2, 'sum': 70} == res[1]
        assert {'id': 3, 'count': 1, 'sum': 30} == res[2]

    def test_groupby_argmax(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', ARGMAX('another', 'val'))
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'argmax'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'argmax': 'f'} == res[0]
        assert {'id': 2, 'argmax': 'e'} == res[1]
        assert {'id': 3, 'argmax': 'c'} ==res[2]

    def test_groupby_argmin(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', ARGMIN('another', 'val'))
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'argmin'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'argmin': 'a'} == res[0]
        assert {'id': 2, 'argmin': 'b'} == res[1]
        assert {'id': 3, 'argmin': 'c'} == res[2]

    def test_groupby_max(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', MAX('another'))
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'max'] == res.column_names()
        assert [int, int] == res.column_types()
        assert {'id': 1, 'max': 60} == res[0]
        assert {'id': 2, 'max': 50} == res[1]
        assert {'id': 3, 'max': 30} == res[2]

    def test_groupby_max_float(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', MAX('another'))
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'max'] == res.column_names()
        assert [int, float] == res.column_types()
        assert {'id': 1, 'max': 60.0} == res[0]
        assert {'id': 2, 'max': 50.0} == res[1]
        assert {'id': 3, 'max': 30.0} == res[2]

    def test_groupby_max_str(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', MAX('val'))
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'max'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'max': 'f'} == res[0]
        assert {'id': 2, 'max': 'e'} == res[1]
        assert {'id': 3, 'max': 'c'} == res[2]

    def test_groupby_min(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', MIN('another'))
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'min'] == res.column_names()
        assert [int, int] == res.column_types()
        assert {'id': 1, 'min': 10} == res[0]
        assert {'id': 2, 'min': 20} == res[1]
        assert {'id': 3, 'min': 30} == res[2]

    def test_groupby_min_float(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', MIN('another'))
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'min'] == res.column_names()
        assert [int, float] == res.column_types()
        assert {'id': 1, 'min': 10.0} == res[0]
        assert {'id': 2, 'min': 20.0} == res[1]
        assert {'id': 3, 'min': 30.0} == res[2]

    def test_groupby_min_str(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', MIN('val'))
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'min'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'min': 'a'} == res[0]
        assert {'id': 2, 'min': 'b'} == res[1]
        assert {'id': 3, 'min': 'c'} == res[2]

    def test_groupby_mean(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', MEAN('another'))
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'mean'] == res.column_names()
        assert [int, float] == res.column_types()
        assert {'id': 1, 'mean': 110.0 / 3.0} == res[0]
        assert {'id': 2, 'mean': 70.0 / 2.0} == res[1]
        assert {'id': 3, 'mean': 30} == res[2]

    def test_groupby_variance(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', VARIANCE('another'))
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'variance'] == res.column_names()
        assert [int, float] == res.column_types()
        assert almost_equal(3800.0 / 9.0, res[0]['variance'])
        assert almost_equal(225.0, res[1]['variance'])
        assert {'id': 3, 'variance': 0.0} == res[2]

    def test_groupby_stdv(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', STDV('another'))
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'stdv'] == res.column_names()
        assert [int, float] == res.column_types()
        assert almost_equal(math.sqrt(3800.0 / 9.0), res[0]['stdv'])
        assert almost_equal(math.sqrt(225.0), res[1]['stdv'])
        assert {'id': 3, 'stdv': 0.0} == res[2] == res[2]

    def test_groupby_select_one(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', SELECT_ONE('another'))
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'select-one'] == res.column_names()
        assert res.column_types() == [int, int]
        assert {'id': 1, 'select-one': 40} == res[0]
        assert {'id': 2, 'select-one': 50} == res[1]
        assert {'id': 3, 'select-one': 30} == res[2]

    def test_groupby_select_one_float(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', SELECT_ONE('another'))
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'select-one'] == res.column_names()
        assert [int, float] == res.column_types()
        assert {'id': 1, 'select-one': 40.0} == res[0]
        assert {'id': 2, 'select-one': 50.0} == res[1]
        assert {'id': 3, 'select-one': 30.0} == res[2]

    def test_groupby_select_one_str(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', SELECT_ONE('val'))
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'select-one'] == sorted(res.column_names())
        assert [int, str] == res.column_types()
        assert {'id': 1, 'select-one': 'a'} == res[0]
        assert {'id': 2, 'select-one': 'b'} == res[1]
        assert {'id': 3, 'select-one': 'c'} == res[2]

    def test_groupby_concat_list(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', CONCAT('another'))
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'concat'] == res.column_names()
        assert [int, list] == res.column_types()
        assert {'id': 1, 'concat': [10, 40, 60]} == res[0]
        assert {'id': 2, 'concat': [20, 50]} == res[1]
        assert {'id': 3, 'concat': [30]} == res[2]

    def test_groupby_concat_dict(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', CONCAT('val', 'another'))
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'concat'] == res.column_names()
        assert [int, dict] == res.column_types()
        assert {'id': 1, 'concat': {'a': 10, 'd': 40, 'f': 60}} == res[0]
        assert {'id': 2, 'concat': {'b': 20, 'e': 50}} == res[1]
        assert {'id': 3, 'concat': {'c': 30}} == res[2]

    def test_groupby_values_list(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
                    'another': [10, 20, 30, 40, 50, 60, 10]})
        res = t.groupby('id', VALUES('another'))
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'values'] == res.column_names()
        assert [int, list] == res.column_types()
        assert dict_keys_equal({'id': 1, 'values': [10, 40, 60]}, res[0])
        assert dict_keys_equal({'id': 2, 'values': [20, 50]}, res[2])
        assert {'id': 3, 'values': [30]} == res[2]

    def test_groupby_values_count_list(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
                    'another': [10, 20, 30, 40, 50, 60, 10]})
        res = t.groupby('id', VALUES_COUNT('another'))
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'values-count'] == res.column_names()
        assert [int, dict] == res.column_types()
        assert dict_keys_equal({'id': 1, 'values-count': {10: 2, 40: 1, 60: 1}}, res[0])
        assert dict_keys_equal({'id': 2, 'values-count': {20: 1, 50: 1}}, res[2])
        assert {'id': 3, 'values-count': {30: 1}} == res[2]

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
        # Because of the None, you cannot take topk on this xframe
        expect_map = {
            1: 2,
            2: 1,
            3: 1,
            None: 2
        }
        assert 4 == len(res)
        assert ['id', 'count'] == res.column_names()
        assert [int, int] == res.column_types()
        for row in res:
            assert expect_map[row['id']] == row['count']

    def test_groupby_sum(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, None, None, 60]})
        res = t.groupby('id', SUM('another'))
#        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'sum'] == res.column_names()
        assert [int, int] == res.column_types()
        expect_map = {
            1: 70,
            2: 20,
            3: 30,
        }
        for row in res:
            assert expect_map[row['id']] == row['sum']

    def test_groupby_argmax(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, 40, None, None]})
        res = t.groupby('id', {'argmax': ARGMAX('another', 'val')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'argmax'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'argmax': 'd'} == res[0]
        assert {'id': 2, 'argmax': 'b'} == res[1]
        assert {'id': 3, 'argmax': 'c'} == res[2]

    def test_groupby_argmin(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, None, 30, 40, 50, 60]})
        res = t.groupby('id', {'argmin': ARGMIN('another', 'val')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'argmin'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'argmin': 'a'} == res[0]
        assert {'id': 2, 'argmin': 'e'} == res[1]
        assert {'id': 3, 'argmin': 'c'} == res[2]

    def test_groupby_max(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, None, None, 60]})
        res = t.groupby('id', {'max': MAX('another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'max'] == res.column_names()
        assert [int, int] == res.column_types()
        assert {'id': 1, 'max': 60} == res[0]
        assert {'id': 2, 'max': 20} == res[1]
        assert {'id': 3, 'max': 30} == res[2]

    def test_groupby_max_float(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10.0, 20.0, 30.0, float('nan'), float('nan'), 60.0]})
        res = t.groupby('id', {'max': MAX('another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'max'] == res.column_names()
        assert [int, float] == res.column_types()
        assert {'id': 1, 'max': 60.0} == res[0]
        assert {'id': 2, 'max': 20.0} == res[1]
        assert {'id': 3, 'max': 30.0} == res[2]

    def test_groupby_max_str(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', None, None, 'f'],
                    'another': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', {'max': MAX('val')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'max'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'max': 'f'} == res[0]
        assert {'id': 2, 'max': 'b'} == res[1]
        assert {'id': 3, 'max': 'c'} == res[2]

    def test_groupby_min(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [None, None, 30, 40, 50, 60]})
        res = t.groupby('id', {'min': MIN('another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'min'] == res.column_names()
        assert [int, int] == res.column_types()
        assert {'id': 1, 'min': 40} == res[0]
        assert {'id': 2, 'min': 50} == res[1]
        assert {'id': 3, 'min': 30} == res[2]

    def test_groupby_min_float(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [None, None, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', {'min': MIN('another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'min'] == res.column_names()
        assert [int, float] == res.column_types()
        assert {'id': 1, 'min': 40.0} == res[0]
        assert {'id': 2, 'min': 50.0} == res[1]
        assert {'id': 3, 'min': 30.0} == res[2]

    def test_groupby_min_str(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': [None, None, 'c', 'd', 'e', 'f'],
                    'another': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', {'min': MIN('val')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'min'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'min': 'd'} == res[0]
        assert {'id': 2, 'min': 'e'} == res[1]
        assert {'id': 3, 'min': 'c'} == res[2]

    def test_groupby_mean(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, None, None, 60]})
        res = t.groupby('id', {'mean': MEAN('another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'mean'] == res.column_names()
        assert [int, float] == res.column_types()
        assert {'id': 1, 'mean': 70.0 / 2.0} == res[0]
        assert {'id': 2, 'mean': 20.0} == res[1]
        assert {'id': 3, 'mean': 30.0} == res[2]

    def test_groupby_variance(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, None, None, 60]})
        res = t.groupby('id', {'variance': VARIANCE('another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'variance'] == res.column_names()
        assert [int, float] == res.column_types()
        assert almost_equal(2500.0 / 4.0, res[0]['variance'])
        assert almost_equal(0.0, res[1]['variance'])
        assert {'id': 3, 'variance': 0.0} == res[2]

    def test_groupby_stdv(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, None, None, 60]})
        res = t.groupby('id', {'stdv': STDV('another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'stdv'] == res.column_names()
        assert [int, float] == res.column_types()
        assert almost_equal(math.sqrt(2500.0 / 4.0), res[0]['stdv'])
        assert almost_equal(0.0, res[1]['stdv'])
        assert {'id': 3, 'stdv': 0.0} == res[2]

    def test_groupby_select_one(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, None, None, 60]})
        res = t.groupby('id', {'select_one': SELECT_ONE('another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'select_one'] == res.column_names()
        assert [int, int] == res.column_types()
        assert {'id': 1, 'select_one': 60} == res[0]
        assert {'id': 2, 'select_one': 20} == res[1]
        assert {'id': 3, 'select_one': 30} == res[2]

    def test_groupby_concat_list(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, 30, None, None, 60]})
        res = t.groupby('id', {'concat': CONCAT('another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'concat'] == res.column_names()
        assert [int, list] == res.column_types()
        assert {'id': 1, 'concat': [10, 60]} == res[0]
        assert {'id': 2, 'concat': [20]} == res[1]
        assert {'id': 3, 'concat': [30]} == res[2]

    def test_groupby_concat_dict(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', None, None, 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', {'concat': CONCAT('val', 'another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'concat'] == res.column_names()
        assert [int, dict] == res.column_types()
        assert {'id': 1, 'concat': {'a': 10, 'f': 60}} == res[0]
        assert {'id': 2, 'concat': {'b': 20}} == res[1]
        assert {'id': 3, 'concat': {'c': 30}} == res[2]


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
        assert 3 == len(res)
        assert ['id', 'count'] == res.column_names()
        assert [int, int] == res.column_types()
        expect_map = {
            1: 2,
            2: 1,
            None: 3,
        }
        for row in res:
            assert expect_map[row['id']] == row['count']

    def test_groupby_sum(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, None, None, None, 60]})
        res = t.groupby('id', SUM('another'))
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'sum'] == res.column_names()
        assert [int, int] == res.column_types()
        assert {'id': 1, 'sum': 70} == res[0]
        assert {'id': 2, 'sum': 20} == res[1]
        assert {'id': 3, 'sum': 0} == res[2]

    def test_groupby_argmax(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, None, 40, None, None]})
        res = t.groupby('id', {'argmax': ARGMAX('another', 'val')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'argmax'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'argmax': 'd'} == res[0]
        assert {'id': 2, 'argmax': 'b'} == res[1]
        assert {'id': 3, 'argmax': None} == res[2]

    def test_groupby_argmin(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, None, None, 40, 50, 60]})
        res = t.groupby('id', {'argmin': ARGMIN('another', 'val')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'argmin'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'argmin': 'a'} == res[0]
        assert {'id': 2, 'argmin': 'e'} == res[1]
        assert {'id': 3, 'argmin': None} == res[2]

    def test_groupby_max(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, None, None, None, 60]})
        res = t.groupby('id', {'max': MAX('another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
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
        assert 3 == len(res)
        assert ['id', 'max'] == res.column_names()
        assert [int, float] == res.column_types()
        assert {'id': 1, 'max': 60.0} == res[0]
        assert {'id': 2, 'max': 20.0} == res[1]
        assert {'id': 3, 'max': None} == res[2]

    def test_groupby_max_str(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', None, None, None, 'f'],
                    'another': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', {'max': MAX('val')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'max'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'max': 'f'} == res[0]
        assert {'id': 2, 'max': 'b'} == res[1]
        assert {'id': 3, 'max': None} == res[2]

    def test_groupby_min(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [None, None, None, 40, 50, 60]})
        res = t.groupby('id', {'min': MIN('another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'min'] == res.column_names()
        assert [int, int] == res.column_types()
        assert {'id': 1, 'min': 40} == res[0]
        assert {'id': 2, 'min': 50} == res[1]
        assert {'id': 3, 'min': None} == res[2]

    def test_groupby_min_float(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [None, None, None, 40.0, 50.0, 60.0]})
        res = t.groupby('id', {'min': MIN('another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'min'] == res.column_names()
        assert [int, float] == res.column_types()
        assert {'id': 1, 'min': 40.0} == res[0]
        assert {'id': 2, 'min': 50.0} == res[1]
        assert {'id': 3, 'min': None} == res[2]

    def test_groupby_min_str(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': [None, None, None, 'd', 'e', 'f'],
                    'another': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        res = t.groupby('id', {'min': MIN('val')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'min'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'min': 'd'} == res[0]
        assert {'id': 2, 'min': 'e'} == res[1]
        assert {'id': 3, 'min': None} == res[2]

    def test_groupby_mean(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, None, None, None, 60]})
        res = t.groupby('id', {'mean': MEAN('another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'mean'] == res.column_names()
        assert [int, float] == res.column_types()
        assert {'id': 1, 'mean': 70.0 / 2.0} == res[0]
        assert {'id': 2, 'mean': 20.0} == res[1]
        assert {'id': 3, 'mean': None} == res[2]

    def test_groupby_variance(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, None, None, None, 60]})
        res = t.groupby('id', {'variance': VARIANCE('another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'variance'] == res.column_names()
        assert [int, float] == res.column_types()
        assert almost_equal(2500.0 / 4.0, res[0]['variance'])
        assert almost_equal(0.0, res[1]['variance'])
        assert {'id': 3, 'variance': None} == res[2]

    def test_groupby_stdv(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, None, None, None, 60]})
        res = t.groupby('id', {'stdv': STDV('another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'stdv'] == res.column_names()
        assert [int, float] == res.column_types()
        assert almost_equal(math.sqrt(2500.0 / 4.0), res[0]['stdv'])
        assert almost_equal(math.sqrt(0.0), res[1]['stdv'])
        assert {'id': 3, 'stdv': None} == res[2]

    def test_groupby_select_one(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, None, None, None, 60]})
        res = t.groupby('id', {'select_one': SELECT_ONE('another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'select_one'] == res.column_names()
        assert [int, int] == res.column_types()
        assert {'id': 1, 'select_one': 60} == res[0]
        assert {'id': 2, 'select_one': 20} == res[1]
        assert {'id': 3, 'select_one': None} == res[2]

    def test_groupby_concat_list(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', 'c', 'd', 'e', 'f'],
                    'another': [10, 20, None, None, None, 60]})
        res = t.groupby('id', {'concat': CONCAT('another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'concat'] == res.column_names()
        assert [int, list] == res.column_types()
        assert {'id': 1, 'concat': [10, 60]} == res[0]
        assert {'id': 2, 'concat': [20]} == res[1]
        assert {'id': 3, 'concat': []} == res[2]

    def test_groupby_concat_dict(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1],
                    'val': ['a', 'b', None, None, None, 'f'],
                    'another': [10, 20, 30, 40, 50, 60]})
        res = t.groupby('id', {'concat': CONCAT('val', 'another')})
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'concat'] == res.column_names()
        assert [int, dict] == res.column_types()
        assert {'id': 1, 'concat': {'a': 10, 'f': 60}} == res[0]
        assert {'id': 2, 'concat': {'b': 20}} == res[1]
        assert {'id': 3, 'concat': {}} == res[2]


# noinspection PyClassHasNoInit
class TestXFrameJoin:
    """
    Tests XFrame join
    """

    def test_join(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 3], 'doubled': ['aa', 'bb', 'cc']})
        res = t1.join(t2).sort('id').head()
        assert 3 == len(res)
        assert ['id', 'val', 'doubled'] == res.column_names()
        assert res.column_types() == [int, str, str]
        assert {'id': 1, 'val': 'a', 'doubled': 'aa'} == res[0]
        assert {'id': 2, 'val': 'b', 'doubled': 'bb'} == res[1]
        assert {'id': 3, 'val': 'c', 'doubled': 'cc'} == res[2]

    def test_join_rename(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 3], 'val': ['aa', 'bb', 'cc']})
        res = t1.join(t2, on='id').sort('id').head()
        assert 3 == len(res)
        assert ['id', 'val', 'val.1'] == res.column_names()
        assert [int, str, str] == res.column_types()
        assert {'id': 1, 'val': 'a', 'val.1': 'aa'} == res[0]
        assert {'id': 2, 'val': 'b', 'val.1': 'bb'} == res[1]
        assert {'id': 3, 'val': 'c', 'val.1': 'cc'} == res[2]

    def test_join_compound_key(self):
        t1 = XFrame({'id1': [1, 2, 3], 'id2': [10, 20, 30], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id1': [1, 2, 3], 'id2': [10, 20, 30], 'doubled': ['aa', 'bb', 'cc']})
        res = t1.join(t2).sort('id1').head()
        assert 3 == len(res)
        assert ['id1', 'id2', 'val', 'doubled'] == res.column_names()
        assert [int, int, str, str] == res.column_types()
        assert {'id1': 1, 'id2': 10, 'val': 'a', 'doubled': 'aa'} == res[0]
        assert {'id1': 2, 'id2': 20, 'val': 'b', 'doubled': 'bb'} == res[1]
        assert {'id1': 3, 'id2': 30, 'val': 'c', 'doubled': 'cc'} == res[2]

    def test_join_dict_key(self):
        t1 = XFrame({'id1': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id2': [1, 2, 3], 'doubled': ['aa', 'bb', 'cc']})
        res = t1.join(t2, on={'id1': 'id2'}).sort('id1').head()
        assert 3 == len(res)
        assert ['id1', 'val', 'doubled'] == res.column_names()
        assert [int, str, str] == res.column_types()
        assert {'id1': 1, 'val': 'a', 'doubled': 'aa'} == res[0]
        assert {'id1': 2, 'val': 'b', 'doubled': 'bb'} == res[1]
        assert {'id1': 3, 'val': 'c', 'doubled': 'cc'} == res[2]

    def test_join_partial(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 4], 'doubled': ['aa', 'bb', 'cc']})
        res = t1.join(t2).sort('id').head()
        assert 2 == len(res)
        assert ['id', 'val', 'doubled'] == res.column_names()
        assert [int, str, str] == res.column_types()
        assert {'id': 1, 'val': 'a', 'doubled': 'aa'} == res[0]
        assert {'id': 2, 'val': 'b', 'doubled': 'bb'} == res[1]

    def test_join_empty(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [4, 5, 6], 'doubled': ['aa', 'bb', 'cc']})
        res = t1.join(t2).head()
        assert 0 == len(res)
        assert ['id', 'val', 'doubled'] == res.column_names()
        assert [int, str, str] == res.column_types()

    def test_join_on_val(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [10, 20, 30], 'val': ['a', 'b', 'c']})
        res = t1.join(t2, on='val').sort('id').head()
        assert 3 == len(res)
        assert ['id', 'val', 'id.1'] ==res.column_names()
        assert [int, str, int] == res.column_types()
        assert {'id': 1, 'val': 'a', 'id.1': 10} == res[0]
        assert {'id': 2, 'val': 'b', 'id.1': 20} == res[1]
        assert {'id': 3, 'val': 'c', 'id.1': 30} == res[2]

    def test_join_inner(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 4], 'doubled': ['aa', 'bb', 'cc']})
        res = t1.join(t2, how='inner').sort('id').head()
        assert 2 == len(res)
        assert ['id', 'val', 'doubled'] == res.column_names()
        assert [int, str, str] == res.column_types()
        assert {'id': 1, 'val': 'a', 'doubled': 'aa'} == res[0]
        assert {'id': 2, 'val': 'b', 'doubled': 'bb'} == res[1]

    def test_join_left(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 4], 'doubled': ['aa', 'bb', 'cc']})
        res = t1.join(t2, how='left').sort('id').head()
        assert 3 == len(res)
        assert ['id', 'val', 'doubled'] == res.column_names()
        assert [int, str, str] == res.column_types()
        assert {'id': 1, 'val': 'a', 'doubled': 'aa'} == res[0]
        assert {'id': 2, 'val': 'b', 'doubled': 'bb'} == res[1]
        assert {'id': 3, 'val': 'c', 'doubled': None} == res[2]

    def test_join_right(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 4], 'doubled': ['aa', 'bb', 'dd']})
        res = t1.join(t2, how='right').sort('id').head()
        assert 3 == len(res)
        assert ['id', 'val', 'doubled'] == res.column_names()
        assert [int, str, str] == res.column_types()
        assert {'id': 1, 'val': 'a', 'doubled': 'aa'} == res[0]
        assert {'id': 2, 'val': 'b', 'doubled': 'bb'} == res[1]
        assert {'id': 4, 'val': None, 'doubled': 'dd'} == res[2]

    def test_join_full(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 4], 'doubled': ['aa', 'bb', 'dd']})
        res = t1.join(t2, how='full').sort('id').head()
        assert 4 == len(res)
        assert ['id', 'val', 'doubled'] == res.column_names()
        assert [int, str, str] == res.column_types()
        assert {'id': 1, 'val': 'a', 'doubled': 'aa'} == res[0]
        assert {'id': 2, 'val': 'b', 'doubled': 'bb'} ==res[1]
        assert {'id': 3, 'val': 'c', 'doubled': None} == res[2]
        assert {'id': 4, 'val': None, 'doubled': 'dd'} == res[3]

    def test_join_cartesian(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [10, 20, 30], 'doubled': ['aa', 'bb', 'cc']})
        res = t1.join(t2, how='cartesian').sort(['id', 'id.1']).head()
        assert 9 == len(res)
        assert ['id', 'val', 'doubled', 'id.1'] == res.column_names()
        assert [int, str, str, int] == res.column_types()
        assert {'id': 1, 'val': 'a', 'doubled': 'aa', 'id.1': 10} == res[0]
        assert {'id': 1, 'val': 'a', 'doubled': 'bb', 'id.1': 20} == res[1]
        assert {'id': 2, 'val': 'b', 'doubled': 'aa', 'id.1': 10} == res[3]
        assert {'id': 3, 'val': 'c', 'doubled': 'cc', 'id.1': 30} == res[8]

    def test_join_bad_how(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 3], 'doubled': ['aa', 'bb', 'cc']})
        with pytest.raises(ValueError) as exception_info:
            t1.join(t2, how='xx')
        exception_message = exception_info.value.args[0]
        assert 'Invalid join type.' == exception_message

    # noinspection PyTypeChecker
    def test_join_bad_right(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            t1.join([1, 2, 3])
        exception_message = exception_info.value.args[0]
        assert 'Can only join two XFrames.' == exception_message

    def test_join_bad_on_list(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 3], 'doubled': ['aa', 'bb', 'cc']})
        with pytest.raises(TypeError) as exception_info:
            t1.join(t2, on=['id', 1])
        exception_message = exception_info.value.args[0]
        assert 'Join keys must each be a str.' == exception_message

    # noinspection PyTypeChecker
    def test_join_bad_on_type(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 3], 'doubled': ['aa', 'bb', 'cc']})
        with pytest.raises(TypeError) as exception_info:
            t1.join(t2, on=1)
        exception_message = exception_info.value.args[0]
        assert "Must pass a 'str', 'list', or 'dict' of join keys." == exception_message

    def test_join_bad_on_col_name(self):
        t1 = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        t2 = XFrame({'id': [1, 2, 3], 'doubled': ['aa', 'bb', 'cc']})
        with pytest.raises(ValueError) as exception_info:
            t1.join(t2, on='xx')
        exception_message = exception_info.value.args[0]
        assert "Key 'xx' is not a column name." == exception_message


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
        assert 3 == len(res)
        assert ['id',
                'val.year', 'val.month', 'val.day',
                'val.hour', 'val.minute', 'val.second'] == res.column_names()
        assert [int, int, int, int, int, int, int] == res.column_types()
        assert [1, 2, 3] == list(res['id'])
        assert [2011, 2012, 2013] == list(res['val.year'])
        assert [1, 2, 3] == list(res['val.month'])
        assert [1, 2, 3] == list(res['val.day'])
        assert [0, 0, 0] == list(res['val.hour'])
        assert [0, 0, 0] == list(res['val.minute'])
        assert [0, 0, 0] == list(res['val.second'])

    # noinspection PyTypeChecker
    def test_split_datetime_col_conflict(self):
        t = XFrame({'id': [1, 2, 3],
                    'val.year': ['x', 'y', 'z'],
                    'val': [datetime(2011, 1, 1),
                            datetime(2012, 2, 2),
                            datetime(2013, 3, 3)]})
        res = t.split_datetime('val', limit='year')
        assert 3 == len(res)
        assert ['id', 'val.year', 'val.year.1'] == res.column_names()
        assert [int, str, int] == res.column_types()
        assert [1, 2, 3] == list(res['id'])
        assert ['x', 'y', 'z'] == list(res['val.year'])
        assert [2011, 2012, 2013] == list(res['val.year.1'])

    def test_split_datetime_bad_col(self):
        t = XFrame({'id': [1, 2, 3], 'val': [datetime(2011, 1, 1),
                                             datetime(2011, 2, 2),
                                             datetime(2011, 3, 3)]})
        with pytest.raises(KeyError) as exception_info:
            t.split_datetime('xx')
        exception_message = exception_info.value.args[0]
        assert "Column 'xx' does not exist in current XFrame." == exception_message


# noinspection PyClassHasNoInit
class TestXFrameFilterby:
    """
    Tests XFrame filterby
    """

    def test_filterby_int_id(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.filterby(1, 'id').sort('id')
        assert 1 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]

    def test_filterby_str_id(self):
        t = XFrame({'id': ['qaz', 'wsx', 'edc', 'rfv'], 'val': ['a', 'b', 'c', 'd']})
        res = t.filterby('qaz', 'id').sort('id')
        assert 1 == len(res)
        assert {'id': 'qaz', 'val': 'a'} == res[0]

    def test_filterby_object_id(self):
        t = XFrame({'id': [datetime(2016, 2, 1, 0, 0),
                           datetime(2016, 2, 2, 0, 0),
                           datetime(2016, 2, 3, 0, 0),
                           datetime(2016, 2, 4, 0, 0)],
                    'val': ['a', 'b', 'c', 'd']})
        res = t.filterby(datetime(2016, 2, 1, 0, 0), 'id').sort('id')
        assert 1 == len(res)
        assert {'id': datetime(2016, 2, 1, 0, 0), 'val': 'a'} == res[0]

    def test_filterby_list_id(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.filterby([1, 3], 'id').sort('id')
        assert 2 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 3, 'val': 'c'} == res[1]

    def test_filterby_tuple_id(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.filterby((1, 3), 'id').sort('id')
        assert 2 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 3, 'val': 'c'} == res[1]

    def test_filterby_iterable_id(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.filterby(range(3), 'id').sort('id')
        assert 2 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]

    def test_filterby_set_id(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.filterby({1, 3}, 'id').sort('id')
        assert 2 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 3, 'val': 'c'} == res[1]

    def test_filterby_list_val(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.filterby(['a', 'b'], 'val').sort('id')
        assert 2 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]
        assert [1, 2] == list(res['id'])

    def test_filterby_xarray(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        a = XArray([1, 3])
        res = t.filterby(a, 'id').sort('id')
        assert 2 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]
        assert res[1] == {'id': 3, 'val': 'c'}
        assert list(res['id']) == [1, 3]

    def test_filterby_function(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.filterby(lambda x: x != 2, 'id').sort('id')
        assert 2 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 3, 'val': 'c'} == res[1]
        assert [1, 3] == list(res['id'])

    def test_filterby_function_exclude(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.filterby(lambda x: x == 2, 'id', exclude=True).sort('id')
        assert 2 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 3, 'val': 'c'} == res[1]
        assert [1, 3] == list(res['id'])

    def test_filterby_function_row(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.filterby(lambda row: row['id'] != 2, None).sort('id')
        assert 2 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 3, 'val': 'c'} == res[1]
        assert [1, 3] == list(res['id'])

    def test_filterby_list_exclude(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.filterby([1, 3], 'id', exclude=True).sort('id')
        assert 2 == len(res)
        assert {'id': 2, 'val': 'b'} == res[0]
        assert {'id': 4, 'val': 'd'} == res[1]
        assert [2, 4] == list(res['id'])

    def test_filterby_bad_column_type_list(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(TypeError) as exception_info:
            t.filterby([1, 3], 'val')
        exception_message = exception_info.value.args[0]
        assert 'Value type (int) does not match column type (str).' == exception_message

    def test_filterby_xarray_exclude(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        a = XArray([1, 3])
        res = t.filterby(a, 'id', exclude=True).sort('id')
        assert 2 == len(res)
        assert {'id': 2, 'val': 'b'} == res[0]
        assert {'id': 4, 'val': 'd'} == res[1]
        assert [2, 4] == list(res['id'])

    # noinspection PyTypeChecker
    def test_filterby_bad_column_name_type(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(TypeError) as exception_info:
            t.filterby([1, 3], 1)
        exception_message = exception_info.value.args[0]
        assert 'Column_name must be a string.' == exception_message

    def test_filterby_bad_column_name(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(KeyError) as exception_info:
            t.filterby([1, 3], 'xx')
        exception_message = exception_info.value.args[0]
        assert "Column 'xx' not in XFrame." == exception_message

    def test_filterby_bad_column_type_xarray(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        a = XArray([1, 3])
        with pytest.raises(TypeError) as exception_info:
            t.filterby(a, 'val')
        exception_message = exception_info.value.args[0]
        assert "Type of given values ('<class 'int'>') does not match " + \
               "type of column 'val' ('<class 'str'>') in XFrame." == exception_message

    def test_filterby_bad_list_empty(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(ValueError) as exception_info:
            t.filterby([], 'id').sort('id')
        exception_message = exception_info.value.args[0]
        assert 'Value list is empty.' == exception_message

    def test_filterby_bad_xarray_empty(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        a = XArray([])
        with pytest.raises(TypeError) as exception_info:
            t.filterby(a, 'val')
        exception_message = exception_info.value.args[0]
        assert "Type of given values ('None') does not match " + \
               "type of column 'val' ('<class 'str'>') in XFrame." == exception_message


# noinspection PyClassHasNoInit
class TestXFramePackColumnsList:
    """
    Tests XFrame pack_columns into list
    """

    def test_pack_columns(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='new')
        assert 4 == len(res)
        assert 1 == res.num_columns()
        assert ['new'] == res.column_names()
        assert [list] == res.column_types()
        assert {'new': [1, 'a']} == res[0]
        assert {'new': [2, 'b']} == res[1]

    def test_pack_columns_all(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.pack_columns(new_column_name='new')
        assert 4 == len(res)
        assert 1 == res.num_columns()
        assert['new'] == res.column_names()
        assert [list] == res.column_types()
        assert {'new': [1, 'a']} == res[0]
        assert {'new': [2, 'b']} == res[1]

    def test_pack_columns_prefix(self):
        t = XFrame({'x.id': [1, 2, 3, 4], 'x.val': ['a', 'b', 'c', 'd'], 'another': [10, 20, 30, 40]})
        res = t.pack_columns(column_prefix='x', new_column_name='new')
        assert 4 == len(res)
        assert 2 == res.num_columns()
        assert['another', 'new'] == res.column_names()
        assert [int, list] == res.column_types()
        assert {'another': 10, 'new': [1, 'a']} == res[0]
        assert {'another': 20, 'new': [2, 'b']} == res[1]

    def test_pack_columns_rest(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd'], 'another': [10, 20, 30, 40]})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='new')
        assert 4 == len(res)
        assert 2 == res.num_columns()
        assert ['another', 'new'] == res.column_names()
        assert [int, list] == res.column_types()
        assert {'another': 10, 'new': [1, 'a']} == res[0]
        assert {'another': 20, 'new': [2, 'b']} == res[1]

    # TODO fix assert
    def test_pack_columns_na(self):
        t = XFrame({'id': [1, 2, None, 4], 'val': ['a', 'b', 'c', None]})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='new', fill_na='x')
        assert 4 == len(res)
        assert 1 == res.num_columns()
        assert ['new'] == res.column_names()
        assert [list] == res.column_types()
        assert {'new': [1, 'a']} == res[0] == res[0]
        assert {'new': [2, 'b']} == res[1]
        assert {'new': ['x', 'c']} ==res[2]
        assert {'new': [4, 'x']} == res[3]

    def test_pack_columns_fill_na(self):
        t = XFrame({'id': [1, 2, None, 4], 'val': ['a', 'b', 'c', None]})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='new', fill_na=99)
        assert 4 == len(res)
        assert 1 == res.num_columns()
        assert  ['new'] == res.column_names()
        assert [list] == res.column_types()
        assert  {'new': [1, 'a']} == res[0]
        assert {'new': [2, 'b']} == res[1]
        assert {'new': [99, 'c']} == res[2]
        assert {'new': [4, 99]} == res[3]

    def test_pack_columns_def_new_name(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.pack_columns(columns=['id', 'val'])
        assert 4 == len(res)
        assert 1 == res.num_columns()
        assert ['X.0'] == res.column_names()
        assert [list] == res.column_types()
        assert {'X.0': [1, 'a']} == res[0]
        assert {'X.0': [2, 'b']} == res[1]

    def test_pack_columns_prefix_def_new_name(self):
        t = XFrame({'x.id': [1, 2, 3, 4], 'x.val': ['a', 'b', 'c', 'd'], 'another': [10, 20, 30, 40]})
        res = t.pack_columns(column_prefix='x')
        assert 4 == len(res)
        assert 2 == res.num_columns()
        assert ['another', 'x'] == res.column_names()
        assert [int, list] == res.column_types()
        assert {'another': 10, 'x': [1, 'a']} == res[0]
        assert {'another': 20, 'x': [2, 'b']} == res[1]

    # noinspection PyTypeChecker
    def test_pack_columns_bad_col_spec(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(ValueError) as exception_info:
            t.pack_columns(columns='id', column_prefix='val')
        exception_message = exception_info.value.args[0]
        assert "'Columns' and 'column_prefix' parameter cannot be given at the same time." == exception_message

    # noinspection PyTypeChecker
    def test_pack_columns_bad_col_prefix_type(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(TypeError) as exception_info:
            t.pack_columns(column_prefix=1)
        exception_message = exception_info.value.args[0]
        assert "'Column_prefix' must be a string. Found 'int': 1." == exception_message

    def test_pack_columns_bad_col_prefix(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(ValueError) as exception_info:
            t.pack_columns(column_prefix='xx')
        exception_message = exception_info.value.args[0]
        assert "There are no column starts with prefix 'xx'." == exception_message

    def test_pack_columns_bad_cols(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(ValueError) as exception_info:
            t.pack_columns(columns=['xx'])
        exception_message = exception_info.value.args[0]
        assert "Current XFrame has no column called 'xx'." == exception_message

    def test_pack_columns_bad_cols_dup(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(ValueError) as exception_info:
            t.pack_columns(columns=['id', 'id'])
        exception_message = exception_info.value.args[0]
        assert 'There are duplicate column names in columns parameter.' == exception_message

    def test_pack_columns_bad_cols_single(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(ValueError) as exception_info:
            t.pack_columns(columns=['id'])
        exception_message = exception_info.value.args[0]
        assert 'Please provide at least two columns to pack.' == exception_message

    # noinspection PyTypeChecker
    def test_pack_columns_bad_dtype(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(ValueError) as exception_info:
            t.pack_columns(columns=['id', 'val'], dtype=int)
        exception_message = exception_info.value.args[0]
        assert "Resulting dtype has to be one of 'dict', 'array.array', 'list', or 'tuple' type." == exception_message

    # noinspection PyTypeChecker
    def test_pack_columns_bad_new_col_name_type(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(TypeError) as exception_info:
            t.pack_columns(columns=['id', 'val'], new_column_name=1)
        exception_message = exception_info.value.args[0]
        assert "'New_column_name' must be a string. Found 'int': 1." == exception_message

    def test_pack_columns_bad_new_col_name_dup_rest(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd'], 'another': [11, 12, 13, 14]})
        with pytest.raises(KeyError) as exception_info:
            t.pack_columns(columns=['id', 'val'], new_column_name='another')
        exception_message = exception_info.value.args[0]
        assert "Current XFrame already contains a column name 'another'." == exception_message

    def test_pack_columns_good_new_col_name_dup_key(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='id')
        assert ['id'] == res.column_names()
        assert {'id': [1, 'a']} == res[0]
        assert {'id': [2, 'b']} == res[1]


class TestXFramePackColumnsTuple:
    """
    Tests XFrame pack_columns into tuple
    """

    def test_pack_columns(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='new', dtype=tuple)
        assert 4 == len(res)
        assert 1 == res.num_columns()
        assert ['new'] == res.column_names()
        assert [tuple] == res.column_types()
        assert {'new': (1, 'a')} == res[0]
        assert {'new': (2, 'b')} == res[1]

# noinspection PyClassHasNoInit
class TestXFramePackColumnsDict:
    """
    Tests XFrame pack_columns into dict
    """

    def test_pack_columns(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='new', dtype=dict)
        assert 4 == len(res)
        assert 1 == res.num_columns()
        assert [dict] == res.dtype()
        assert ['new'] == res.column_names()
        assert {'new': {'id': 1, 'val': 'a'}} == res[0]
        assert {'new': {'id': 2, 'val': 'b'}} == res[1]

    def test_pack_columns_prefix(self):
        t = XFrame({'x.id': [1, 2, 3, 4], 'x.val': ['a', 'b', 'c', 'd'], 'another': [10, 20, 30, 40]})
        res = t.pack_columns(column_prefix='x', dtype=dict)
        assert 4 == len(res)
        assert 2 == res.num_columns()
        assert [int, dict] == res.dtype()
        assert ['another', 'x'] == res.column_names()
        assert {'another': 10, 'x': {'id': 1, 'val': 'a'}} == res[0]
        assert {'another': 20, 'x': {'id': 2, 'val': 'b'}} == res[1]

    def test_pack_columns_prefix_named(self):
        t = XFrame({'x.id': [1, 2, 3, 4], 'x.val': ['a', 'b', 'c', 'd'], 'another': [10, 20, 30, 40]})
        res = t.pack_columns(column_prefix='x', dtype=dict, new_column_name='new')
        assert 4 == len(res)
        assert 2 == res.num_columns()
        assert [int, dict] == res.dtype()
        assert ['another', 'new'] == res.column_names()
        assert {'another': 10, 'new': {'id': 1, 'val': 'a'}} == res[0]
        assert {'another': 20, 'new': {'id': 2, 'val': 'b'}} ==res[1]

    def test_pack_columns_prefix_no_remove(self):
        t = XFrame({'x.id': [1, 2, 3, 4], 'x.val': ['a', 'b', 'c', 'd'], 'another': [10, 20, 30, 40]})
        res = t.pack_columns(column_prefix='x', dtype=dict, remove_prefix=False)
        assert 4 == len(res)
        assert 2 == res.num_columns()
        assert [int, dict] == res.dtype()
        assert ['another', 'x'] == res.column_names()
        assert {'another': 10, 'x': {'x.id': 1, 'x.val': 'a'}} == res[0]
        assert {'another': 20, 'x': {'x.id': 2, 'x.val': 'b'}} == res[1]

    def test_pack_columns_drop_missing(self):
        t = XFrame({'id': [1, 2, None, 4], 'val': ['a', 'b', 'c', None]})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='new', dtype=dict)
        assert 4 == len(res)
        assert 1 == res.num_columns()
        assert [dict] == res.dtype()
        assert ['new'] == res.column_names()
        assert {'new': {'id': 1, 'val': 'a'}} == res[0]
        assert {'new': {'id': 2, 'val': 'b'}} == res[1]
        assert {'new': {'val': 'c'}} == res[2]
        assert {'new': {'id': 4}} == res[3]

    def test_pack_columns_fill_na(self):
        t = XFrame({'id': [1, 2, None, 4], 'val': ['a', 'b', 'c', None]})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='new', dtype=dict, fill_na=99)
        assert 4 == len(res)
        assert 1 == res.num_columns()
        assert [dict] == res.dtype()
        assert ['new'] == res.column_names()
        assert {'new': {'id': 1, 'val': 'a'}} == res[0]
        assert {'new': {'id': 2, 'val': 'b'}} == res[1]
        assert {'new': {'id': 99, 'val': 'c'}} == res[2]
        assert {'new': {'id': 4, 'val': 99}} == res[3]


# noinspection PyClassHasNoInit
class TestXFramePackColumnsArray:
    """
    Tests XFrame pack_columns into array
    """

    def test_pack_columns(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [10, 20, 30, 40]})
        res = t.pack_columns(columns=['id', 'val'], new_column_name='new', dtype=array.array)
        assert 4 == len(res)
        assert 1 == res.num_columns()
        assert [array.array] == res.dtype()
        assert ['new'] == res.column_names()
        assert {'new': array.array('d', [1.0, 10.0])} == res[0]
        assert {'new': array.array('d', [2.0, 20.0])} == res[1]

    def test_pack_columns_bad_fill_na_not_numeric(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [10, 20, 30, 40]})
        with pytest.raises(ValueError) as exception_info:
            t.pack_columns(columns=['id', 'val'], new_column_name='new', dtype=array.array, fill_na='a')
        exception_message = exception_info.value.args[0]
        assert 'Fill_na value for array needs to be numeric type.' == exception_message

    def test_pack_columns_bad_not_numeric(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': ['a', 'b', 'c', 'd']})
        with pytest.raises(TypeError) as exception_info:
            t.pack_columns(columns=['id', 'val'], new_column_name='new', dtype=array.array)
        exception_message = exception_info.value.args[0]
        assert "Column 'val' type is not numeric, cannot pack into array type." == exception_message

        # TODO list


# noinspection PyClassHasNoInit
class TestXFrameUnpackList:
    """
    Tests XFrame unpack where the unpacked column contains a list
    """

    def test_unpack(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [[10, 'a'], [20, 'b'], [30, 'c'], [40, 'd']]})
        res = t.unpack('val')
        assert 4 == len(res)
        assert ['id', 'val.0', 'val.1'] == res.column_names()
        assert [int, int, str] == res.column_types()
        assert {'id': 1, 'val.0': 10, 'val.1': 'a'} == res[0]
        assert {'id': 2, 'val.0': 20, 'val.1': 'b'} == res[1]

    def test_unpack_prefix(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [[10, 'a'], [20, 'b'], [30, 'c'], [40, 'd']]})
        res = t.unpack('val', column_name_prefix='x')
        assert 4 == len(res)
        assert ['id', 'x.0', 'x.1'] == res.column_names()
        assert [int, int, str] == res.column_types()
        assert {'id': 1, 'x.0': 10, 'x.1': 'a'} == res[0]
        assert {'id': 2, 'x.0': 20, 'x.1': 'b'} == res[1]

    def test_unpack_types(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [[10, 'a'], [20, 'b'], [30, 'c'], [40, 'd']]})
        res = t.unpack('val', column_types=[str, str])
        assert 4 == len(res)
        assert ['id', 'val.0', 'val.1'] == res.column_names()
        assert [int, str, str] == res.column_types()
        assert {'id': 1, 'val.0': '10', 'val.1': 'a'} == res[0]
        assert {'id': 2, 'val.0': '20', 'val.1': 'b'} == res[1]

    def test_unpack_na_value(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [[10, 'a'], [20, 'b'], [None, 'c'], [40, None]]})
        res = t.unpack('val', na_value=99)
        assert 4 == len(res)
        assert ['id', 'val.0', 'val.1'] == res.column_names()
        assert [int, int, str] == res.column_types()
        assert {'id': 1, 'val.0': 10, 'val.1': 'a'} == res[0]
        assert {'id': 2, 'val.0': 20, 'val.1': 'b'} == res[1]
        assert {'id': 3, 'val.0': 99, 'val.1': 'c'} == res[2]
        assert {'id': 4, 'val.0': 40, 'val.1': '99'} == res[3]

    def test_unpack_limit(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [[10, 'a'], [20, 'b'], [30, 'c'], [40, 'd']]})
        res = t.unpack('val', limit=[1])
        assert 4 == len(res)
        assert ['id', 'val.1'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 1, 'val.1': 'a'} == res[0]
        assert {'id': 2, 'val.1': 'b'} == res[1]


# noinspection PyClassHasNoInit
class TestXFrameUnpackDict:
    """
    Tests XFrame unpack where the unpacked column contains a dict
    """

    def test_unpack(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [{'a': 1}, {'b': 2}, {'c': 3}, {'d': 4}]})
        res = t.unpack('val')
        assert 4 == len(res)
        assert ['id', 'val.a', 'val.b', 'val.c', 'val.d'] == sorted(res.column_names())
        assert [int, int, int, int, int] == res.column_types()
        assert {'id': 1, 'val.a': 1, 'val.c': None, 'val.b': None, 'val.d': None} == res[0]
        assert {'id': 2, 'val.a': None, 'val.c': None, 'val.b': 2, 'val.d': None} == res[1]

    def test_unpack_mult(self):
        t = XFrame({'id': [1, 2, 3], 'val': [{'a': 1}, {'b': 2}, {'a': 1, 'b': 2}]})
        res = t.unpack('val')
        assert 3 == len(res)
        assert ['id', 'val.a', 'val.b'] == sorted(res.column_names())
        assert [int, int, int] == res.column_types()
        assert {'id': 1, 'val.a': 1, 'val.b': None} == res[0]
        assert {'id': 2, 'val.a': None, 'val.b': 2} == res[1]
        assert {'id': 3, 'val.a': 1, 'val.b': 2} == res[2]

    def test_unpack_prefix(self):
        t = XFrame({'id': [1, 2, 3], 'val': [{'a': 1}, {'b': 2}, {'a': 1, 'b': 2}]})
        res = t.unpack('val', column_name_prefix='x')
        assert 3 == len(res)
        assert ['id', 'x.a', 'x.b'] == sorted(res.column_names())
        assert [int, int, int] == res.column_types()
        assert {'id': 1, 'x.a': 1, 'x.b': None} == res[0]
        assert {'id': 2, 'x.a': None, 'x.b': 2} == res[1]
        assert {'id': 3, 'x.a': 1, 'x.b': 2} == res[2]

    def test_unpack_types(self):
        t = XFrame({'id': [1, 2, 3], 'val': [{'a': 1}, {'b': 2}, {'a': 1, 'b': 2}]})
        res = t.unpack('val', column_types=[str, str], limit=['a', 'b'])
        assert 3 == len(res)
        assert ['id', 'val.a', 'val.b'] == sorted(res.column_names())
        assert [int, str, str] == res.column_types()
        assert {'id': 1, 'val.a': '1', 'val.b': None} == res[0]
        assert {'id': 2, 'val.a': None, 'val.b': '2'} == res[1]
        assert {'id': 3, 'val.a': '1', 'val.b': '2'} == res[2]

    def test_unpack_na_value(self):
        t = XFrame({'id': [1, 2, 3], 'val': [{'a': 1}, {'b': 2}, {'a': 1, 'b': 2}]})
        res = t.unpack('val', na_value=99)
        assert 3 == len(res)
        assert ['id', 'val.a', 'val.b'] == sorted(res.column_names())
        assert [int, int, int] == res.column_types()
        assert {'id': 1, 'val.a': 1, 'val.b': 99} == res[0]
        assert {'id': 2, 'val.a': 99, 'val.b': 2} == res[1]
        assert {'id': 3, 'val.a': 1, 'val.b': 2} == res[2]

    def test_unpack_limit(self):
        t = XFrame({'id': [1, 2, 3], 'val': [{'a': 1}, {'b': 2}, {'a': 1, 'b': 2}]})
        res = t.unpack('val', limit=['b'])
        assert 3 == len(res)
        assert ['id', 'val.b'] == res.column_names()
        assert [int, int] == res.column_types()
        assert {'id': 1, 'val.b': None} == res[0]
        assert {'id': 2, 'val.b': 2} == res[1]
        assert {'id': 3, 'val.b': 2} == res[2]

    def test_unpack_bad_types_no_limit(self):
        t = XFrame({'id': [1, 2, 3], 'val': [{'a': 1}, {'b': 2}, {'a': 1, 'b': 2}]})
        with pytest.raises(ValueError) as exception_info:
            t.unpack('val', column_types=[str, str])
        exception_message = exception_info.value.args[0]
        assert "If 'column_types' is given, 'limit' has to be provided to unpack dict type." == exception_message


# TODO unpack array

# noinspection PyClassHasNoInit
class TestXFrameStackList:
    """
    Tests XFrame stack where column is a list
    """

    def test_stack_list(self):
        t = XFrame({'id': [1, 2, 3], 'val': [['a1', 'b1', 'c1'], ['a2', 'b2'], ['a3', 'b3', 'c3', None]]})
        res = t.stack('val')
        assert 9 == len(res)
        assert ['id', 'X'] == res.column_names()
        assert {'id': 1, 'X': 'a1'} == res[0]
        assert {'id': 3, 'X': None} == res[8]

    def test_stack_list_drop_na(self):
        t = XFrame({'id': [1, 2, 3], 'val': [['a1', 'b1', 'c1'], ['a2', 'b2'], ['a3', 'b3', 'c3', None]]})
        res = t.stack('val', drop_na=True)
        assert 8 == len(res)
        assert ['id', 'X'] == res.column_names()
        assert {'id': 1, 'X': 'a1'} == res[0]
        assert {'id': 3, 'X': 'c3'} == res[7]

    def test_stack_name(self):
        t = XFrame({'id': [1, 2, 3], 'val': [['a1', 'b1', 'c1'], ['a2', 'b2'], ['a3', 'b3', 'c3', None]]})
        res = t.stack('val', new_column_name='flat_val')
        assert 9 == len(res)
        assert ['id', 'flat_val'] == res.column_names()

    def test_stack_bad_col_name(self):
        t = XFrame({'id': [1, 2, 3], 'val': [['a1', 'b1', 'c1'], ['a2', 'b2'], ['a3', 'b3', 'c3', None]]})
        with pytest.raises(ValueError) as exception_info:
            t.stack('xx')
        exception_message = exception_info.value.args[0]
        assert "Cannot find column 'xx' in the XFrame." == exception_message

    def test_stack_bad_col_value(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            t.stack('val')
        exception_message = exception_info.value.args[0]
        assert "Stack is only supported for column of 'dict', 'list', or 'array' type." == exception_message

    # noinspection PyTypeChecker
    def test_stack_bad_new_col_name_type(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            t.stack('val', new_column_name=1)
        exception_message = exception_info.value.args[0]
        assert "Stack is only supported for column of 'dict', 'list', or 'array' type." == exception_message

    def test_stack_new_col_name_dup_ok(self):
        t = XFrame({'id': [1, 2, 3], 'val': [['a1', 'b1', 'c1'], ['a2', 'b2'], ['a3', 'b3', 'c3', None]]})
        res = t.stack('val', new_column_name='val')
        assert ['id', 'val'] == res.column_names()

    def test_stack_bad_new_col_name_dup(self):
        t = XFrame({'id': [1, 2, 3], 'val': [['a1', 'b1', 'c1'], ['a2', 'b2'], ['a3', 'b3', 'c3', None]]})
        with pytest.raises(ValueError) as exception_info:
            t.stack('val', new_column_name='id')
        exception_message = exception_info.value.args[0]
        assert "Column with name 'id' already exists, pick a new column name." == exception_message

    def test_stack_bad_no_data(self):
        t = XFrame({'id': [1, 2, 3], 'val': [['a1', 'b1', 'c1'], ['a2', 'b2'], ['a3', 'b3', 'c3', None]]})
        t = t.head(0)
        with pytest.raises(ValueError) as exception_info:
            t.stack('val', new_column_name='val')
        exception_message = exception_info.value.args[0]
        assert 'Cannot infer column type because there are not enough rows.' == exception_message


# noinspection PyClassHasNoInit
class TestXFrameStackDict:
    """
    Tests XFrame stack where column is a dict
    """

    def test_stack_dict(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [{'a': 3, 'b': 2}, {'a': 2, 'c': 2}, {'c': 1, 'd': 3}, {}]})
        res = t.stack('val')
        assert 7 == len(res)
        assert ['id', 'K', 'V'] == res.column_names()
        assert {'id': 1, 'K': 'a', 'V': 3} == res[0]
        assert {'id': 4, 'K': None, 'V': None} == res[6]

    def test_stack_names(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [{'a': 3, 'b': 2}, {'a': 2, 'c': 2}, {'c': 1, 'd': 3}, {}]})
        res = t.stack('val', ['new_k', 'new_v'])
        assert 7 == len(res)
        assert ['id', 'new_k', 'new_v'] == res.column_names()
        assert {'id': 1, 'new_k': 'a', 'new_v': 3} == res[0]
        assert {'id': 4, 'new_k': None, 'new_v': None} == res[6]

    def test_stack_dropna(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [{'a': 3, 'b': 2}, {'a': 2, 'c': 2}, {'c': 1, 'd': 3}, {}]})
        res = t.stack('val', drop_na=True)
        assert 6 == len(res)
        assert ['id', 'K', 'V'] == res.column_names()
        assert {'id': 1, 'K': 'a', 'V': 3} == res[0]
        assert {'id': 3, 'K': 'd', 'V': 3} == res[5]

    def test_stack_bad_col_name(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [{'a': 3, 'b': 2}, {'a': 2, 'c': 2}, {'c': 1, 'd': 3}, {}]})
        with pytest.raises(ValueError) as exception_info:
            t.stack('xx')
        exception_message = exception_info.value.args[0]
        assert "Cannot find column 'xx' in the XFrame." == exception_message

    # noinspection PyTypeChecker
    def test_stack_bad_new_col_name_type(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [{'a': 3, 'b': 2}, {'a': 2, 'c': 2}, {'c': 1, 'd': 3}, {}]})
        with pytest.raises(TypeError) as exception_info:
            t.stack('val', new_column_name=1)
        exception_message = exception_info.value.args[0]
        assert "'New_column_name' has to be a 'list' to stack 'dict' type. Found 'int': 1" == exception_message

    def test_stack_bad_new_col_name_len(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [{'a': 3, 'b': 2}, {'a': 2, 'c': 2}, {'c': 1, 'd': 3}, {}]})
        with pytest.raises(TypeError) as exception_info:
            t.stack('val', new_column_name=['a'])
        exception_message = exception_info.value.args[0]
        assert "'New_column_name' must have length of two." == exception_message

    def test_stack_bad_new_col_name_dup(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [{'a': 3, 'b': 2}, {'a': 2, 'c': 2}, {'c': 1, 'd': 3}, {}]})
        with pytest.raises(ValueError) as exception_info:
            t.stack('val', new_column_name=['id', 'xx'])
        exception_message = exception_info.value.args[0]
        assert "Column with name 'id' already exists, pick a new column name." == exception_message

    def test_stack_bad_no_data(self):
        t = XFrame({'id': [1, 2, 3, 4], 'val': [{'a': 3, 'b': 2}, {'a': 2, 'c': 2}, {'c': 1, 'd': 3}, {}]})
        t = t.head(0)
        with pytest.raises(ValueError) as exception_info:
            t.stack('val', new_column_name=['k', 'v'])
        exception_message = exception_info.value.args[0]
        assert 'Cannot infer column type because there are not enough rows.' == exception_message


# noinspection PyClassHasNoInit
class TestXFrameUnstackList:
    """
    Tests XFrame unstack where unstack column is list
    """

    def test_unstack(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1, 3], 'val': ['a1', 'b1', 'c1', 'a2', 'b2', 'a3', 'c3']})
        res = t.unstack('val')
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'unstack'] == res.column_names()
        assert [int, list] == res.column_types()
        assert {'id': 1, 'unstack': ['a1', 'a2', 'a3']} == res[0]
        assert {'id': 2, 'unstack': ['b1', 'b2']} == res[1]
        assert {'id': 3, 'unstack': ['c1', 'c3']} == res[2]

    def test_unstack_name(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1, 3], 'val': ['a1', 'b1', 'c1', 'a2', 'b2', 'a3', 'c3']})
        res = t.unstack('val', new_column_name='vals')
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'vals'] == res.column_names()
        assert [int, list] == res.column_types()
        assert {'id': 1, 'vals': ['a1', 'a2', 'a3']} == res[0]
        assert {'id': 2, 'vals': ['b1', 'b2']} == res[1]
        assert {'id': 3, 'vals': ['c1', 'c3']} == res[2]


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
        assert 3 == len(res)
        assert ['id', 'unstack'] == res.column_names()
        assert [int, dict] == res.column_types()
        assert {'id': 1, 'unstack': {'ka1': 'a1', 'ka2': 'a2', 'ka3': 'a3'}} == res[0]
        assert {'id': 2, 'unstack': {'kb1': 'b1', 'kb2': 'b2'}} == res[1]
        assert {'id': 3, 'unstack': {'kc1': 'c1', 'kc3': 'c3'}} == res[2]

    def test_unstack_name(self):
        t = XFrame({'id': [1, 2, 3, 1, 2, 1, 3],
                    'key': ['ka1', 'kb1', 'kc1', 'ka2', 'kb2', 'ka3', 'kc3'],
                    'val': ['a1', 'b1', 'c1', 'a2', 'b2', 'a3', 'c3']})
        res = t.unstack(['key', 'val'], new_column_name='vals')
        res = res.topk('id', reverse=True)
        assert 3 == len(res)
        assert ['id', 'vals'] == res.column_names()
        assert [int, dict] == res.column_types()
        assert {'id': 1, 'vals': {'ka1': 'a1', 'ka2': 'a2', 'ka3': 'a3'}} == res[0]
        assert {'id': 2, 'vals': {'kb1': 'b1', 'kb2': 'b2'}} == res[1]
        assert {'id': 3, 'vals': {'kc1': 'c1', 'kc3': 'c3'}} == res[2]


# noinspection PyClassHasNoInit
class TestXFrameUnique:
    """
    Tests XFrame unique
    """

    def test_unique_noop(self):
        t = XFrame({'id': [3, 2, 1], 'val': ['c', 'b', 'a']})
        res = t.unique()
        assert 3 == len(res)
        assert [1, 2, 3] == sorted(list(res['id']))
        assert ['a', 'b', 'c'] == sorted(list(res['val']))

    def test_unique(self):
        t = XFrame({'id': [3, 2, 1, 1], 'val': ['c', 'b', 'a', 'a']})
        res = t.unique()
        assert 3 == len(res)
        assert [1, 2, 3] == sorted(list(res['id']))
        assert ['a', 'b', 'c'] == sorted(list(res['val']))

    def test_unique_part(self):
        t = XFrame({'id': [3, 2, 1, 1], 'val': ['c', 'b', 'a', 'x']})
        res = t.unique()
        assert 4 == len(res)
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
        assert 3 == len(res)
        assert [1, 2, 3] == list(res['id'])
        assert ['a', 'b', 'c'] == list(res['val'])

    def test_sort_descending(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.sort('id', ascending=False)
        assert 3 == len(res)
        assert [3, 2, 1] == list(res['id'])
        assert ['c', 'b', 'a'] == list(res['val'])

    def test_sort_multi_col(self):
        t = XFrame({'id': [3, 2, 1, 1], 'val': ['c', 'b', 'b', 'a']})
        res = t.sort(['id', 'val'])
        assert 4 == len(res)
        assert [1, 1, 2, 3] == list(res['id'])
        assert ['a', 'b', 'b', 'c'] == list(res['val'])

    def test_sort_multi_col_asc_desc(self):
        t = XFrame({'id': [3, 2, 1, 1], 'val': ['c', 'b', 'b', 'a']})
        res = t.sort([('id', True), ('val', False)])
        assert 4 == len(res)
        assert [1, 1, 2, 3] == list(res['id'])
        assert ['b', 'a', 'b', 'c'] == list(res['val'])


# noinspection PyClassHasNoInit
class TestXFrameDropna:
    """
    Tests XFrame dropna
    """

    def test_dropna_no_drop(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.dropna()
        assert 3 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': 'b'} == res[1]
        assert {'id': 3, 'val': 'c'} == res[2]

    def test_dropna_none(self):
        t = XFrame({'id': [1, None, 3], 'val': ['a', 'b', 'c']})
        res = t.dropna()
        assert 2 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 3, 'val': 'c'} == res[1]

    def test_dropna_nan(self):
        t = XFrame({'id': [1.0, float('nan'), 3.0], 'val': ['a', 'b', 'c']})
        res = t.dropna()
        assert 2 == len(res)
        assert {'id': 1.0, 'val': 'a'} == res[0]
        assert {'id': 3.0, 'val': 'c'} == res[1]

    def test_dropna_float_none(self):
        t = XFrame({'id': [1.0, None, 3.0], 'val': ['a', 'b', 'c']})
        res = t.dropna()
        assert 2 == len(res)
        assert {'id': 1.0, 'val': 'a'} == res[0]
        assert {'id': 3.0, 'val': 'c'} == res[1]

    def test_dropna_empty_list(self):
        t = XFrame({'id': [1, None, 3], 'val': ['a', 'b', 'c']})
        res = t.dropna(columns=[])
        assert 3 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': None, 'val': 'b'} == res[1]
        assert {'id': 3, 'val': 'c'} == res[2]

    def test_dropna_any(self):
        t = XFrame({'id': [1, None, None], 'val': ['a', None, 'c']})
        res = t.dropna()
        assert 1 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]

    def test_dropna_all(self):
        t = XFrame({'id': [1, None, None], 'val': ['a', None, 'c']})
        res = t.dropna(how='all')
        assert 2 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': None, 'val': 'c'} == res[1]

    def test_dropna_col_val(self):
        t = XFrame({'id': [1, None, None], 'val': ['a', None, 'c']})
        res = t.dropna(columns='val')
        assert 2 == len(res)
        assert {'id': 1, 'val': 'a'} ==res[0]
        assert {'id': None, 'val': 'c'} == res[1]

    def test_dropna_col_id(self):
        t = XFrame({'id': [1, 2, None], 'val': ['a', None, 'c']})
        res = t.dropna(columns='id')
        assert 2 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 2, 'val': None} == res[1]

    # noinspection PyTypeChecker
    def test_dropna_bad_col_arg(self):
        t = XFrame({'id': [1, 2, None], 'val': ['a', None, 'c']})
        with pytest.raises(TypeError) as exception_info:
            t.dropna(columns=1)
        exception_message = exception_info.value.args[0]
        assert "Must give columns as a 'list', 'str', or 'None'." == exception_message

    def test_dropna_bad_col_name_in_list(self):
        t = XFrame({'id': [1, 2, None], 'val': ['a', None, 'c']})
        with pytest.raises(TypeError) as exception_info:
            t.dropna(columns=['id', 2])
        exception_message = exception_info.value.args[0]
        assert "All columns must be of 'str' type." == exception_message

    def test_dropna_bad_how(self):
        t = XFrame({'id': [1, 2, None], 'val': ['a', None, 'c']})
        with pytest.raises(ValueError) as exception_info:
            t.dropna(how='xx')
        exception_message = exception_info.value.args[0]
        assert "Must specify 'any' or 'all'." == exception_message

        # TODO drop_missing


# noinspection PyClassHasNoInit
class TestXFrameDropnaSplit:
    """
    Tests XFrame dropna_split
    """

    def test_dropna_split_no_drop(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res1, res2 = t.dropna_split()
        assert 3 == len(res1)
        assert {'id': 1, 'val': 'a'} ==res1[0]
        assert {'id': 2, 'val': 'b'} == res1[1]
        assert {'id': 3, 'val': 'c'} == res1[2]
        assert 0 == len(res2)

    def test_dropna_split_none(self):
        t = XFrame({'id': [1, None, 3], 'val': ['a', 'b', 'c']})
        res1, res2 = t.dropna_split()
        assert 2 == len(res1)
        assert {'id': 1, 'val': 'a'} ==res1[0]
        assert {'id': 3, 'val': 'c'} == res1[1]
        assert 1 == len(res2)
        assert {'id': None, 'val': 'b'} == res2[0]

    def test_dropna_split_all(self):
        t = XFrame({'id': [1, None, None], 'val': ['a', None, 'c']})
        res1, res2 = t.dropna_split(how='all')
        assert 2 == len(res1)
        assert {'id': 1, 'val': 'a'} == res1[0]
        assert {'id': None, 'val': 'c'} == res1[1]
        assert 1 == len(res2)
        assert {'id': None, 'val': None} == res2[0]


# noinspection PyClassHasNoInit
class TestXFrameFillna:
    """
    Tests XFrame fillna
    """

    def test_fillna(self):
        t = XFrame({'id': [1, None, None], 'val': ['a', 'b', 'c']})
        res = t.fillna('id', 0)
        assert 3 == len(res)
        assert {'id': 1, 'val': 'a'} == res[0]
        assert {'id': 0, 'val': 'b'} == res[1]
        assert {'id': 0, 'val': 'c'} == res[2]

    def test_fillna_bad_col_name(self):
        t = XFrame({'id': [1, None, None], 'val': ['a', 'b', 'c']})
        with pytest.raises(ValueError) as exception_info:
            t.fillna('xx', 0)
        exception_message = exception_info.value.args[0]
        assert "Column name does not exist: 'xx'." == exception_message

    # noinspection PyTypeChecker
    def test_fillna_bad_arg_type(self):
        t = XFrame({'id': [1, None, None], 'val': ['a', 'b', 'c']})
        with pytest.raises(TypeError) as exception_info:
            t.fillna(1, 0)
        exception_message = exception_info.value.args[0]
        assert "Must give column name as a 'str'. Found 'int': 1." == exception_message


# noinspection PyClassHasNoInit
class TestXFrameAddRowNumber:
    """
    Tests XFrame add_row_number
    """

    def test_add_row_number(self):
        t = XFrame({'ident': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.add_row_number()
        assert ['id', 'ident', 'val'] == res.column_names()
        assert {'id': 0, 'ident': 1, 'val': 'a'} == res[0]
        assert {'id': 1, 'ident': 2, 'val': 'b'} == res[1]
        assert {'id': 2, 'ident': 3, 'val': 'c'} == res[2]

    def test_add_row_number_start(self):
        t = XFrame({'ident': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.add_row_number(start=10)
        assert ['id', 'ident', 'val'] == res.column_names()
        assert {'id': 10, 'ident': 1, 'val': 'a'} == res[0]
        assert {'id': 11, 'ident': 2, 'val': 'b'} == res[1]
        assert {'id': 12, 'ident': 3, 'val': 'c'} == res[2]

    def test_add_row_number_name(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.add_row_number(column_name='row_number')
        assert ['row_number', 'id', 'val'] == res.column_names()
        assert {'row_number': 0, 'id': 1, 'val': 'a'} == res[0]
        assert {'row_number': 1, 'id': 2, 'val': 'b'} == res[1]
        assert {'row_number': 2, 'id': 3, 'val': 'c'} == res[2]


# noinspection PyClassHasNoInit
class TestXFrameShape:
    """
    Tests XFrame shape
    """

    def test_shape(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        assert (3, 2) == t.shape

    def test_shape_empty(self):
        t = XFrame()
        assert (0, 0) == t.shape


# noinspection SqlNoDataSourceInspection,SqlDialectInspection
# noinspection PyClassHasNoInit
class TestXFrameSql:
    """
    Tests XFrame sql
    """

    def test_sql(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.sql("SELECT * FROM xframe WHERE id > 1 ORDER BY id")

        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 2, 'val': 'b'} == res[0]
        assert {'id': 3, 'val': 'c'} == res[1]

    def test_sql_name(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        res = t.sql("SELECT * FROM tmp_tbl WHERE id > 1 ORDER BY id", table_name='tmp_tbl')
        assert ['id', 'val'] == res.column_names()
        assert [int, str] == res.column_types()
        assert {'id': 2, 'val': 'b'} == res[0]
        assert {'id': 3, 'val': 'c'} == res[1]
