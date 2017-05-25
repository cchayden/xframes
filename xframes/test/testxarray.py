from __future__ import absolute_import

import unittest
import pytest
import math
import os
import array
import datetime
import pickle
import shutil

# python testxarray.py
# python -m unittest testxarray
# python -m unittest testxarray.TestXArrayVersion
# python -m unittest testxarray.TestXArrayVersion.test_version

from xframes import XArray
from xframes import XFrame
from xframes import object_utils


def delete_file_or_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.isfile(path):
        os.remove(path)


# noinspection PyClassHasNoInit
class TestXArrayVersion:
    """
    Tests XArray version
    """

    def test_version(self):
        ver = object_utils.version()
        assert type(ver) == str


# noinspection PyClassHasNoInit
class TestXArrayConstructorLocal:
    """
    Tests XArray constructors that create data from local sources.
    """

    def test_construct_list_int_infer(self):
        t = XArray([1, 2, 3])
        assert len(t) == 3
        assert t[0] == 1
        assert t.dtype() is int

    def test_construct_list_int(self):
        t = XArray([1, 2, 3], dtype=int)
        assert len(t) == 3
        assert t[0] == 1
        assert t.dtype() is int

    def test_construct_list_str_infer(self):
        t = XArray(['a', 'b', 'c'])
        assert len(t) == 3
        assert t[0] == 'a'
        assert t.dtype() is str

    def test_construct_list_str(self):
        t = XArray([1, 2, 3], dtype=str)
        assert len(t) == 3
        assert t[0] == '1'
        assert t.dtype() is str

    def test_construct_list_float_infer(self):
        t = XArray([1.0, 2.0, 3.0])
        assert len(t) == 3
        assert t[0] == 1.0
        assert t.dtype() is float

    def test_construct_list_float(self):
        t = XArray([1, 2, 3], dtype=float)
        assert len(t) == 3
        assert t[0] == 1.0
        assert t.dtype() is float

    def test_construct_list_bool_infer(self):
        t = XArray([True, False])
        assert len(t) == 2
        assert t[0] is True
        assert t.dtype() is bool

    def test_construct_list_bool(self):
        t = XArray([True, False], dtype=bool)
        assert len(t) == 2
        assert t[0] is True
        assert t.dtype() is bool

    def test_construct_list_list_infer(self):
        t = XArray([[1, 2, 3], [10]])
        assert len(t) == 2
        assert t[0] == [1, 2, 3]
        assert t[1] == [10]
        assert t.dtype() is list

    def test_construct_list_list(self):
        t = XArray([[1, 2, 3], [10]], dtype=list)
        assert len(t) == 2
        assert t[0] == [1, 2, 3]
        assert t[1] == [10]
        assert t.dtype() is list

    def test_construct_list_dict_infer(self):
        t = XArray([{'a': 1, 'b': 2}, {'x': 10}])
        assert len(t) == 2
        assert t[0] == {'a': 1, 'b': 2}
        assert t.dtype() is dict

    def test_construct_list_dict(self):
        t = XArray([{'a': 1, 'b': 2}, {'x': 10}], dtype=dict)
        assert len(t) == 2
        assert t[0] == {'a': 1, 'b': 2}
        assert t.dtype() is dict

    def test_construct_empty_list_infer(self):
        t = XArray([])
        assert len(t) == 0
        assert t.dtype() is None
    
    def test_construct_empty_list(self):
        t = XArray([], dtype=int)
        assert len(t) == 0
        assert t.dtype() is int

    def test_construct_list_int_cast_fail(self):
        with pytest.raises(ValueError):
            t = XArray(['a', 'b', 'c'], dtype=int)
            len(t)     # force materialization

    def test_construct_list_int_cast_ignore(self):
        t = XArray(['1', '2', 'c'], dtype=int, ignore_cast_failure=True)
        assert len(t) == 3
        assert t[0] == 1
        assert t[2] is None
        assert t.dtype() is int


# noinspection PyClassHasNoInit
class TestXArrayConstructorRange:
    """
    Tests XArray constructors for sequential ranges.
    """

    # noinspection PyArgumentList
    def test_construct_none(self):
        with pytest.raises(TypeError):
            XArray.from_sequence()

    # noinspection PyTypeChecker
    def test_construct_nonint_stop(self):
        with pytest.raises(TypeError):
            XArray.from_sequence(1.0)

    # noinspection PyTypeChecker
    def test_construct_nonint_start(self):
        with pytest.raises(TypeError):
            XArray.from_sequence(1.0, 10.0)

    def test_construct_stop(self):
        t = XArray.from_sequence(100, 200)
        assert len(t) == 100
        assert t[0] == 100
        assert t.dtype() is int

    def test_construct_start(self):
        t = XArray.from_sequence(100)
        assert len(t) == 100
        assert t[0] == 0
        assert t.dtype() is int


# noinspection PyClassHasNoInit
class TestXArrayConstructFromRdd:
    """
    Tests XArray from_rdd class method
    """

    def test_construct_from_rdd(self):
        # TODO test
        pass


# noinspection PyClassHasNoInit
class TestXArrayConstructorLoad:
    """
    Tests XArray constructors that loads from file.
    """

    def test_construct_local_file_int(self):
        t = XArray('files/test-array-int')
        assert len(t) == 4
        assert t.dtype() is int
        assert t[0] == 1

    def test_construct_local_file_float(self):
        t = XArray('files/test-array-float')
        assert len(t) == 4
        assert t.dtype() is float
        assert t[0] == 1.0

    def test_construct_local_file_str(self):
        t = XArray('files/test-array-str')
        assert len(t) == 4
        assert t.dtype() is str
        assert t[0] == 'a'

    def test_construct_local_file_list(self):
        t = XArray('files/test-array-list')
        assert len(t) == 4
        assert t.dtype() is list
        assert t[0] == [1, 2]

    def test_construct_local_file_dict(self):
        t = XArray('files/test-array-dict')
        assert len(t) == 4
        assert t.dtype() is dict
        assert t[0] == {1: 'a', 2: 'b'}

    def test_construct_local_file_datetime(self):
        t = XArray('files/test-array-datetime')
        assert len(t) == 3
        assert t.dtype() is datetime.datetime
        assert t[0] == datetime.datetime(2015, 8, 15)
        assert t[1] == datetime.datetime(2016, 9, 16)
        assert t[2] == datetime.datetime(2017, 10, 17)

    def test_construct_local_file_not_exist(self):
        with pytest.raises(ValueError):
            _ = XArray('files/does-not-exist')


# noinspection PyClassHasNoInit
class TestXArrayReadText:
    """
    Tests XArray read_text class method
    """

    def test_read_text(self):
        t = XArray.read_text('files/test-array-int')
        assert len(t) == 4
        assert list(t) == ['1', '2', '3', '4']


# noinspection PyClassHasNoInit
class TestXArrayFromConst:
    """
    Tests XArray constructed from const.
    """

    def test_from_const_int(self):
        t = XArray.from_const(1, 10)
        assert len(t) == 10
        assert t[0] == 1
        assert t.dtype() is int

    def test_from_const_float(self):
        t = XArray.from_const(1.0, 10)
        assert len(t) == 10
        assert t[0] == 1.0
        assert t.dtype() is float

    def test_from_const_str(self):
        t = XArray.from_const('a', 10)
        assert len(t) == 10
        assert t[0] == 'a'
        assert t.dtype() is str

    def test_from_const_datetime(self):
        t = XArray.from_const(datetime.datetime(2015, 10, 11), 10)
        assert len(t) == 10
        assert t[0] == datetime.datetime(2015, 10, 11)
        assert t.dtype() is datetime.datetime

    def test_from_const_list(self):
        t = XArray.from_const([1, 2], 10)
        assert len(t) == 10
        assert t[0] == [1, 2]
        assert t.dtype() is list

    def test_from_const_dict(self):
        t = XArray.from_const({1: 'a'}, 10)
        assert len(t) == 10
        assert t[0] == {1: 'a'}
        assert t.dtype() is dict

    def test_from_const_negint(self):
        with pytest.raises(ValueError):
            XArray.from_const(1, -10)

    # noinspection PyTypeChecker
    def test_from_const_nonint(self):
        with pytest.raises(TypeError):
            XArray.from_const(1, 'a')

    def test_from_const_bad_type(self):
        with pytest.raises(TypeError):
            XArray.from_const((1, 1), 10)


# noinspection PyClassHasNoInit
class TestXArraySaveBinary:
    """
    Tests XArray save binary format
    """
    def test_save(self, tmpdir):
        t = XArray([1, 2, 3])
        path = os.path.join(str(tmpdir), 'array-binary')
        t.save(path)
        success_path = os.path.join(path, '_SUCCESS')
        assert os.path.isfile(success_path)

    def test_save_format(self, tmpdir):
        t = XArray([1, 2, 3])
        path = os.path.join(str(tmpdir), 'array-binary')
        t.save(path, format='binary')
        success_path = os.path.join(path, '_SUCCESS')
        assert os.path.isfile(success_path)

    def test_save_not_exist(self, tmpdir):
        t = XArray([1, 2, 3])
        path = os.path.join(str(tmpdir), 'xxx/does-not-exist')
        t.save(path, format='binary')
        assert os.path.isdir(path)


# noinspection PyClassHasNoInit
class TestXArraySaveText:
    """
    Tests XArray save text format
    """
    def test_save(self, tmpdir):
        t = XArray([1, 2, 3])
        path = os.path.join(str(tmpdir), 'array-text.txt')
        t.save(path)
        success_path = os.path.join(path, '_SUCCESS')
        assert os.path.isfile(success_path)

    def test_save_format(self, tmpdir):
        t = XArray([1, 2, 3])
        path = os.path.join(str(tmpdir), 'array-text')
        t.save(path, format='text')
        success_path = os.path.join(path, '_SUCCESS')
        assert os.path.isfile(success_path)


# noinspection PyClassHasNoInit
class TestXArraySaveCsv:
    """
    Tests XArray save csv format
    """
    def test_save(self, tmpdir):
        t = XArray([1, 2, 3])
        path = os.path.join(str(tmpdir), 'array-csv.csv')
        t.save(path)
        with open(path) as f:
            assert f.readline().strip() == '1'
            assert f.readline().strip() == '2'
            assert f.readline().strip() == '3'

    def test_save_format(self, tmpdir):
        t = XArray([1, 2, 3])
        path = os.path.join(str(tmpdir), 'array-csv.csv')
        t.save(path, format='csv')
        with open(path) as f:
            assert f.readline().strip() == '1'
            assert f.readline().strip() == '2'
            assert f.readline().strip() == '3'


# noinspection PyClassHasNoInit
class TestXArrayRepr:
    """
    Tests XArray __repr__ function.
    """
    def test_repr(self):
        t = XArray([1, 2, 3])
        s = t.__repr__()
        assert s == """dtype: int
Rows: 3
[1, 2, 3]"""


# noinspection PyClassHasNoInit
class TestXArrayStr:
    """
    Tests XArray __str__ function.
    """
    def test_str(self):
        t = XArray(range(200))
        s = t.__repr__()
        assert s == "dtype: int\nRows: 200\n[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11," + \
                    " 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25," + \
                    " 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41," + \
                    " 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57," + \
                    " 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73," + \
                    " 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90," + \
                    " 91, 92, 93, 94, 95, 96, 97, 98, 99, ... ]"


# noinspection PyClassHasNoInit
class TestXArrayNonzero:
    """
    Tests XArray __nonzero__ function
    """
    def test_nonzero_nonzero(self):
        t = XArray([0])
        assert bool(t) is True

    def test_nonzero_zero(self):
        t = XArray([])
        assert bool(t) is False


# noinspection PyClassHasNoInit
class TestXArrayLen:
    """
    Tests XArray __len__ function
    """
    def test_len_nonzero(self):
        t = XArray([0])
        assert len(t) == 1

    def test_len_zero(self):
        t = XArray([])
        assert len(t) == 0


# noinspection PyClassHasNoInit
class TestXArrayIterator:
    """
    Tests XArray iteration function
    """
    def test_iter_empty(self):
        t = XArray([])
        for _ in t:
            assert False  # should not iterate

    def test_iter_1(self):
        t = XArray([0])
        for elem in t:
            assert elem == 0

    def test_iter_3(self):
        t = XArray([0, 1, 2])
        for elem, expect in zip(t, [0, 1, 2]):
            assert elem == expect


# noinspection PyClassHasNoInit
class TestXArrayAddScalar:
    """
    Tests XArray Scalar Addition
    """
    # noinspection PyTypeChecker,PyUnresolvedReferences
    def test_add_scalar(self):
        t = XArray([1, 2, 3])
        assert len(t) == 3
        assert t[0] == 1
        assert t.dtype() is int
        t = t + 2
        assert t[0] == 3
        assert t[1] == 4
        assert t[2] == 5


# noinspection PyClassHasNoInit
class TestXArrayAddVector:
    """
    Tests XArray Vector Addition
    """
    def test_add_vector(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([4, 5, 6])
        t = t1 + t2
        assert len(t) == 3
        assert t.dtype() is int
        assert t[0] == 5
        assert t[1] == 7
        assert t[2] == 9

    def test_add_vector_safe(self):
        t1 = XArray([1, 2, 3])
        t = t1 + t1
        assert len(t) == 3
        assert t.dtype() is int
        assert t[0] == 2
        assert t[1] == 4
        assert t[2] == 6

        
# noinspection PyClassHasNoInit
class TestXArrayOpScalar:
    """
    Tests XArray Scalar operations other than addition
    """
    # noinspection PyTypeChecker
    def test_sub_scalar(self):
        t = XArray([1, 2, 3])
        res = t - 1
        assert res[0] == 0
        assert res[1] == 1
        assert res[2] == 2

    # noinspection PyTypeChecker
    def test_mul_scalar(self):
        t = XArray([1, 2, 3])
        res = t * 2
        assert res[0] == 2
        assert res[1] == 4
        assert res[2] == 6

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def test_div_scalar(self):
        t = XArray([1, 2, 3])
        res = t / 2
        assert res[0] == 0
        assert res[1] == 1
        assert res[2] == 1

    # noinspection PyTypeChecker
    def test_pow_scalar(self):
        t = XArray([1, 2, 3])
        res = t ** 2
        assert res[0] == 1
        assert res[1] == 4
        assert res[2] == 9

    # noinspection PyUnresolvedReferences
    def test_lt_scalar(self):
        t = XArray([1, 2, 3])
        res = t < 3
        assert res[0] is True
        assert res[1] is True
        assert res[2] is False

    # noinspection PyUnresolvedReferences
    def test_le_scalar(self):
        t = XArray([1, 2, 3])
        res = t <= 2
        assert res[0] is True
        assert res[1] is True
        assert res[2] is False

    # noinspection PyUnresolvedReferences
    def test_gt_scalar(self):
        t = XArray([1, 2, 3])
        res = t > 2
        assert res[0] is False
        assert res[1] is False
        assert res[2] is True

    # noinspection PyUnresolvedReferences
    def test_ge_scalar(self):
        t = XArray([1, 2, 3])
        res = t >= 3
        assert res[0] is False
        assert res[1] is False
        assert res[2] is True

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def test_radd_scalar(self):
        t = XArray([1, 2, 3])
        res = 1 + t
        assert res[0] == 2
        assert res[1] == 3
        assert res[2] == 4

    # noinspection PyUnresolvedReferences
    def test_rsub_scalar(self):
        t = XArray([1, 2, 3])
        res = 1 - t
        assert res[0] == 0
        assert res[1] == -1
        assert res[2] == -2

    # noinspection PyUnresolvedReferences
    def test_rmul_scalar(self):
        t = XArray([1, 2, 3])
        res = 2 * t
        assert res[0] == 2
        assert res[1] == 4
        assert res[2] == 6

    # noinspection PyUnresolvedReferences
    def test_rdiv_scalar(self):
        t = XArray([1, 2, 3])
        res = 12 / t
        assert res[0] == 12
        assert res[1] == 6
        assert res[2] == 4

    # noinspection PyUnresolvedReferences
    def test_eq_scalar(self):
        t = XArray([1, 2, 3])
        res = t == 2
        assert res[0] is False
        assert res[1] is True
        assert res[2] is False

    # noinspection PyUnresolvedReferences
    def test_ne_scalar(self):
        t = XArray([1, 2, 3])
        res = t != 2
        assert res[0] is True
        assert res[1] is False
        assert res[2] is True

    def test_and_scalar(self):
        t = XArray([1, 2, 3])
        with pytest.raises(TypeError):
            _ = t & True

    def test_or_scalar(self):
        t = XArray([1, 2, 3])
        with pytest.raises(TypeError):
            _ = t | False


# noinspection PyUnresolvedReferences
# noinspection PyClassHasNoInit
class TestXArrayOpVector:
    """
    Tests XArray Vector operations other than addition
    """
    def test_sub_vector(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([4, 5, 6])
        t = t2 - t1
        assert t[0] == 3
        assert t[1] == 3
        assert t[2] == 3

    def test_mul_vector(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([4, 5, 6])
        res = t1 * t2
        assert res[0] == 4
        assert res[1] == 10
        assert res[2] == 18

    def test_div_vector(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([4, 6, 12])
        res = t2 / t1
        assert res[0] == 4
        assert res[1] == 3
        assert res[2] == 4

    def test_lt_vector(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([4, 2, 2])
        res = t1 < t2
        assert res[0] is True
        assert res[1] is False
        assert res[2] is False

    def test_le_vector(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([4, 2, 2])
        res = t1 <= t2
        assert res[0] is True
        assert res[1] is True
        assert res[2] is False

    def test_gt_vector(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([4, 2, 2])
        res = t1 > t2
        assert res[0] is False
        assert res[1] is False
        assert res[2] is True

    def test_ge_vector(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([4, 2, 2])
        res = t1 >= t2
        assert res[0] is False
        assert res[1] is True
        assert res[2] is True

    def test_eq_vector(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([4, 2, 2])
        res = t1 == t2
        assert res[0] is False
        assert res[1] is True
        assert res[2] is False

    def test_ne_vector(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([4, 2, 2])
        res = t1 != t2
        assert res[0] is True
        assert res[1] is False
        assert res[2] is True

    def test_and_vector(self):
        t1 = XArray([False, False, True, True])
        t2 = XArray([False, True, False, True])
        res = t1 & t2
        assert res[0] is False
        assert res[1] is False
        assert res[2] is False
        assert res[3] is True

    def test_or_vector(self):
        t1 = XArray([False, False, True, True])
        t2 = XArray([False, True, False, True])
        res = t1 | t2
        assert res[0] is False
        assert res[1] is True
        assert res[2] is True
        assert res[3] is True


# noinspection PyClassHasNoInit
class TestXArrayOpUnary:
    """
    Tests XArray Unary operations
    """
    def test_neg_unary(self):
        t = XArray([1, -2, 3])
        res = -t
        assert res[0] == -1
        assert res[1] == 2
        assert res[2] == -3

    def test_pos_unary(self):
        t = XArray([1, -2, 3])
        res = +t
        assert res[0] == 1
        assert res[1] == -2
        assert res[2] == 3

    # noinspection PyTypeChecker
    def test_abs_unary(self):
        t = XArray([1, -2, 3])
        res = abs(t)
        assert res[0] == 1
        assert res[1] == 2
        assert res[2] == 3


# noinspection PyClassHasNoInit
class TestXArrayLogicalFilter:
    """
    Tests XArray logical filter (XArray indexed by XArray)
    """
    def test_logical_filter_array(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([1, 0, 1])
        res = t1[t2]
        assert len(res) == 2
        assert res[0] == 1
        assert res[1] == 3

    def test_logical_filter_test(self):
        t1 = XArray([1, 2, 3])
        res = t1[t1 != 2]
        assert len(res) == 2
        assert res[0] == 1
        assert res[1] == 3

    def test_logical_filter_len_error(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([1, 0])
        with pytest.raises(IndexError):
            _ = t1[t2]


# noinspection PyClassHasNoInit
class TestXArrayCopyRange:
    """
    Tests XArray integer and range indexing
    """
    def test_copy_range_pos(self):
        t = XArray([1, 2, 3])
        assert t[0] == 1

    def test_copy_range_neg(self):
        t = XArray([1, 2, 3])
        assert t[-1] == 3

    def test_copy_range_index_err(self):
        t = XArray([1, 2, 3])
        with pytest.raises(IndexError):
            _ = t[3]
        
    def test_copy_range_slice(self):
        t = XArray([1, 2, 3])
        res = t[0:2]
        assert len(res) == 2
        assert res[0] == 1
        assert res[1] == 2

    def test_copy_range_slice_neg_start(self):
        t = XArray([1, 2, 3, 4, 5])
        res = t[-3:4]
        assert len(res) == 2
        assert res[0] == 3
        assert res[1] == 4

    def test_copy_range_slice_neg_stop(self):
        t = XArray([1, 2, 3, 4, 5])
        res = t[1:-2]
        assert len(res) == 2
        assert res[0] == 2
        assert res[1] == 3

    def test_copy_range_slice_stride(self):
        t = XArray([1, 2, 3, 4, 5])
        res = t[0:4:2]
        assert len(res) == 2
        assert res[0] == 1
        assert res[1] == 3

    def test_copy_range_bad_type(self):
        t = XArray([1, 2, 3])
        with pytest.raises(IndexError):
            _ = t[{1, 2, 3}]


# noinspection PyClassHasNoInit
class TestXArraySize:
    """
    Tests XArray size operation
    """
    def test_size(self):
        t = XArray([1, 2, 3])
        assert t.size() == 3


# noinspection PyClassHasNoInit
class TestXArrayDtype:
    """
    Tests XArray dtype operation
    """
    def test_dtype(self):
        t = XArray([1, 2, 3])
        assert t.dtype() is int


# noinspection PyClassHasNoInit
class TestXArrayTableLineage:
    """
    Tests XArray ltable lineage operation
    """
    def test_lineage_program(self):
        res = XArray([1, 2, 3])
        lineage = res.lineage()['table']
        assert len(lineage) == 1
        item = list(lineage)[0]
        assert item == 'PROGRAM'

    def test_lineage_file(self):
        res = XArray('files/test-array-int')
        lineage = res.lineage()['table']
        assert len(lineage) == 1
        item = os.path.basename(list(lineage)[0])
        assert item == 'test-array-int'

    def test_lineage_apply(self):
        res = XArray('files/test-array-int').apply(lambda x: -x)
        lineage = res.lineage()['table']
        assert len(lineage) == 1
        item = os.path.basename(list(lineage)[0])
        assert item == 'test-array-int'

    def test_lineage_range(self):
        res = XArray.from_sequence(100, 200)
        lineage = res.lineage()['table']
        assert len(lineage) == 1
        item = list(lineage)[0]
        assert item == 'RANGE'

    def test_lineage_const(self):
        res = XArray.from_const(1, 10)
        lineage = res.lineage()['table']
        assert len(lineage) == 1
        item = list(lineage)[0]
        assert item == 'CONST'

    def test_lineage_binary_op(self):
        res_int = XArray('files/test-array-int')
        res_float = XArray('files/test-array-float')
        res = res_int + res_float
        lineage = res.lineage()['table']
        assert len(lineage) == 2
        basenames = set([os.path.basename(item) for item in lineage])
        assert 'test-array-int' in basenames
        assert 'test-array-float' in basenames

    # noinspection PyAugmentAssignment,PyUnresolvedReferences
    def test_lineage_left_op(self):
        res = XArray('files/test-array-int')
        res = res + 2
        lineage = res.lineage()['table']
        assert len(lineage) == 1
        item = os.path.basename(list(lineage)[0])
        assert item == 'test-array-int'

    # noinspection PyAugmentAssignment,PyUnresolvedReferences
    def test_lineage_right_op(self):
        res = XArray('files/test-array-int')
        res = 2 + res
        lineage = res.lineage()['table']
        assert len(lineage) == 1
        item = os.path.basename(list(lineage)[0])
        assert item == 'test-array-int'

    def test_lineage_unary(self):
        res = XArray('files/test-array-int')
        res = -res
        lineage = res.lineage()['table']
        assert len(lineage) == 1
        item = os.path.basename(list(lineage)[0])
        assert item == 'test-array-int'

    def test_lineage_append(self):
        res1 = XArray('files/test-array-int')
        res2 = XArray('files/test-array-float')
        res3 = res2.apply(lambda x: int(x))
        res = res1.append(res3)
        lineage = res.lineage()['table']
        assert len(lineage) == 2
        basenames = set([os.path.basename(item) for item in lineage])
        assert 'test-array-int' in basenames
        assert 'test-array-float' in basenames

    def test_lineage_save(self):
        res = XArray('files/test-array-int')
        path = '/tmp/xarray'
        res.save(path, format='binary')
        with open(os.path.join(path, '_metadata')) as f:
            metadata = pickle.load(f)
        assert metadata is int
        with open(os.path.join(path, '_lineage')) as f:
            lineage = pickle.load(f)
            table_lineage = lineage[0]
            assert len(table_lineage) == 1
            basenames = set([os.path.basename(item) for item in table_lineage])
            assert 'test-array-int' in basenames

    def test_lineage_save_text(self):
        res = XArray('files/test-array-str')
        path = '/tmp/xarray'
        res.save(path, format='text')
        with open(os.path.join(path, '_metadata')) as f:
            metadata = pickle.load(f)
        assert metadata is str
        with open(os.path.join(path, '_lineage')) as f:
            lineage = pickle.load(f)
            table_lineage = lineage[0]
            assert len(table_lineage) == 1
            basenames = set([os.path.basename(item) for item in table_lineage])
            assert 'test-array-str' in basenames

    def test_lineage_load(self):
        res = XArray('files/test-array-int')
        path = 'tmp/array'
        res.save(path, format='binary')
        res = XArray(path)
        lineage = res.lineage()['table']
        assert len(lineage) == 1
        basenames = set([os.path.basename(item) for item in lineage])
        assert 'test-array-int' in basenames


# noinspection PyClassHasNoInit
class TestXArrayColumnLineage:
    """
    Tests XArray column lineage operation
    """

    # helper function
    # noinspection PyMethodMayBeStatic
    def get_column_lineage(self, xa, keys=None):
        # noinspection PyUnusedLocal
        __tracebackhide__ = True
        lineage = xa.lineage()['column']
        keys = keys or ['_XARRAY']
        keys = sorted(keys)
        count = len(keys)
        assert len(lineage) == count
        assert sorted(lineage.keys()) == keys
        return lineage

    def test_construct_empty(self):
        t = XArray()
        lineage = self.get_column_lineage(t)
        assert lineage['_XARRAY'] == {('EMPTY', '_XARRAY')}

    # noinspection PyTypeChecker
    def test_construct_from_xarrayt(self):
        t = XArray([1, 2, 3])
        res = XArray(t)
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY')}

    def test_construct_list_int(self):
        t = XArray([1, 2, 3])
        lineage = self.get_column_lineage(t)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY')}

    def test_construct_from_const(self):
        t = XArray.from_const(1, 3)
        lineage = self.get_column_lineage(t)
        assert lineage['_XARRAY'] == {('CONST', '_XARRAY')}

    def test_lineage_file(self):
        path = 'files/test-array-int'
        realpath = os.path.realpath(path)
        res = XArray(path)
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {(realpath, '_XARRAY')}

    def test_save(self):
        t = XArray([1, 2, 3])
        path = 'tmp/array-binary'
        t.save(path)
        lineage = self.get_column_lineage(t)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY')}

    def test_save_as_text(self):
        t = XArray([1, 2, 3])
        path = 'tmp/array-text'
        t.save(path, format='text')
        lineage = self.get_column_lineage(t)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY')}

    def test_from_rdd(self):
        t = XArray([1, 2, 3])
        rdd = t.to_rdd()
        res = XArray.from_rdd(rdd, int)
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('RDD', '_XARRAY')}

    def test_topk_index(self):
        t = XArray([1, 2, 3])
        res = t.topk_index(0)
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY')}

    def test_add_vector(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([4, 5, 6])
        res = t1 + t2
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY')}

    # noinspection PyTypeChecker
    def test_add_scalar(self):
        t = XArray([1, 2, 3])
        assert len(t) == 3
        assert t[0] == 1
        assert t.dtype() is int
        res = t + 2
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY')}

    def test_sample_no_seed(self):
        t = XArray(range(10))
        res = t.sample(0.3)
        assert len(res) < 10
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY')}

    def test_logical_filter_array(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([1, 0, 1])
        res = t1[t2]
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY')}

    def test_from_sequence(self):
        res = XArray.from_sequence(100, 200)
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('RANGE', '_XARRAY')}

    def test_range(self):
        t = XArray([1, 2, 3])
        res = t[1:2]
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('RANGE', '_XARRAY')}

    def test_filter(self):
        t = XArray([1, 2, 3])
        res = t.filter(lambda x: x == 2)
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY')}

    def test_append(self):
        t = XArray([1, None, 3])
        path = 'files/test-array-int'
        realpath = os.path.realpath(path)
        u = XArray(path)
        res = t.append(u)
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY'), (realpath, '_XARRAY')}

    def test_apply(self):
        t = XArray([1, 2, 3])
        res = t.apply(lambda x: x * 2)
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY')}

    def test_flat_map(self):
        t = XArray([[1], [1, 2], [1, 2, 3]])
        res = t.flat_map(lambda x: x)
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY')}

    def test_astype_int_float(self):
        t = XArray([1, 2, 3])
        res = t.astype(float)
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY')}

    def test_clip_int_clip(self):
        t = XArray([1, 2, 3])
        res = t.clip(2, 2)
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY')}

    def test_fillna(self):
        t = XArray([1, 2, 3])
        res = t.fillna(10)
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY')}

    def test_unpack_list(self):
        t = XArray([[1, 0, 1],
                    [1, 1, 1],
                    [0, 1]])
        res = t.unpack()
        lineage = self.get_column_lineage(res, ['X.0', 'X.1', 'X.2'])
        assert 'X.0' in lineage
        assert 'X.1' in lineage
        assert lineage['X.0'] == {('PROGRAM', '_XARRAY')}
        assert lineage['X.1'] == {('PROGRAM', '_XARRAY')}

    def test_sort(self):
        t = XArray([3, 2, 1])
        res = t.sort()
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY')}

    def test_split_datetime_all(self):
        t = XArray([datetime.datetime(2011, 1, 1, 1, 1, 1),
                    datetime.datetime(2012, 2, 2, 2, 2, 2),
                    datetime.datetime(2013, 3, 3, 3, 3, 3)])
        res = t.split_datetime('date')
        lineage = self.get_column_lineage(res, ['date.year', 'date.month', 'date.day',
                                                'date.hour', 'date.minute', 'date.second'])
        assert 'date.year' in lineage
        assert 'date.month' in lineage
        assert 'date.day' in lineage
        assert lineage['date.year'] == {('RDD', 'date.year')}
        assert lineage['date.month'] == {('RDD', 'date.month')}
        assert lineage['date.day'] == {('RDD', 'date.day')}

    def test_datetime_to_str(self):
        t = XArray([datetime.datetime(2015, 8, 21),
                    datetime.datetime(2016, 9, 22),
                    datetime.datetime(2017, 10, 23)])
        res = t.datetime_to_str('%Y %m %d')
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY')}

    def test_str_to_datetime(self):
        t = XArray(['2015 08 21', '2015 08 22', '2015 08 23'])
        res = t.str_to_datetime('%Y %m %d')
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY')}

    def test_dict_trim_by_keys_include(self):
        t = XArray([{'a': 0, 'b': 0, 'c': 0}, {'x': 1}])
        res = t.dict_trim_by_keys(['a'], exclude=False)
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY')}

    def test_dict_trim_by_values(self):
        t = XArray([{'a': 0, 'b': 1, 'c': 2, 'd': 3}, {'x': 1}])
        res = t.dict_trim_by_values(1, 2)
        lineage = self.get_column_lineage(res)
        assert lineage['_XARRAY'] == {('PROGRAM', '_XARRAY')}

    def test_dict_keys(self):
        t = XArray([{'a': 0, 'b': 0, 'c': 0}, {'x': 1, 'y': 2, 'z': 3}])
        res = t.dict_keys()
        lineage = self.get_column_lineage(res, ['X.0', 'X.1', 'X.2'])
        assert 'X.0' in lineage
        assert 'X.1' in lineage
        assert 'X.2' in lineage
        assert lineage['X.0'] == {('RDD', 'X.0')}
        assert lineage['X.1'] == {('RDD', 'X.1')}

    def test_values(self):
        t = XArray([{'a': 0, 'b': 1, 'c': 2}, {'x': 10, 'y': 20, 'z': 30}])
        res = t.dict_values()
        lineage = self.get_column_lineage(res, ['X.0', 'X.1', 'X.2'])
        assert 'X.0' in lineage
        assert 'X.1' in lineage
        assert 'X.2' in lineage
        assert lineage['X.0'] == {('RDD', 'X.0')}
        assert lineage['X.1'] == {('RDD', 'X.1')}


# noinspection PyClassHasNoInit
class TestXArrayHead:
    """
    Tests XArray head operation
    """
    def test_head(self):
        t = XArray([1, 2, 3])
        assert len(t.head()) == 3

    def test_head_10(self):
        t = XArray(range(100))
        assert len(t.head()) == 10

    def test_head_5(self):
        t = XArray(range(100))
        assert len(t.head(5)) == 5


# noinspection PyClassHasNoInit
class TestXArrayVectorSlice:
    """
    Tests XArray vector_slice operation
    """
    def test_vector_slice_start_0(self):
        t = XArray([[1, 2, 3], [10, 11, 12]])
        res = t.vector_slice(0)
        assert len(res) == 2
        assert res[0] == 1
        assert res[1] == 10

    def test_vector_slice_start_1(self):
        t = XArray([[1, 2, 3], [10, 11, 12]])
        res = t.vector_slice(1)
        assert len(res) == 2
        assert res[0] == 2
        assert res[1] == 11

    def test_vector_slice_start_end(self):
        t = XArray([[1, 2, 3], [10, 11, 12]])
        res = t.vector_slice(0, 2)
        assert len(res) == 2
        assert res[0] == [1, 2]
        assert res[1] == [10, 11]

    def test_vector_slice_start_none(self):
        t = XArray([[1], [1, 2], [1, 2, 3]])
        res = t.vector_slice(2)
        assert len(res) == 3
        assert res[0] is None
        assert res[1] is None
        assert res[2] == 3

    def test_vector_slice_start_end_none(self):
        t = XArray([[1], [1, 2], [1, 2, 3]])
        res = t.vector_slice(0, 2)
        assert len(res) == 3
        assert res[0] is None
        assert res[1] == [1, 2]
        assert res[2] == [1, 2]


# noinspection PyClassHasNoInit
class TestXArrayCountWords:
    """
    Tests XArray count_words
    """
    def test_count_words(self):
        pass


# noinspection PyClassHasNoInit
class TestXArrayCountNgrams:
    """
    Tests XArray count_ngrams
    """
    def test_count_ngrams(self):
        pass


# noinspection PyClassHasNoInit
class TestXArrayApply:
    """
    Tests XArray apply
    """
    def test_apply_int(self):
        t = XArray([1, 2, 3])
        res = t.apply(lambda x: x * 2)
        assert len(res) == 3
        assert res.dtype() is int
        assert res[0] == 2
        assert res[1] == 4
        assert res[2] == 6

    def test_apply_float_cast(self):
        t = XArray([1, 2, 3])
        res = t.apply(lambda x: x * 2, float)
        assert len(res) == 3
        assert res.dtype() is float
        assert res[0] == 2.0
        assert res[1] == 4.0
        assert res[2] == 6.0

    def test_apply_skip_undefined(self):
        t = XArray([1, 2, 3, None])
        res = t.apply(lambda x: x * 2)
        assert len(res) == 4
        assert res.dtype() is int
        assert res[0] == 2
        assert res[1] == 4
        assert res[2] == 6
        assert res[3] is None

    def test_apply_type_err(self):
        t = XArray([1, 2, 3, None])
        with pytest.raises(ValueError):
            t.apply(lambda x: x * 2, skip_undefined=False)

    def test_apply_fun_err(self):
        t = XArray([1, 2, 3, None])
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            t.apply(1)


# noinspection PyClassHasNoInit
class TestXArrayFlatMap:
    """
    Tests XArray flat_map
    """
    def test_flat_map(self):
        t = XArray([[1], [1, 2], [1, 2, 3]])
        res = t.flat_map(lambda x: x)
        assert len(res) == 6
        assert res.dtype() is int
        assert res[0] == 1
        assert res[1] == 1
        assert res[2] == 2
        assert res[3] == 1
        assert res[4] == 2
        assert res[5] == 3

    def test_flat_map_int(self):
        t = XArray([[1], [1, 2], [1, 2, 3]])
        res = t.flat_map(lambda x: [v * 2 for v in x])
        assert len(res) == 6
        assert res.dtype() is int
        assert res[0] == 2
        assert res[1] == 2
        assert res[2] == 4
        assert res[3] == 2
        assert res[4] == 4
        assert res[5] == 6

    def test_flat_map_str(self):
        t = XArray([['a'], ['a', 'b'], ['a', 'b', 'c']])
        res = t.flat_map(lambda x: x)
        assert len(res) == 6
        assert res.dtype() is str
        assert res[0] == 'a'
        assert res[1] == 'a'
        assert res[2] == 'b'
        assert res[3] == 'a'
        assert res[4] == 'b'
        assert res[5] == 'c'

    def test_flat_map_float_cast(self):
        t = XArray([[1], [1, 2], [1, 2, 3]])
        res = t.flat_map(lambda x: x, dtype=float)
        assert len(res) == 6
        assert res.dtype() is float
        assert res[0] == 1.0
        assert res[1] == 1.0
        assert res[2] == 2.0
        assert res[3] == 1.0
        assert res[4] == 2.0
        assert res[5] == 3.0

    def test_flat_map_skip_undefined(self):
        t = XArray([[1], [1, 2], [1, 2, 3], None, [None]])
        res = t.flat_map(lambda x: x)
        assert len(res) == 6
        assert res.dtype() is int
        assert res[0] == 1
        assert res[1] == 1
        assert res[2] == 2
        assert res[3] == 1
        assert res[4] == 2
        assert res[5] == 3

    def test_flat_map_no_fun(self):
        t = XArray([[1], [1, 2], [1, 2, 3]])
        res = t.flat_map()
        assert len(res) == 6
        assert res.dtype() is int
        assert res[0] == 1
        assert res[1] == 1
        assert res[2] == 2
        assert res[3] == 1
        assert res[4] == 2
        assert res[5] == 3

    def test_flat_map_type_err(self):
        t = XArray([[1], [1, 2], [1, 2, 3], [None]])
        with pytest.raises(ValueError):
            t.flat_map(lambda x: x * 2, skip_undefined=False)


# noinspection PyClassHasNoInit
class TestXArrayFilter:
    """
    Tests XArray filter
    """
    def test_filter(self):
        t = XArray([1, 2, 3])
        res = t.filter(lambda x: x == 2)
        assert len(res) == 1

    def test_filter_empty(self):
        t = XArray([1, 2, 3])
        res = t.filter(lambda x: x == 10)
        assert len(res) == 0


# noinspection PyClassHasNoInit
class TestXArraySample:
    """
    Tests XArray sample
    """
    def test_sample_no_seed(self):
        t = XArray(range(10))
        res = t.sample(0.3)
        assert len(res) < 10

    @unittest.skip('depends on number of partitions')
    def test_sample_seed(self):
        t = XArray(range(10))
        res = t.sample(0.3, seed=1)
        # get 3, 6, 9 with this seed
        assert len(res) == 3
        assert res[0] == 3
        assert res[1] == 6

    def test_sample_zero(self):
        t = XArray(range(10))
        res = t.sample(0.0)
        assert len(res) == 0

    def test_sample_err_gt(self):
        t = XArray(range(10))
        with pytest.raises(ValueError):
            t.sample(2, seed=1)

    def test_sample_err_lt(self):
        t = XArray(range(10))
        with pytest.raises(ValueError):
            t.sample(-0.5, seed=1)


# noinspection PyClassHasNoInit
class TestXArrayAll:
    """
    Tests XArray all
    """
    # int
    def test_all_int_none(self):
        t = XArray([1, None])
        assert t.all() is False

    def test_all_int_zero(self):
        t = XArray([1, 0])
        assert t.all() is False

    def test_all_int_true(self):
        t = XArray([1, 2])
        assert t.all() is True

    # float
    def test_all_float_nan(self):
        t = XArray([1.0, float('nan')])
        assert t.all() is False

    def test_all_float_none(self):
        t = XArray([1.0, None])
        assert t.all() is False

    def test_all_float_zero(self):
        t = XArray([1.0, 0.0])
        assert t.all() is False

    def test_all_float_true(self):
        t = XArray([1.0, 2.0])
        assert t.all() is True

    # str
    def test_all_str_empty(self):
        t = XArray(['hello', ''])
        assert t.all() is False

    def test_all_str_none(self):
        t = XArray(['hello', None])
        assert t.all() is False

    def test_all_str_true(self):
        t = XArray(['hello', 'world'])
        assert t.all() is True

    # list
    def test_all_list_empty(self):
        t = XArray([[1, 2], []])
        assert t.all() is False

    def test_all_list_none(self):
        t = XArray([[1, 2], None])
        assert t.all() is False

    def test_all_list_true(self):
        t = XArray([[1, 2], [2, 3]])
        assert t.all() is True

    # dict
    def test_all_dict_empty(self):
        t = XArray([{1: 'a'}, {}])
        assert t.all() is False

    def test_all_dict_none(self):
        t = XArray([{1: 'a'}, None])
        assert t.all() is False

    def test_all_dict_true(self):
        t = XArray([{1: 'a'}, {2: 'b'}])
        assert t.all() is True

    # empty
    def test_all_empty(self):
        t = XArray([])
        assert t.all() is True


# noinspection PyClassHasNoInit
class TestXArrayAny:
    """
    Tests XArray any
    """
    # int
    def test_any_int(self):
        t = XArray([1, 2])
        assert t.any() is True

    def test_any_int_true(self):
        t = XArray([0, 1])
        assert t.any() is True

    def test_any_int_false(self):
        t = XArray([0, 0])
        assert t.any() is False

    def test_any_int_missing_true(self):
        t = XArray([1, None])
        assert t.any() is True

    def test_any_int_missing_false(self):
        t = XArray([None, 0])
        assert t.any() is False

    # float
    def test_any_float(self):
        t = XArray([1., 2.])
        assert t.any() is True

    def test_any_float_true(self):
        t = XArray([0.0, 1.0])
        assert t.any() is True

    def test_any_float_false(self):
        t = XArray([0.0, 0.0])
        assert t.any() is False

    def test_any_float_missing_true(self):
        t = XArray([1.0, None])
        assert t.any() is True

    def test_any_float_missing_true_nan(self):
        t = XArray([1.0, float('nan')])
        assert t.any() is True

    def test_any_float_missing_true_none(self):
        t = XArray([1.0, None])
        assert t.any() is True

    def test_any_float_missing_false(self):
        t = XArray([None, 0.0])
        assert t.any() is False

    def test_any_float_missing_false_nan(self):
        t = XArray([float('nan'), 0.0])
        assert t.any() is False

    def test_any_float_missing_false_none(self):
        t = XArray([None, 0.0])
        assert t.any() is False

    # str
    def test_any_str(self):
        t = XArray(['a', 'b'])
        assert t.any() is True

    def test_any_str_true(self):
        t = XArray(['', 'a'])
        assert t.any() is True

    def test_any_str_false(self):
        t = XArray(['', ''])
        assert t.any() is False

    def test_any_str_missing_true(self):
        t = XArray(['a', None])
        assert t.any() is True

    def test_any_str_missing_false(self):
        t = XArray([None, ''])
        assert t.any() is False

    # list
    def test_any_list(self):
        t = XArray([[1], [2]])
        assert t.any() is True

    def test_any_list_true(self):
        t = XArray([[], ['a']])
        assert t.any() is True

    def test_any_list_false(self):
        t = XArray([[], []])
        assert t.any() is False

    def test_any_list_missing_true(self):
        t = XArray([['a'], None])
        assert t.any() is True

    def test_any_list_missing_false(self):
        t = XArray([None, []])
        assert t.any() is False

    # dict
    def test_any_dict(self):
        t = XArray([{'a': 1, 'b': 2}])
        assert t.any() is True

    def test_any_dict_true(self):
        t = XArray([{}, {'a': 1}])
        assert t.any() is True

    def test_any_dict_false(self):
        t = XArray([{}, {}])
        assert t.any() is False

    def test_any_dict_missing_true(self):
        t = XArray([{'a': 1}, None])
        assert t.any() is True

    def test_any_dict_missing_false(self):
        t = XArray([None, {}])
        assert t.any() is False

    # empty
    def test_any_empty(self):
        t = XArray([])
        assert t.any() is False


# noinspection PyClassHasNoInit
class TestXArrayMax:
    """
    Tests XArray max
    """
    def test_max_empty(self):
        t = XArray([])
        assert t.max() is None

    def test_max_err(self):
        t = XArray(['a'])
        with pytest.raises(TypeError):
            t.max()

    def test_max_int(self):
        t = XArray([1, 2, 3])
        assert t.max() == 3

    def test_max_float(self):
        t = XArray([1.0, 2.0, 3.0])
        assert t.max() == 3.0


# noinspection PyClassHasNoInit
class TestXArrayMin:
    """
    Tests XArray min
    """
    def test_min_empty(self):
        t = XArray([])
        assert t.min() is None

    def test_min_err(self):
        t = XArray(['a'])
        with pytest.raises(TypeError):
            t.min()

    def test_min_int(self):
        t = XArray([1, 2, 3])
        assert t.min() == 1

    def test_min_float(self):
        t = XArray([1.0, 2.0, 3.0])
        assert t.min() == 1.0


# noinspection PyClassHasNoInit
class TestXArraySum:
    """
    Tests XArray sum
    """
    def test_sum_empty(self):
        t = XArray([])
        assert t.sum() is None

    def test_sum_err(self):
        t = XArray(['a'])
        with pytest.raises(TypeError):
            t.sum()

    def test_sum_int(self):
        t = XArray([1, 2, 3])
        assert t.sum() == 6

    def test_sum_float(self):
        t = XArray([1.0, 2.0, 3.0])
        assert t.sum() == 6.0

    def test_sum_array(self):
        t = XArray([array.array('l', [10, 20, 30]), array.array('l', [40, 50, 60])])
        assert t.sum() == array.array('l', [50, 70, 90])

    def test_sum_list(self):
        t = XArray([[10, 20, 30], [40, 50, 60]])
        assert t.sum() == [50, 70, 90]

    def test_sum_dict(self):
        t = XArray([{'x': 1, 'y': 2}, {'x': 3, 'y': 4}])
        assert t.sum() == {'x': 4, 'y': 6}


# noinspection PyClassHasNoInit
class TestXArrayMean:
    """
    Tests XArray mean
    """
    def test_mean_empty(self):
        t = XArray([])
        assert t.mean() is None

    def test_mean_err(self):
        t = XArray(['a'])
        with pytest.raises(TypeError):
            t.mean()

    def test_mean_int(self):
        t = XArray([1, 2, 3])
        assert t.mean() == 2

    def test_mean_float(self):
        t = XArray([1.0, 2.0, 3.0])
        assert t.mean() == 2.0


# noinspection PyClassHasNoInit
class TestXArrayStd:
    """
    Tests XArray std
    """
    def test_std_empty(self):
        t = XArray([])
        assert t.std() is None

    def test_std_err(self):
        t = XArray(['a'])
        with pytest.raises(TypeError):
            t.std()

    def test_std_int(self):
        t = XArray([1, 2, 3])
        expect = math.sqrt(2.0 / 3.0)
        assert t.std() == expect

    def test_std_float(self):
        t = XArray([1.0, 2.0, 3.0])
        expect = math.sqrt(2.0 / 3.0)
        assert t.std() == expect


# noinspection PyClassHasNoInit
class TestXArrayVar:
    """
    Tests XArray var
    """
    def test_var_empty(self):
        t = XArray([])
        assert t.var() is None

    def test_var_err(self):
        t = XArray(['a'])
        with pytest.raises(TypeError):
            t.var()

    def test_var_int(self):
        t = XArray([1, 2, 3])
        expect = 2.0 / 3.0
        assert t.var() == expect

    def test_var_float(self):
        t = XArray([1.0, 2.0, 3.0])
        expect = 2.0 / 3.0
        assert t.var() == expect


# noinspection PyClassHasNoInit
class TestXArrayNumMissing:
    """
    Tests XArray num_missing
    """
    def test_num_missing_empty(self):
        t = XArray([])
        assert t.num_missing() == 0

    def test_num_missing_zero(self):
        t = XArray([1, 2, 3])
        assert t.num_missing() == 0

    def test_num_missing_int_none(self):
        t = XArray([1, 2, None])
        assert t.num_missing() == 1

    def test_num_missing_int_all(self):
        t = XArray([None, None, None], dtype=int)
        assert t.num_missing() == 3

    def test_num_missing_float_none(self):
        t = XArray([1.0, 2.0, None])
        assert t.num_missing() == 1

    def test_num_missing_float_nan(self):
        t = XArray([1.0, 2.0, float('nan')])
        assert t.num_missing() == 1


# noinspection PyClassHasNoInit
class TestXArrayNumNonzero:
    """
    Tests XArray nnz
    """
    def test_nnz_empty(self):
        t = XArray([])
        assert t.nnz() == 0

    def test_nnz_zero_int(self):
        t = XArray([0, 0, 0])
        assert t.nnz() == 0

    def test_nnz_zero_float(self):
        t = XArray([0.0, 0.0, 0.0])
        assert t.nnz() == 0

    def test_nnz_int_none(self):
        t = XArray([1, 2, None])
        assert t.nnz() == 2

    def test_nnz_int_all(self):
        t = XArray([None, None, None], dtype=int)
        assert t.nnz() == 0

    def test_nnz_float_none(self):
        t = XArray([1.0, 2.0, None])
        assert t.nnz() == 2

    def test_nnz_float_nan(self):
        t = XArray([1.0, 2.0, float('nan')])
        assert t.nnz() == 2


# noinspection PyClassHasNoInit
class TestXArrayDatetimeToStr:
    """
    Tests XArray datetime_to_str
    """
    def test_datetime_to_str(self):
        t = XArray([datetime.datetime(2015, 8, 21),
                    datetime.datetime(2016, 9, 22),
                    datetime.datetime(2017, 10, 23)])
        res = t.datetime_to_str('%Y %m %d')
        assert res.dtype() is str
        assert res[0] == '2015 08 21'
        assert res[1] == '2016 09 22'
        assert res[2] == '2017 10 23'

    def test_datetime_to_str_bad_type(self):
        t = XArray([1, 2, 3])
        with pytest.raises(TypeError):
            t.datetime_to_str('%Y %M %d')


# noinspection PyClassHasNoInit
class TestXArrayStrToDatetime:
    """
    Tests XArray str_to_datetime
    """
    def test_str_to_datetime(self):
        t = XArray(['2015 08 21', '2015 08 22', '2015 08 23'])
        res = t.str_to_datetime('%Y %m %d')
        assert res.dtype() is datetime.datetime
        assert res[0] == datetime.datetime(2015, 8, 21)
        assert res[1] == datetime.datetime(2015, 8, 22)
        assert res[2] == datetime.datetime(2015, 8, 23)

    def test_str_to_datetime_parse(self):
        t = XArray(['2015 8 21', '2015 Aug 22', '23 Aug 2015', 'Aug 24 2015'])
        res = t.str_to_datetime()
        assert res.dtype() is datetime.datetime
        assert res[0] == datetime.datetime(2015, 8, 21)
        assert res[1] == datetime.datetime(2015, 8, 22)
        assert res[2] == datetime.datetime(2015, 8, 23)
        assert res[3] == datetime.datetime(2015, 8, 24)

    def test_str_to_datetime_bad_type(self):
        t = XArray([1, 2, 3])
        with pytest.raises(TypeError):
            t.str_to_datetime()


# noinspection PyClassHasNoInit
class TestXArrayAstype:
    """
    Tests XArray astype
    """
    def test_astype_empty(self):
        t = XArray([])
        res = t.astype(int)
        assert res.dtype() is int

    def test_astype_int_int(self):
        t = XArray([1, 2, 3])
        res = t.astype(int)
        assert res.dtype() is int
        assert res[0] == 1

    def test_astype_int_float(self):
        t = XArray([1, 2, 3])
        res = t.astype(float)
        assert res.dtype() is float
        assert res[0] == 1.0

    def test_astype_float_float(self):
        t = XArray([1.0, 2.0, 3.0])
        res = t.astype(float)
        assert res.dtype() is float
        assert res[0] == 1.0

    def test_astype_float_int(self):
        t = XArray([1.0, 2.0, 3.0])
        res = t.astype(int)
        assert res.dtype() is int
        assert res[0] == 1

    def test_astype_int_str(self):
        t = XArray([1, 2, 3])
        res = t.astype(str)
        assert res.dtype() is str
        assert res[0] == '1'

    def test_astype_str_list(self):
        t = XArray(['[1, 2, 3]', '[4, 5, 6]'])
        res = t.astype(list)
        assert res.dtype() is list
        assert res[0] == [1, 2, 3]

    def test_astype_str_dict(self):
        t = XArray(['{"a": 1, "b": 2}', '{"x": 3}'])
        res = t.astype(dict)
        assert res.dtype() is dict
        assert res[0] == {'a': 1, 'b': 2}

    # noinspection PyTypeChecker
    def test_astype_str_array(self):
        t = XArray(['[1, 2, 3]', '[4, 5, 6]'])
        res = t.astype(array)
        assert res.dtype() is array
        assert list(res[0]) == [1, 2, 3]

    def test_astype_str_datetime(self):
        t = XArray(['Aug 23, 2015', '2015 8 24'])
        res = t.astype(datetime.datetime)
        assert res.dtype() is datetime.datetime
        assert res[0] == datetime.datetime(2015, 8, 23)
        assert res[1] == datetime.datetime(2015, 8, 24)


# noinspection PyClassHasNoInit
class TestXArrayClip:
    """
    Tests XArray clip
    """
    # noinspection PyTypeChecker
    def test_clip_int_nan(self):
        nan = float('nan')
        t = XArray([1, 2, 3])
        res = t.clip(nan, nan)
        assert list(res) == [1, 2, 3]

    def test_clip_int_none(self):
        t = XArray([1, 2, 3])
        res = t.clip(None, None)
        assert list(res) == [1, 2, 3]

    def test_clip_int_def(self):
        t = XArray([1, 2, 3])
        res = t.clip()
        assert list(res) == [1, 2, 3]

    # noinspection PyTypeChecker
    def test_clip_float_nan(self):
        nan = float('nan')
        t = XArray([1.0, 2.0, 3.0])
        res = t.clip(nan, nan)
        assert list(res) == [1.0, 2.0, 3.0]

    def test_clip_float_none(self):
        t = XArray([1.0, 2.0, 3.0])
        res = t.clip(None, None)
        assert list(res) == [1.0, 2.0, 3.0]

    def test_clip_float_def(self):
        t = XArray([1.0, 2.0, 3.0])
        res = t.clip()
        assert list(res) == [1.0, 2.0, 3.0]

    def test_clip_int_all(self):
        t = XArray([1, 2, 3])
        res = t.clip(1, 3)
        assert list(res) == [1.0, 2.0, 3.0]

    # noinspection PyTypeChecker
    def test_clip_float_all(self):
        t = XArray([1.0, 2.0, 3.0])
        res = t.clip(1.0, 3.0)
        assert list(res) == [1.0, 2.0, 3.0]

    def test_clip_int_clip(self):
        t = XArray([1, 2, 3])
        res = t.clip(2, 2)
        assert list(res) == [2, 2, 2]

    # noinspection PyTypeChecker
    def test_clip_float_clip(self):
        t = XArray([1.0, 2.0, 3.0])
        res = t.clip(2.0, 2.0)
        assert list(res) == [2.0, 2.0, 2.0]

    # noinspection PyTypeChecker
    def test_clip_list_nan(self):
        nan = float('nan')
        t = XArray([[1, 2, 3]])
        res = t.clip(nan, nan)
        assert list(res) == [[1, 2, 3]]

    def test_clip_list_none(self):
        t = XArray([[1, 2, 3]])
        res = t.clip(None, None)
        assert list(res) == [[1, 2, 3]]

    def test_clip_list_def(self):
        t = XArray([[1, 2, 3]])
        res = t.clip()
        assert list(res) == [[1, 2, 3]]

    def test_clip_list_all(self):
        t = XArray([[1, 2, 3]])
        res = t.clip(1, 3)
        assert list(res) == [[1, 2, 3]]

    def test_clip_list_clip(self):
        t = XArray([[1, 2, 3]])
        res = t.clip(2, 2)
        assert list(res) == [[2, 2, 2]]


# noinspection PyClassHasNoInit
class TestXArrayClipLower:
    """
    Tests XArray clip_lower
    """
    def test_clip_lower_int_all(self):
        t = XArray([1, 2, 3])
        res = t.clip_lower(1)
        assert list(res) == [1, 2, 3]

    def test_clip_int_clip(self):
        t = XArray([1, 2, 3])
        res = t.clip_lower(2)
        assert list(res) == [2, 2, 3]

    def test_clip_lower_list_all(self):
        t = XArray([[1, 2, 3]])
        res = t.clip_lower(1)
        assert list(res) == [[1, 2, 3]]

    def test_clip_lower_list_clip(self):
        t = XArray([[1, 2, 3]])
        res = t.clip_lower(2)
        assert list(res) == [[2, 2, 3]]


# noinspection PyClassHasNoInit
class TestXArrayClipUpper:
    """
    Tests XArray clip_upper
    """
    def test_clip_upper_int_all(self):
        t = XArray([1, 2, 3])
        res = t.clip_upper(3)
        assert list(res) == [1, 2, 3]

    def test_clip_int_clip(self):
        t = XArray([1, 2, 3])
        res = t.clip_upper(2)
        assert list(res) == [1, 2, 2]

    def test_clip_upper_list_all(self):
        t = XArray([[1, 2, 3]])
        res = t.clip_upper(3)
        assert list(res) == [[1, 2, 3]]

    def test_clip_upper_list_clip(self):
        t = XArray([[1, 2, 3]])
        res = t.clip_upper(2)
        assert list(res) == [[1, 2, 2]]


# noinspection PyClassHasNoInit
class TestXArrayTail:
    """
    Tests XArray tail
    """
    def test_tail(self):
        t = XArray(range(1, 100))
        res = t.tail(10)
        assert res == range(90, 100)

    def test_tail_all(self):
        t = XArray(range(1, 100))
        res = t.tail(100)
        assert res == range(1, 100)


# noinspection PyClassHasNoInit
class TestXArrayCountna:
    """
    Tests XArray countna
    """
    def test_countna_not(self):
        t = XArray([1, 2, 3])
        res = t.countna()
        assert res == 0

    def test_countna_none(self):
        t = XArray([1, 2, None])
        res = t.countna()
        assert res == 1

    def test_countna_nan(self):
        t = XArray([1.0, 2.0, float('nan')])
        res = t.countna()
        assert res == 1

    def test_countna_float_none(self):
        t = XArray([1.0, 2.0, None])
        res = t.countna()
        assert res == 1


# noinspection PyClassHasNoInit
class TestXArrayDropna:
    """
    Tests XArray dropna
    """
    def test_dropna_not(self):
        t = XArray([1, 2, 3])
        res = t.dropna()
        assert list(res) == [1, 2, 3]

    def test_dropna_none(self):
        t = XArray([1, 2, None])
        res = t.dropna()
        assert list(res) == [1, 2]

    def test_dropna_nan(self):
        t = XArray([1.0, 2.0, float('nan')])
        res = t.dropna()
        assert list(res) == [1.0, 2.0]

    def test_dropna_float_none(self):
        t = XArray([1.0, 2.0, None])
        res = t.dropna()
        assert list(res) == [1.0, 2.0]


# noinspection PyClassHasNoInit
class TestXArrayFillna:
    """
    Tests XArray fillna
    """
    def test_fillna_not(self):
        t = XArray([1, 2, 3])
        res = t.fillna(10)
        assert list(res) == [1, 2, 3]

    def test_fillna_none(self):
        t = XArray([1, 2, None])
        res = t.fillna(10)
        assert list(res) == [1, 2, 10]

    def test_fillna_none_cast(self):
        t = XArray([1, 2, None])
        res = t.fillna(10.0)
        assert list(res) == [1, 2, 10]

    def test_fillna_nan(self):
        t = XArray([1.0, 2.0, float('nan')])
        res = t.fillna(10.0)
        assert list(res) == [1.0, 2.0, 10.0]

    def test_fillna_float_none(self):
        t = XArray([1.0, 2.0, None])
        res = t.fillna(10.0)
        assert list(res) == [1.0, 2.0, 10.0]

    def test_fillna_nan_cast(self):
        t = XArray([1.0, 2.0, float('nan')])
        res = t.fillna(10)
        assert list(res) == [1.0, 2.0, 10.0]

    def test_fillna_none_float_cast(self):
        t = XArray([1.0, 2.0, None])
        res = t.fillna(10)
        assert list(res) == [1.0, 2.0, 10.0]


# noinspection PyClassHasNoInit
class TestXArrayTopkIndex:
    """
    Tests XArray topk_index
    """
    def test_topk_index_0(self):
        t = XArray([1, 2, 3])
        res = t.topk_index(0)
        assert list(res) == [0, 0, 0]

    def test_topk_index_1(self):
        t = XArray([1, 2, 3])
        res = t.topk_index(1)
        assert list(res) == [0, 0, 1]

    def test_topk_index_2(self):
        t = XArray([1, 2, 3])
        res = t.topk_index(2)
        assert list(res) == [0, 1, 1]

    def test_topk_index_3(self):
        t = XArray([1, 2, 3])
        res = t.topk_index(3)
        assert list(res) == [1, 1, 1]

    def test_topk_index_4(self):
        t = XArray([1, 2, 3])
        res = t.topk_index(4)
        assert list(res) == [1, 1, 1]

    def test_topk_index_float_1(self):
        t = XArray([1.0, 2.0, 3.0])
        res = t.topk_index(1)
        assert list(res) == [0, 0, 1]

    def test_topk_index_str_1(self):
        t = XArray(['a', 'b', 'c'])
        res = t.topk_index(1)
        assert list(res) == [0, 0, 1]

    def test_topk_index_list_1(self):
        t = XArray([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        res = t.topk_index(1)
        assert list(res) == [0, 0, 1]

    def test_topk_index_reverse_int(self):
        t = XArray([1, 2, 3])
        res = t.topk_index(1, reverse=True)
        assert list(res) == [1, 0, 0]

    def test_topk_index_reverse_float(self):
        t = XArray([1.0, 2.0, 3.0])
        res = t.topk_index(1, reverse=True)
        assert list(res) == [1, 0, 0]

    def test_topk_index_reverse_str(self):
        t = XArray(['a', 'b', 'c'])
        res = t.topk_index(1, reverse=True)
        assert list(res) == [1, 0, 0]

    def test_topk_index_reverse_list(self):
        t = XArray([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        res = t.topk_index(1, reverse=True)
        assert list(res) == [1, 0, 0]


# noinspection PyClassHasNoInit
class TestXArraySketchSummary:
    """
    Tests XArray sketch_summary
    """
    def test_sketch_summary_size(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert ss.size() == 5

    def test_sketch_summary_min(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert ss.min() == 1

    def test_sketch_summary_max(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert ss.max() == 5

    def test_sketch_summary_mean(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert ss.mean() == 3.0

    def test_sketch_summary_sum(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert ss.sum() == 15

    def test_sketch_summary_var(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert ss.var() == 2.0

    def test_sketch_summary_std(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert abs(ss.std() - math.sqrt(2.0)) < 0.00001

    def test_sketch_summary_num_undefined(self):
        t = XArray([1, None, 3, None, 5])
        ss = t.sketch_summary()
        assert ss.num_undefined() == 2

    def test_sketch_summary_num_unique(self):
        t = XArray([1, 3, 3, 3, 5])
        ss = t.sketch_summary()
        assert ss.num_unique() == 3

    # TODO files on multiple workers
    # probably something wrong with combiner
    def test_sketch_summary_frequent_items(self):
        t = XArray([1, 3, 3, 3, 5])
        ss = t.sketch_summary()
        assert ss.frequent_items() == {1: 1, 3: 3, 5: 1}

    def test_sketch_summary_frequency_count(self):
        t = XArray([1, 3, 3, 3, 5])
        ss = t.sketch_summary()
        assert ss.frequency_count(1) == 1
        assert ss.frequency_count(3) == 3
        assert ss.frequency_count(5) == 1


# noinspection PyClassHasNoInit
class TestXArrayAppend:
    """
    Tests XArray append
    """
    def test_append(self):
        t = XArray([1, 2, 3])
        u = XArray([10, 20, 30])
        res = t.append(u)
        assert list(res) == [1, 2, 3, 10, 20, 30]

    def test_append_empty_t(self):
        t = XArray([], dtype=int)
        u = XArray([10, 20, 30])
        res = t.append(u)
        assert list(res) == [10, 20, 30]

    def test_append_empty_u(self):
        t = XArray([1, 2, 3])
        u = XArray([], dtype=int)
        res = t.append(u)
        assert list(res) == [1, 2, 3]

    def test_append_int_float_err(self):
        t = XArray([1, 2, 3])
        u = XArray([10., 20., 30.])
        with pytest.raises(RuntimeError):
            t.append(u)

    def test_append_int_str_err(self):
        t = XArray([1, 2, 3])
        u = XArray(['a', 'b', 'c'])
        with pytest.raises(RuntimeError):
            t.append(u)


# noinspection PyClassHasNoInit
class TestXArrayUnique:
    """
    Tests XArray unique
    """
    def test_unique_dict_err(self):
        t = XArray([{'a': 1, 'b': 2, 'c': 3}])
        with pytest.raises(TypeError):
            t.unique()

    def test_unique_int_noop(self):
        t = XArray([1, 2, 3])
        res = t.unique()
        assert len(res) == 3
        assert sorted(list(res)) == [1, 2, 3]

    def test_unique_float_noop(self):
        t = XArray([1.0, 2.0, 3.0])
        res = t.unique()
        assert len(res) == 3
        assert sorted(list(res)) == [1.0, 2.0, 3.0]

    def test_unique_str_noop(self):
        t = XArray(['1', '2', '3'])
        res = t.unique()
        assert len(res) == 3
        assert sorted(list(res)) == ['1', '2', '3']

    def test_unique_int(self):
        t = XArray([1, 2, 3, 1, 2])
        res = t.unique()
        assert len(res) == 3
        assert sorted(list(res)) == [1, 2, 3]

    def test_unique_float(self):
        t = XArray([1.0, 2.0, 3.0, 1.0, 2.0])
        res = t.unique()
        assert len(res) == 3
        assert sorted(list(res)) == [1.0, 2.0, 3.0]

    def test_unique_str(self):
        t = XArray(['1', '2', '3', '1', '2'])
        res = t.unique()
        assert len(res) == 3
        assert sorted(list(res)) == ['1', '2', '3']


# noinspection PyClassHasNoInit
class TestXArrayItemLength:
    """
    Tests XArray item_length
    """
    def test_item_length_int(self):
        t = XArray([1, 2, 3])
        with pytest.raises(TypeError):
            t.item_length()

    def test_item_length_float(self):
        t = XArray([1.0, 2.0, 3.0])
        with pytest.raises(TypeError):
            t.item_length()

    def test_item_length_str(self):
        t = XArray(['a', 'bb', 'ccc'])
        res = t.item_length()
        assert list(res) == [1, 2, 3]
        assert res.dtype() is int

    def test_item_length_list(self):
        t = XArray([[1], [1, 2], [1, 2, 3]])
        res = t.item_length()
        assert list(res) == [1, 2, 3]
        assert res.dtype() is int

    def test_item_length_dict(self):
        t = XArray([{1: 'a'}, {1: 'a', 2: 'b'}, {1: 'a', 2: 'b', 3: '3'}])
        res = t.item_length()
        assert list(res) == [1, 2, 3]
        assert res.dtype() is int


# noinspection PyClassHasNoInit
class TestXArraySplitDatetime:
    """
    Tests XArray split_datetime
    """

    def test_split_datetime_year(self):
        t = XArray([datetime.datetime(2011, 1, 1),
                    datetime.datetime(2012, 2, 2),
                    datetime.datetime(2013, 3, 3)])
        res = t.split_datetime('date', limit='year')
        assert isinstance(res, XFrame)
        assert res.column_names() == ['date.year']
        assert res.column_types() == [int]
        assert len(res) == 3
        assert list(res['date.year']) == [2011, 2012, 2013]

    def test_split_datetime_year_mo(self):
        t = XArray([datetime.datetime(2011, 1, 1),
                    datetime.datetime(2012, 2, 2),
                    datetime.datetime(2013, 3, 3)])
        res = t.split_datetime('date', limit=['year', 'month'])
        assert isinstance(res, XFrame)
        assert res.column_names() == ['date.year', 'date.month']
        assert res.column_types() == [int, int]
        assert len(res) == 3
        assert list(res['date.year']) == [2011, 2012, 2013]
        assert list(res['date.month']) == [1, 2, 3]

    def test_split_datetime_all(self):
        t = XArray([datetime.datetime(2011, 1, 1, 1, 1, 1),
                    datetime.datetime(2012, 2, 2, 2, 2, 2),
                    datetime.datetime(2013, 3, 3, 3, 3, 3)])
        res = t.split_datetime('date')
        assert isinstance(res, XFrame)
        assert res.column_names() == ['date.year', 'date.month', 'date.day',
                                      'date.hour', 'date.minute', 'date.second']
        assert res.column_types() == [int, int, int, int, int, int]
        assert len(res) == 3
        assert list(res['date.year']) == [2011, 2012, 2013]
        assert list(res['date.month']) == [1, 2, 3]
        assert list(res['date.day']) == [1, 2, 3]
        assert list(res['date.hour']) == [1, 2, 3]
        assert list(res['date.minute']) == [1, 2, 3]
        assert list(res['date.second']) == [1, 2, 3]

    def test_split_datetime_year_no_prefix(self):
        t = XArray([datetime.datetime(2011, 1, 1),
                    datetime.datetime(2012, 2, 2),
                    datetime.datetime(2013, 3, 3)])
        res = t.split_datetime(limit='year')
        assert isinstance(res, XFrame)
        assert res.column_names() == ['X.year']
        assert res.column_types() == [int]
        assert len(res) == 3
        assert list(res['X.year']) == [2011, 2012, 2013]

    def test_split_datetime_year_null_prefix(self):
        t = XArray([datetime.datetime(2011, 1, 1),
                    datetime.datetime(2012, 2, 2),
                    datetime.datetime(2013, 3, 3)])
        res = t.split_datetime(column_name_prefix=None, limit='year')
        assert isinstance(res, XFrame)
        assert res.column_names() == ['year']
        assert res.column_types() == [int]
        assert len(res) == 3
        assert list(res['year']) == [2011, 2012, 2013]

    def test_split_datetime_bad_col_type(self):
        t = XArray([1, 2, 3])
        with pytest.raises(TypeError):
            t.split_datetime('date')

    # noinspection PyTypeChecker
    def test_split_datetime_bad_prefix_type(self):
        t = XArray([datetime.datetime(2011, 1, 1),
                    datetime.datetime(2011, 2, 2),
                    datetime.datetime(2011, 3, 3)])
        with pytest.raises(TypeError):
            t.split_datetime(1)

    def test_split_datetime_bad_limit_val(self):
        t = XArray([datetime.datetime(2011, 1, 1),
                    datetime.datetime(2011, 2, 2),
                   datetime. datetime(2011, 3, 3)])
        with pytest.raises(ValueError):
            t.split_datetime('date', limit='xx')

    # noinspection PyTypeChecker
    def test_split_datetime_bad_limit_type(self):
        t = XArray([datetime.datetime(2011, 1, 1),
                    datetime.datetime(2011, 2, 2),
                    datetime.datetime(2011, 3, 3)])
        with pytest.raises(TypeError):
            t.split_datetime('date', limit=1)

    def test_split_datetime_bad_limit_not_list(self):
        t = XArray([datetime.datetime(2011, 1, 1),
                    datetime.datetime(2011, 2, 2),
                    datetime.datetime(2011, 3, 3)])
        with pytest.raises(TypeError):
            t.split_datetime('date', limit=datetime.datetime(2011, 1, 1))


# noinspection PyClassHasNoInit
class TestXArrayUnpackErrors:
    """
    Tests XArray unpack errors
    """
    def test_unpack_str(self):
        t = XArray(['a', 'b', 'c'])
        with pytest.raises(TypeError):
            t.unpack()

    # noinspection PyTypeChecker
    def test_unpack_bad_prefix(self):
        t = XArray([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(TypeError):
            t.unpack(column_name_prefix=1)

    # noinspection PyTypeChecker
    def test_unpack_bad_limit_type(self):
        t = XArray([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(TypeError):
            t.unpack(limit=1)

    def test_unpack_bad_limit_val(self):
        t = XArray([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(TypeError):
            t.unpack(limit=['a', 1])

    def test_unpack_bad_limit_dup(self):
        t = XArray([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            t.unpack(limit=[1, 1])

    # noinspection PyTypeChecker
    def test_unpack_bad_column_types(self):
        t = XArray([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(TypeError):
            t.unpack(column_types=1)

    # noinspection PyTypeChecker
    def test_unpack_bad_column_types_bool(self):
        t = XArray([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(TypeError):
            t.unpack(column_types=[True])

    def test_unpack_column_types_limit_mismatch(self):
        t = XArray([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            t.unpack(limit=[1], column_types=[int, int])

    def test_unpack_dict_column_types_no_limit(self):
        t = XArray([{'a': 1, 'b': 2}, {'c': 3, 'd': 4}])
        with pytest.raises(ValueError):
            t.unpack(column_types=[int, int])

    def test_unpack_empty_no_column_types(self):
        t = XArray([], dtype=list)
        with pytest.raises(RuntimeError):
            t.unpack()

    def test_unpack_empty_list_column_types(self):
        t = XArray([[]], dtype=list)
        with pytest.raises(RuntimeError):
            t.unpack()


# noinspection PyClassHasNoInit
class TestXArrayUnpack:
    """
    Tests XArray unpack list
    """
    def test_unpack_list(self):
        t = XArray([[1, 0, 1],
                    [1, 1, 1],
                    [0, 1]])
        res = t.unpack()
        assert res.column_names() == ['X.0', 'X.1', 'X.2']
        assert res[0] == {'X.0': 1, 'X.1': 0, 'X.2': 1}
        assert res[1] == {'X.0': 1, 'X.1': 1, 'X.2': 1}
        assert res[2] == {'X.0': 0, 'X.1': 1, 'X.2': None}

    def test_unpack_list_limit(self):
        t = XArray([[1, 0, 1],
                    [1, 1, 1],
                    [0, 1]])
        res = t.unpack(limit=[1])
        assert res.column_names() == ['X.1']
        assert res[0] == {'X.1': 0}
        assert res[1] == {'X.1': 1}
        assert res[2] == {'X.1': 1}

    def test_unpack_list_na_values(self):
        t = XArray([[1, 0, 1],
                    [1, 1, 1],
                    [0, 1]])
        res = t.unpack(na_value=0)
        assert res.column_names() == ['X.0', 'X.1', 'X.2']
        assert res[0] == {'X.0': 1, 'X.1': 0, 'X.2': 1}
        assert res[1] == {'X.0': 1, 'X.1': 1, 'X.2': 1}
        assert res[2] == {'X.0': 0, 'X.1': 1, 'X.2': 0}

    def test_unpack_list_na_values_col_types(self):
        t = XArray([[1, 0, 1],
                    [1, 1, 1],
                    [0, 1]])
        res = t.unpack(column_types=[int, int, int], na_value=0)
        assert res.column_names() == ['X.0', 'X.1', 'X.2']
        assert res[0] == {'X.0': 1, 'X.1': 0, 'X.2': 1}
        assert res[1] == {'X.0': 1, 'X.1': 1, 'X.2': 1}
        assert res[2] == {'X.0': 0, 'X.1': 1, 'X.2': 0}

    def test_unpack_list_cast_str(self):
        t = XArray([[1, 0, 1],
                    [1, 1, 1],
                    [0, 1]])
        res = t.unpack(column_types=[str, str, str])
        assert res.column_names() == ['X.0', 'X.1', 'X.2']
        assert res[0] == {'X.0': '1', 'X.1': '0', 'X.2': '1'}
        assert res[1] == {'X.0': '1', 'X.1': '1', 'X.2': '1'}
        assert res[2] == {'X.0': '0', 'X.1': '1', 'X.2': None}

    def test_unpack_list_no_prefix(self):
        t = XArray([[1, 0, 1],
                    [1, 1, 1],
                    [0, 1]])
        res = t.unpack(column_name_prefix='')
        assert res.column_names() == ['0', '1', '2']
        assert res[0] == {'0': 1, '1': 0, '2': 1}
        assert res[1] == {'0': 1, '1': 1, '2': 1}
        assert res[2] == {'0': 0, '1': 1, '2': None}

    def test_unpack_dict_limit(self):
        t = XArray([{'word': 'a', 'count': 1},
                    {'word': 'cat', 'count': 2},
                    {'word': 'is', 'count': 3},
                    {'word': 'coming', 'count': 4}])
        res = t.unpack(limit=['word', 'count'], column_types=[str, int])
        assert res.column_names() == ['X.word', 'X.count']
        assert res[0] == {'X.word': 'a', 'X.count': 1}
        assert res[1] == {'X.word': 'cat', 'X.count': 2}
        assert res[2] == {'X.word': 'is', 'X.count': 3}
        assert res[3] == {'X.word': 'coming', 'X.count': 4}

    def test_unpack_dict_limit_word(self):
        t = XArray([{'word': 'a', 'count': 1},
                    {'word': 'cat', 'count': 2},
                    {'word': 'is', 'count': 3},
                    {'word': 'coming', 'count': 4}])
        res = t.unpack(limit=['word'])
        assert res.column_names() == ['X.word']
        assert res[0] == {'X.word': 'a'}
        assert res[1] == {'X.word': 'cat'}
        assert res[2] == {'X.word': 'is'}
        assert res[3] == {'X.word': 'coming'}

    def test_unpack_dict_limit_count(self):
        t = XArray([{'word': 'a', 'count': 1},
                    {'word': 'cat', 'count': 2},
                    {'word': 'is', 'count': 3},
                    {'word': 'coming', 'count': 4}])
        res = t.unpack(limit=['count'])
        assert res.column_names() == ['X.count']
        assert res[0] == {'X.count': 1}
        assert res[1] == {'X.count': 2}
        assert res[2] == {'X.count': 3}
        assert res[3] == {'X.count': 4}

    def test_unpack_dict_incomplete(self):
        t = XArray([{'word': 'a', 'count': 1},
                    {'word': 'cat', 'count': 2},
                    {'word': 'is'},
                    {'word': 'coming', 'count': 4}])
        res = t.unpack(limit=['word', 'count'], column_types=[str, int])
        assert res.column_names() == ['X.word', 'X.count']
        assert res[0] == {'X.count': 1, 'X.word': 'a'}
        assert res[1] == {'X.count': 2, 'X.word': 'cat'}
        assert res[2] == {'X.count': None, 'X.word': 'is'}
        assert res[3] == {'X.count': 4, 'X.word': 'coming'}

    def test_unpack_dict(self):
        t = XArray([{'word': 'a', 'count': 1},
                    {'word': 'cat', 'count': 2},
                    {'word': 'is', 'count': 3},
                    {'word': 'coming', 'count': 4}])
        res = t.unpack()
        assert res.column_names() == ['X.count', 'X.word']
        assert res[0] == {'X.count': 1, 'X.word': 'a'}
        assert res[1] == {'X.count': 2, 'X.word': 'cat'}
        assert res[2] == {'X.count': 3, 'X.word': 'is'}
        assert res[3] == {'X.count': 4, 'X.word': 'coming'}

    def test_unpack_dict_no_prefix(self):
        t = XArray([{'word': 'a', 'count': 1},
                    {'word': 'cat', 'count': 2},
                    {'word': 'is', 'count': 3},
                    {'word': 'coming', 'count': 4}])
        res = t.unpack(column_name_prefix=None)
        assert res.column_names() == ['count', 'word']
        assert res[0] == {'count': 1, 'word': 'a'}
        assert res[1] == {'count': 2, 'word': 'cat'}
        assert res[2] == {'count': 3, 'word': 'is'}
        assert res[3] == {'count': 4, 'word': 'coming'}


# noinspection PyClassHasNoInit
class TestXArraySort:
    """
    Tests XArray sort
    """
    def test_sort_int(self):
        t = XArray([3, 2, 1])
        res = t.sort()
        assert list(res) == [1, 2, 3]

    def test_sort_float(self):
        t = XArray([3, 2, 1])
        res = t.sort()
        assert list(res) == [1.0, 2.0, 3.0]

    def test_sort_str(self):
        t = XArray(['c', 'b', 'a'])
        res = t.sort()
        assert list(res) == ['a', 'b', 'c']

    def test_sort_list(self):
        t = XArray([[3, 4], [2, 3], [1, 2]])
        with pytest.raises(TypeError):
            t.sort()

    def test_sort_dict(self):
        t = XArray([{'c': 3}, {'b': 2}, {'a': 1}])
        with pytest.raises(TypeError):
            t.sort()

    def test_sort_int_desc(self):
        t = XArray([1, 2, 3])
        res = t.sort(ascending=False)
        assert list(res) == [3, 2, 1]

    def test_sort_float_desc(self):
        t = XArray([1.0, 2.0, 3.0])
        res = t.sort(ascending=False)
        assert list(res) == [3.0, 2.0, 1.0]

    def test_sort_str_desc(self):
        t = XArray(['a', 'b', 'c'])
        res = t.sort(ascending=False)
        assert list(res) == ['c', 'b', 'a']


# noinspection PyClassHasNoInit
class TestXArrayDictTrimByKeys:
    """
    Tests XArray dict_trim_by_keys
    """
    def test_dict_trim_by_keys_bad_type(self):
        t = XArray([3, 2, 1])
        with pytest.raises(TypeError):
            t.dict_trim_by_keys(['a'])

    def test_dict_trim_by_keys_include(self):
        t = XArray([{'a': 0, 'b': 0, 'c': 0}, {'x': 1}])
        res = t.dict_trim_by_keys(['a'], exclude=False)
        assert list(res) == [{'a': 0}, {}]

    def test_dict_trim_by_keys_exclude(self):
        t = XArray([{'a': 0, 'b': 1, 'c': 2}, {'x': 1}])
        res = t.dict_trim_by_keys(['a'])
        assert list(res) == [{'b': 1, 'c': 2}, {'x': 1}]


# noinspection PyClassHasNoInit
class TestXArrayDictTrimByValues:
    """
    Tests XArray dict_trim_by_values
    """
    def test_dict_trim_by_values_bad_type(self):
        t = XArray([3, 2, 1])
        with pytest.raises(TypeError):
            t.dict_trim_by_values(1, 2)

    def test_dict_trim_by_values(self):
        t = XArray([{'a': 0, 'b': 1, 'c': 2, 'd': 3}, {'x': 1}])
        res = t.dict_trim_by_values(1, 2)
        assert list(res) == [{'b': 1, 'c': 2}, {'x': 1}]


# noinspection PyClassHasNoInit
class TestXArrayDictKeys:
    """
    Tests XArray dict_keys
    """
    # noinspection PyArgumentList
    def test_dict_keys_bad_type(self):
        t = XArray([3, 2, 1])
        with pytest.raises(TypeError):
            t.dict_keys(['a'])

    def test_dict_keys_bad_len(self):
        t = XArray([{'a': 0, 'b': 0, 'c': 0}, {'x': 1}])
        with pytest.raises(ValueError):
            t.dict_keys()

    def test_dict_keys(self):
        t = XArray([{'a': 0, 'b': 0, 'c': 0}, {'x': 1, 'y': 2, 'z': 3}])
        res = t.dict_keys()
        assert len(res) == 2
        assert res[0] == {'X.0': 'a', 'X.1': 'c', 'X.2': 'b'}
        assert res[1] == {'X.0': 'y', 'X.1': 'x', 'X.2': 'z'}


# noinspection PyClassHasNoInit
class TestXArrayDictValues:
    """
    Tests XArray dict_values
    """
    # noinspection PyArgumentList
    def test_values_bad_type(self):
        t = XArray([3, 2, 1])
        with pytest.raises(TypeError):
            t.dict_values(['a'])

    def test_values_bad_len(self):
        t = XArray([{'a': 0, 'b': 1, 'c': 2}, {'x': 10}])
        with pytest.raises(ValueError):
            t.dict_values()

    def test_values(self):
        t = XArray([{'a': 0, 'b': 1, 'c': 2}, {'x': 10, 'y': 20, 'z': 30}])
        res = t.dict_values()
        assert len(res) == 2
        assert res[0] == {'X.0': 0, 'X.1': 2, 'X.2': 1}
        assert res[1] == {'X.0': 20, 'X.1': 10, 'X.2': 30}


# noinspection PyClassHasNoInit
class TestXArrayDictHasAnyKeys:
    """
    Tests XArray dict_has_any_keys
    """
    def test_dict_has_any_keys_bad(self):
        t = XArray([3, 2, 1])
        with pytest.raises(TypeError):
            t.dict_has_any_keys(['a'])

    def test_dict_has_any_keys(self):
        t = XArray([{'a': 0, 'b': 0, 'c': 0}, {'x': 1}])
        res = t.dict_has_any_keys(['a'])
        assert list(res) == [True, False]


# noinspection PyClassHasNoInit
class TestXArrayDictHasAllKeys:
    """
    Tests XArray dict_has_all_keys
    """
    def test_dict_has_all_keys_bad(self):
        t = XArray([3, 2, 1])
        with pytest.raises(TypeError):
            t.dict_has_all_keys(['a'])

    def test_dict_has_all_keys(self):
        t = XArray([{'a': 0, 'b': 0, 'c': 0}, {'a': 1, 'b': 1}])
        res = t.dict_has_all_keys(['a', 'b', 'c'])
        assert list(res) == [True, False]

if __name__ == '__main__':
    unittest.main()
