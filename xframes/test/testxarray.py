from __future__ import absolute_import

import pytest
import math
import os
import array
import datetime

# pytest testxarray.py
# pytest testxarray.py::TestXArrayVersion
# pytest testxarray.py::TestXArrayVersion::test_version

from xframes import XArray
from xframes import XFrame
from xframes import object_utils


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
        assert 3 == len(t)
        assert 1 == t[0]
        assert int is t.dtype()

    def test_construct_list_int(self):
        t = XArray([1, 2, 3], dtype=int)
        assert 3 == len(t)
        assert 1 == t[0]
        assert int is t.dtype()

    def test_construct_list_str_infer(self):
        t = XArray(['a', 'b', 'c'])
        assert 3 == len(t)
        assert 'a' == t[0]
        assert str is t.dtype()

    def test_construct_list_str(self):
        t = XArray([1, 2, 3], dtype=str)
        assert 3 == len(t)
        assert '1' == t[0]
        assert str is t.dtype()

    def test_construct_list_float_infer(self):
        t = XArray([1.0, 2.0, 3.0])
        assert 3 == len(t)
        assert 1.0 == t[0]
        assert float is t.dtype()

    def test_construct_list_float(self):
        t = XArray([1, 2, 3], dtype=float)
        assert 3 == len(t)
        assert 1.0 == t[0]
        assert float is t.dtype()

    def test_construct_list_bool_infer(self):
        t = XArray([True, False])
        assert 2 == len(t)
        assert True is t[0]
        assert bool is t.dtype()

    def test_construct_list_bool(self):
        t = XArray([True, False], dtype=bool)
        assert 2 == len(t)
        assert True is t[0]
        assert bool is t.dtype()

    def test_construct_list_list_infer(self):
        t = XArray([[1, 2, 3], [10]])
        assert 2 == len(t)
        assert [1, 2, 3] == t[0]
        assert [10] == t[1]
        assert list is t.dtype()

    def test_construct_list_list(self):
        t = XArray([[1, 2, 3], [10]], dtype=list)
        assert 2 == len(t)
        assert [1, 2, 3] == t[0]
        assert [10] == t[1]
        assert list is t.dtype()

    def test_construct_list_dict_infer(self):
        t = XArray([{'a': 1, 'b': 2}, {'x': 10}])
        assert 2 == len(t)
        assert {'a': 1, 'b': 2} == t[0]
        assert dict is t.dtype()

    def test_construct_list_dict(self):
        t = XArray([{'a': 1, 'b': 2}, {'x': 10}], dtype=dict)
        assert 2 == len(t)
        assert {'a': 1, 'b': 2} == t[0]
        assert dict is t.dtype()

    def test_construct_empty_list_infer(self):
        t = XArray([])
        assert 0 == len(t)
        assert None is t.dtype()
    
    def test_construct_empty_list(self):
        t = XArray([], dtype=int)
        assert 0 == len(t)
        assert int is t.dtype()

    def test_construct_list_int_cast_fail(self):
        with pytest.raises(ValueError):
            t = XArray(['a', 'b', 'c'], dtype=int)
            len(t)     # force materialization

    def test_construct_list_int_cast_ignore(self):
        t = XArray(['1', '2', 'c'], dtype=int, ignore_cast_failure=True)
        assert 3 == len(t)
        assert 1 == t[0]
        assert 2 == t[1]
        assert None is t[2]
        assert int is t.dtype()


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
        assert 100 == len(t)
        assert 100 == t[0]
        assert int is t.dtype()

    def test_construct_start(self):
        t = XArray.from_sequence(100)
        assert 100 == len(t)
        assert 0 == t[0]
        assert int is t.dtype()


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
        assert 4 == len(t)
        assert int is t.dtype()
        assert 1 == t[0]

    def test_construct_local_file_float(self):
        t = XArray('files/test-array-float')
        assert 4 == len(t)
        assert float is t.dtype()
        assert 1.0 == t[0]

    def test_construct_local_file_str(self):
        t = XArray('files/test-array-str')
        assert 4 == len(t)
        assert str is t.dtype()
        assert 'a' == t[0]

    def test_construct_local_file_list(self):
        t = XArray('files/test-array-list')
        assert 4 == len(t)
        assert list is t.dtype()
        assert [1, 2] == t[0]

    def test_construct_local_file_dict(self):
        t = XArray('files/test-array-dict')
        assert 4 == len(t)
        assert dict is t.dtype()
        assert {1: 'a', 2: 'b'} == t[0]

    def test_construct_local_file_datetime(self):
        t = XArray('files/test-array-datetime')
        assert 3 == len(t)
        assert datetime.datetime is t.dtype()
        assert datetime.datetime(2015, 8, 15) == t[0]
        assert datetime.datetime(2016, 9, 16) == t[1]
        assert datetime.datetime(2017, 10, 17) == t[2]

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
        assert 4 == len(t)
        assert ['1', '2', '3', '4'] == list(t)


# noinspection PyClassHasNoInit
class TestXArrayFromConst:
    """
    Tests XArray constructed from const.
    """

    def test_from_const_int(self):
        t = XArray.from_const(1, 10)
        assert 10 == len(t)
        assert 1 == t[0]
        assert int is t.dtype()

    def test_from_const_float(self):
        t = XArray.from_const(1.0, 10)
        assert 10 == len(t)
        assert 1.0 == t[0]
        assert float is t.dtype()

    def test_from_const_str(self):
        t = XArray.from_const('a', 10)
        assert 10 == len(t)
        assert 'a' == t[0]
        assert str is t.dtype()

    def test_from_const_datetime(self):
        t = XArray.from_const(datetime.datetime(2015, 10, 11), 10)
        assert 10 == len(t)
        assert datetime.datetime(2015, 10, 11) == t[0]
        assert datetime.datetime is t.dtype()

    def test_from_const_list(self):
        t = XArray.from_const([1, 2], 10)
        assert 10 == len(t)
        assert [1, 2] == t[0]
        assert list is t.dtype()

    def test_from_const_dict(self):
        t = XArray.from_const({1: 'a'}, 10)
        assert 10 == len(t)
        assert {1: 'a'} == t[0]
        assert dict is t.dtype()

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
            assert '1' == f.readline().strip()
            assert '2' == f.readline().strip()
            assert '3' == f.readline().strip()

    def test_save_format(self, tmpdir):
        t = XArray([1, 2, 3])
        path = os.path.join(str(tmpdir), 'array-csv.csv')
        t.save(path, format='csv')
        with open(path) as f:
            assert '1' == f.readline().strip()
            assert '2' == f.readline().strip()
            assert '3' == f.readline().strip()


# noinspection PyClassHasNoInit
class TestXArrayRepr:
    """
    Tests XArray __repr__ function.
    """
    def test_repr(self):
        t = XArray([1, 2, 3])
        s = t.__repr__()
        assert """dtype: int
Rows: 3
[1, 2, 3]""" == s


# noinspection PyClassHasNoInit
class TestXArrayStr:
    """
    Tests XArray __str__ function.
    """
    def test_str(self):
        # noinspection PyTypeChecker
        t = XArray(range(200))
        s = t.__repr__()
        assert "dtype: int\nRows: 200\n[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11," + \
               " 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25," + \
               " 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41," + \
               " 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57," + \
               " 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73," + \
               " 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90," + \
               " 91, 92, 93, 94, 95, 96, 97, 98, 99, ... ]" == s


# noinspection PyClassHasNoInit
class TestXArrayNonzero:
    """
    Tests XArray __nonzero__ function
    """
    def test_nonzero_nonzero(self):
        t = XArray([0])
        assert True is bool(t)

    def test_nonzero_zero(self):
        t = XArray([])
        assert False is bool(t)


# noinspection PyClassHasNoInit
class TestXArrayLen:
    """
    Tests XArray __len__ function
    """
    def test_len_nonzero(self):
        t = XArray([0])
        assert 1 == len(t)

    def test_len_zero(self):
        t = XArray([])
        assert 0 == len(t)


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
            assert 0 == elem

    def test_iter_3(self):
        t = XArray([0, 1, 2])
        for elem, expect in zip(t, [0, 1, 2]):
            assert expect == elem


# noinspection PyClassHasNoInit
class TestXArrayAddScalar:
    """
    Tests XArray Scalar Addition
    """
    # noinspection PyTypeChecker,PyUnresolvedReferences
    def test_add_scalar(self):
        t = XArray([1, 2, 3])
        assert 3 == len(t)
        assert 1 == t[0]
        assert int is t.dtype()
        t = t + 2
        assert 3 == t[0]
        assert 4 == t[1]
        assert 5 == t[2]


# noinspection PyClassHasNoInit
class TestXArrayAddVector:
    """
    Tests XArray Vector Addition
    """
    def test_add_vector(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([4, 5, 6])
        t = t1 + t2
        assert 3 == len(t)
        assert int is t.dtype()
        assert 5 == t[0]
        assert 7 == t[1]
        assert 9 == t[2]

    def test_add_vector_safe(self):
        t1 = XArray([1, 2, 3])
        t = t1 + t1
        assert 3 == len(t)
        assert int is t.dtype()
        assert 2 == t[0]
        assert 4 == t[1]
        assert 6 == t[2]

        
# noinspection PyClassHasNoInit
class TestXArrayOpScalar:
    """
    Tests XArray Scalar operations other than addition
    """
    # noinspection PyTypeChecker,PyUnresolvedReferences
    def test_sub_scalar(self):
        t = XArray([1, 2, 3])
        res = t - 1
        assert 0 == res[0]
        assert 1 == res[1]
        assert 2 == res[2]

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def test_mul_scalar(self):
        t = XArray([1, 2, 3])
        res = t * 2
        assert 2 == res[0]
        assert 4 == res[1]
        assert 6 == res[2]

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def test_floordiv_scalar(self):
        t = XArray([1, 2, 3])
        res = t // 2
        assert 0 == res[0]
        assert 1 == res[1]
        assert 1 == res[2]

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def test_truediv_scalar(self):
        t = XArray([1, 2, 3])
        res = t / 2
        assert 0.5 == res[0]
        assert 1.0 == res[1]
        assert 1.5 == res[2]

    # noinspection PyTypeChecker
    def test_pow_scalar(self):
        t = XArray([1, 2, 3])
        res = t ** 2
        assert 1 == res[0]
        assert 4 == res[1]
        assert 9 == res[2]

    # noinspection PyUnresolvedReferences
    def test_lt_scalar(self):
        t = XArray([1, 2, 3])
        res = t < 3
        assert True is res[0]
        assert True is res[1]
        assert False is res[2]

    # noinspection PyUnresolvedReferences
    def test_le_scalar(self):
        t = XArray([1, 2, 3])
        res = t <= 2
        assert True is res[0]
        assert True is res[1]
        assert False is res[2]

    # noinspection PyUnresolvedReferences
    def test_gt_scalar(self):
        t = XArray([1, 2, 3])
        res = t > 2
        assert False is res[0]
        assert False is res[1]
        assert True is res[2]

    # noinspection PyUnresolvedReferences
    def test_ge_scalar(self):
        t = XArray([1, 2, 3])
        res = t >= 3
        assert False is res[0]
        assert False is res[1]
        assert True is res[2]

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def test_radd_scalar(self):
        t = XArray([1, 2, 3])
        res = 1 + t
        assert 2 == res[0]
        assert 3 == res[1]
        assert 4 == res[2]

    # noinspection PyUnresolvedReferences
    def test_rsub_scalar(self):
        t = XArray([1, 2, 3])
        res = 1 - t
        assert 0 == res[0]
        assert -1 == res[1]
        assert -2 == res[2]

    # noinspection PyUnresolvedReferences
    def test_rmul_scalar(self):
        t = XArray([1, 2, 3])
        res = 2 * t
        assert 2 == res[0]
        assert 4 == res[1]
        assert 6 == res[2]

    # noinspection PyUnresolvedReferences
    def test_rdiv_scalar(self):
        t = XArray([1, 2, 3])
        res = 12 / t
        assert 12 == res[0]
        assert 6 == res[1]
        assert 4 == res[2]

    # noinspection PyUnresolvedReferences
    def test_eq_scalar(self):
        t = XArray([1, 2, 3])
        res = t == 2
        assert False is res[0]
        assert True is res[1]
        assert False is res[2]

    # noinspection PyUnresolvedReferences
    def test_ne_scalar(self):
        t = XArray([1, 2, 3])
        res = t != 2
        assert True is res[0]
        assert False is res[1]
        assert True is res[2]

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
        assert 3 == t[0]
        assert 3 == t[1]
        assert 3 == t[2]

    def test_mul_vector(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([4, 5, 6])
        res = t1 * t2
        assert 4 == res[0]
        assert 10 == res[1]
        assert 18 == res[2]

    def test_div_vector(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([4, 6, 12])
        res = t2 / t1
        assert 4 == res[0]
        assert 3 == res[1]
        assert 4 == res[2]

    def test_lt_vector(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([4, 2, 2])
        res = t1 < t2
        assert True is res[0]
        assert False is res[1]
        assert False is res[2]

    def test_le_vector(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([4, 2, 2])
        res = t1 <= t2
        assert True is res[0]
        assert True is res[1]
        assert False is res[2]

    def test_gt_vector(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([4, 2, 2])
        res = t1 > t2
        assert False is res[0]
        assert False is res[1]
        assert True is res[2]

    def test_ge_vector(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([4, 2, 2])
        res = t1 >= t2
        assert False is res[0]
        assert True is res[1]
        assert True is res[2]

    def test_eq_vector(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([4, 2, 2])
        res = t1 == t2
        assert False is res[0]
        assert True is res[1]
        assert False is res[2]

    def test_ne_vector(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([4, 2, 2])
        res = t1 != t2
        assert True is res[0]
        assert False is res[1]
        assert True is res[2]

    def test_and_vector(self):
        t1 = XArray([False, False, True, True])
        t2 = XArray([False, True, False, True])
        res = t1 & t2
        assert False is res[0]
        assert False is res[1]
        assert False is res[2]
        assert True is res[3]

    def test_or_vector(self):
        t1 = XArray([False, False, True, True])
        t2 = XArray([False, True, False, True])
        res = t1 | t2
        assert False is res[0]
        assert True is res[1]
        assert True is res[2]
        assert True is res[3]


# noinspection PyClassHasNoInit
class TestXArrayOpUnary:
    """
    Tests XArray Unary operations
    """
    def test_neg_unary(self):
        t = XArray([1, -2, 3])
        res = -t
        assert -1 == res[0]
        assert 2 == res[1]
        assert -3 == res[2]

    def test_pos_unary(self):
        t = XArray([1, -2, 3])
        res = +t
        assert 1 == res[0]
        assert -2 == res[1]
        assert 3 == res[2]

    # noinspection PyTypeChecker
    def test_abs_unary(self):
        t = XArray([1, -2, 3])
        res = abs(t)
        assert 1 == res[0]
        assert 2 == res[1]
        assert 3 == res[2]


# noinspection PyClassHasNoInit
class TestXArrayLogicalFilter:
    """
    Tests XArray logical filter (XArray indexed by XArray)
    """
    def test_logical_filter_array(self):
        t1 = XArray([1, 2, 3])
        t2 = XArray([1, 0, 1])
        res = t1[t2]
        assert 2 == len(res)
        assert 1 == res[0]
        assert 3 == res[1]

    def test_logical_filter_test(self):
        t1 = XArray([1, 2, 3])
        res = t1[t1 != 2]
        assert 2 == len(res)
        assert 1 == res[0]
        assert 3 == res[1]

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
        assert 1 == t[0]

    def test_copy_range_neg(self):
        t = XArray([1, 2, 3])
        assert 3 == t[-1]

    def test_copy_range_index_err(self):
        t = XArray([1, 2, 3])
        with pytest.raises(IndexError):
            _ = t[3]
        
    def test_copy_range_slice(self):
        t = XArray([1, 2, 3])
        res = t[0:2]
        assert 2 == len(res)
        assert 1 == res[0]
        assert 2 == res[1]

    def test_copy_range_slice_neg_start(self):
        t = XArray([1, 2, 3, 4, 5])
        res = t[-3:4]
        assert 2 == len(res)
        assert 3 == res[0]
        assert 4 == res[1]

    def test_copy_range_slice_neg_stop(self):
        t = XArray([1, 2, 3, 4, 5])
        res = t[1:-2]
        assert 2 == len(res)
        assert 2 == res[0]
        assert 3 == res[1]

    def test_copy_range_slice_stride(self):
        t = XArray([1, 2, 3, 4, 5])
        res = t[0:4:2]
        assert 2 == len(res)
        assert 1 == res[0]
        assert 3 == res[1]

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
        assert 3 == t.size()


# noinspection PyClassHasNoInit
class TestXArrayDtype:
    """
    Tests XArray dtype operation
    """
    def test_dtype(self):
        t = XArray([1, 2, 3])
        assert int is t.dtype()


# noinspection PyClassHasNoInit
class TestXArrayHead:
    """
    Tests XArray head operation
    """
    def test_head(self):
        t = XArray([1, 2, 3])
        assert 3 == len(t.head())

    def test_head_10(self):
        # noinspection PyTypeChecker
        t = XArray(range(100))
        assert 10 == len(t.head())

    def test_head_5(self):
        # noinspection PyTypeChecker
        t = XArray(range(100))
        assert 5 == len(t.head(5))


# noinspection PyClassHasNoInit
class TestXArrayVectorSlice:
    """
    Tests XArray vector_slice operation
    """
    def test_vector_slice_start_0(self):
        t = XArray([[1, 2, 3], [10, 11, 12]])
        res = t.vector_slice(0)
        assert 2 == len(res)
        assert 1 == res[0]
        assert 10 == res[1]

    def test_vector_slice_start_1(self):
        t = XArray([[1, 2, 3], [10, 11, 12]])
        res = t.vector_slice(1)
        assert 2 == len(res)
        assert 2 == res[0]
        assert 11 == res[1]

    def test_vector_slice_start_end(self):
        t = XArray([[1, 2, 3], [10, 11, 12]])
        res = t.vector_slice(0, 2)
        assert 2 == len(res)
        assert [1, 2] == res[0]
        assert [10, 11] == res[1]

    def test_vector_slice_start_none(self):
        t = XArray([[1], [1, 2], [1, 2, 3]])
        res = t.vector_slice(2)
        assert 3 == len(res)
        assert None is res[0]
        assert None is res[1]
        assert 3 == res[2]

    def test_vector_slice_start_end_none(self):
        t = XArray([[1], [1, 2], [1, 2, 3]])
        res = t.vector_slice(0, 2)
        assert 3 == len(res)
        assert None is res[0]
        assert [1, 2] == res[1]
        assert [1, 2] == res[2]


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
        assert 3 == len(res)
        assert int is res.dtype()
        assert 2 == res[0]
        assert 4 == res[1]
        assert 6 == res[2]

    def test_apply_float_cast(self):
        t = XArray([1, 2, 3])
        res = t.apply(lambda x: x * 2, float)
        assert 3 == len(res)
        assert float is res.dtype()
        assert 2.0 == res[0]
        assert 4.0 == res[1]
        assert 6.0 == res[2]

    def test_apply_skip_undefined(self):
        t = XArray([1, 2, 3, None])
        res = t.apply(lambda x: x * 2)
        assert 4 == len(res)
        assert int is res.dtype()
        assert 2 == res[0]
        assert 4 == res[1]
        assert 6 == res[2]
        assert None is res[3]

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
        assert 6 == len(res)
        assert int is res.dtype()
        assert 1 == res[0]
        assert 1 == res[1]
        assert 2 == res[2]
        assert 1 == res[3]
        assert 2 == res[4]
        assert 3 == res[5]

    def test_flat_map_int(self):
        t = XArray([[1], [1, 2], [1, 2, 3]])
        res = t.flat_map(lambda x: [v * 2 for v in x])
        assert 6 == len(res)
        assert int is res.dtype()
        assert 2 == res[0]
        assert 2 == res[1]
        assert 4 == res[2]
        assert 2 == res[3]
        assert 4 == res[4]
        assert 6 == res[5]

    def test_flat_map_str(self):
        t = XArray([['a'], ['a', 'b'], ['a', 'b', 'c']])
        res = t.flat_map(lambda x: x)
        assert 6 == len(res)
        assert str is res.dtype()
        assert 'a' == res[0]
        assert 'a' == res[1]
        assert 'b' == res[2]
        assert 'a' == res[3]
        assert 'b' == res[4]
        assert 'c' == res[5]

    def test_flat_map_float_cast(self):
        t = XArray([[1], [1, 2], [1, 2, 3]])
        res = t.flat_map(lambda x: x, dtype=float)
        assert 6 == len(res)
        assert float is res.dtype()
        assert 1.0 == res[0]
        assert 1.0 == res[1]
        assert 2.0 == res[2]
        assert 1.0 == res[3]
        assert 2.0 == res[4]
        assert 3.0 == res[5]

    def test_flat_map_skip_undefined(self):
        t = XArray([[1], [1, 2], [1, 2, 3], None, [None]])
        res = t.flat_map(lambda x: x)
        assert 6 == len(res)
        assert int is res.dtype()
        assert 1 == res[0]
        assert 1 == res[1]
        assert 2 == res[2]
        assert 1 == res[3]
        assert 2 == res[4]
        assert 3 == res[5]

    def test_flat_map_no_fun(self):
        t = XArray([[1], [1, 2], [1, 2, 3]])
        res = t.flat_map()
        assert 6 == len(res)
        assert int is res.dtype()
        assert 1 == res[0]
        assert 1 == res[1]
        assert 2 == res[2]
        assert 1 == res[3]
        assert 2 == res[4]
        assert 3 == res[5]

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
        assert 1 == len(res)

    def test_filter_empty(self):
        t = XArray([1, 2, 3])
        res = t.filter(lambda x: x == 10)
        assert 0 == len(res)


# noinspection PyClassHasNoInit
class TestXArraySample:
    """
    Tests XArray sample
    """
    def test_sample_no_seed(self):
        # noinspection PyTypeChecker
        t = XArray(range(10))
        res = t.sample(0.3)
        assert 10 > len(res)

    @pytest.mark.skip(reason='depends on number of partitions')
    def test_sample_seed(self):
        # noinspection PyTypeChecker
        t = XArray(range(10))
        res = t.sample(0.3, seed=1)
        # get 3, 6, 9 with this seed
        assert 3 == len(res)
        assert 3 == res[0]
        assert 6 == res[1]

    def test_sample_zero(self):
        # noinspection PyTypeChecker
        t = XArray(range(10))
        res = t.sample(0.0)
        assert 0 == len(res)

    def test_sample_err_gt(self):
        # noinspection PyTypeChecker
        t = XArray(range(10))
        with pytest.raises(ValueError):
            t.sample(2, seed=1)

    def test_sample_err_lt(self):
        # noinspection PyTypeChecker
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
        assert False is t.all()

    def test_all_int_zero(self):
        t = XArray([1, 0])
        assert False is t.all()

    def test_all_int_true(self):
        t = XArray([1, 2])
        assert True is t.all()

    # float
    def test_all_float_nan(self):
        t = XArray([1.0, float('nan')])
        assert False is t.all()

    def test_all_float_none(self):
        t = XArray([1.0, None])
        assert False is t.all()

    def test_all_float_zero(self):
        t = XArray([1.0, 0.0])
        assert False is t.all()

    def test_all_float_true(self):
        t = XArray([1.0, 2.0])
        assert True is t.all()

    # str
    def test_all_str_empty(self):
        t = XArray(['hello', ''])
        assert False is t.all()

    def test_all_str_none(self):
        t = XArray(['hello', None])
        assert False is t.all()

    def test_all_str_true(self):
        t = XArray(['hello', 'world'])
        assert True is t.all()

    # list
    def test_all_list_empty(self):
        t = XArray([[1, 2], []])
        assert False is t.all()

    def test_all_list_none(self):
        t = XArray([[1, 2], None])
        assert False is t.all()

    def test_all_list_true(self):
        t = XArray([[1, 2], [2, 3]])
        assert True is t.all()

    # dict
    def test_all_dict_empty(self):
        t = XArray([{1: 'a'}, {}])
        assert False is t.all()

    def test_all_dict_none(self):
        t = XArray([{1: 'a'}, None])
        assert False is t.all()

    def test_all_dict_true(self):
        t = XArray([{1: 'a'}, {2: 'b'}])
        assert True is t.all()

    # empty
    def test_all_empty(self):
        t = XArray([])
        assert True is t.all()


# noinspection PyClassHasNoInit
class TestXArrayAny:
    """
    Tests XArray any
    """
    # int
    def test_any_int(self):
        t = XArray([1, 2])
        assert True is t.any()

    def test_any_int_true(self):
        t = XArray([0, 1])
        assert True is t.any()

    def test_any_int_false(self):
        t = XArray([0, 0])
        assert False is t.any()

    def test_any_int_missing_true(self):
        t = XArray([1, None])
        assert True is t.any()

    def test_any_int_missing_false(self):
        t = XArray([None, 0])
        assert False is t.any()

    # float
    def test_any_float(self):
        t = XArray([1., 2.])
        assert True is t.any()

    def test_any_float_true(self):
        t = XArray([0.0, 1.0])
        assert True is t.any()

    def test_any_float_false(self):
        t = XArray([0.0, 0.0])
        assert False is t.any()

    def test_any_float_missing_true(self):
        t = XArray([1.0, None])
        assert True is t.any()

    def test_any_float_missing_true_nan(self):
        t = XArray([1.0, float('nan')])
        assert True is t.any()

    def test_any_float_missing_true_none(self):
        t = XArray([1.0, None])
        assert True is t.any()

    def test_any_float_missing_false(self):
        t = XArray([None, 0.0])
        assert False is t.any()

    def test_any_float_missing_false_nan(self):
        t = XArray([float('nan'), 0.0])
        assert False is t.any()

    def test_any_float_missing_false_none(self):
        t = XArray([None, 0.0])
        assert False is t.any()

    # str
    def test_any_str(self):
        t = XArray(['a', 'b'])
        assert True is t.any()

    def test_any_str_true(self):
        t = XArray(['', 'a'])
        assert True is t.any()

    def test_any_str_false(self):
        t = XArray(['', ''])
        assert False is t.any()

    def test_any_str_missing_true(self):
        t = XArray(['a', None])
        assert True is t.any()

    def test_any_str_missing_false(self):
        t = XArray([None, ''])
        assert False is t.any()

    # list
    def test_any_list(self):
        t = XArray([[1], [2]])
        assert True is t.any()

    def test_any_list_true(self):
        t = XArray([[], ['a']])
        assert True is t.any()

    def test_any_list_false(self):
        t = XArray([[], []])
        assert False is t.any()

    def test_any_list_missing_true(self):
        t = XArray([['a'], None])
        assert True is t.any()

    def test_any_list_missing_false(self):
        t = XArray([None, []])
        assert False is t.any()

    # dict
    def test_any_dict(self):
        t = XArray([{'a': 1, 'b': 2}])
        assert True is t.any()

    def test_any_dict_true(self):
        t = XArray([{}, {'a': 1}])
        assert True is t.any()

    def test_any_dict_false(self):
        t = XArray([{}, {}])
        assert False is t.any()

    def test_any_dict_missing_true(self):
        t = XArray([{'a': 1}, None])
        assert True is t.any()

    def test_any_dict_missing_false(self):
        t = XArray([None, {}])
        assert False is t.any()

    # empty
    def test_any_empty(self):
        t = XArray([])
        assert False is t.any()


# noinspection PyClassHasNoInit
class TestXArrayMax:
    """
    Tests XArray max
    """
    def test_max_empty(self):
        t = XArray([])
        assert None is t.max()

    def test_max_err(self):
        t = XArray(['a'])
        with pytest.raises(TypeError):
            t.max()

    def test_max_int(self):
        t = XArray([1, 2, 3])
        assert 3 == t.max()

    def test_max_float(self):
        t = XArray([1.0, 2.0, 3.0])
        assert 3.0 == t.max()


# noinspection PyClassHasNoInit
class TestXArrayMin:
    """
    Tests XArray min
    """
    def test_min_empty(self):
        t = XArray([])
        assert None is t.min()

    def test_min_err(self):
        t = XArray(['a'])
        with pytest.raises(TypeError):
            t.min()

    def test_min_int(self):
        t = XArray([1, 2, 3])
        assert 1 == t.min()

    def test_min_float(self):
        t = XArray([1.0, 2.0, 3.0])
        assert 1.0 == t.min()


# noinspection PyClassHasNoInit
class TestXArraySum:
    """
    Tests XArray sum
    """
    def test_sum_empty(self):
        t = XArray([])
        assert None is t.sum()

    def test_sum_err(self):
        t = XArray(['a'])
        with pytest.raises(TypeError):
            t.sum()

    def test_sum_int(self):
        t = XArray([1, 2, 3])
        assert 6 == t.sum()

    def test_sum_float(self):
        t = XArray([1.0, 2.0, 3.0])
        assert 6.0 == t.sum()

    def test_sum_array(self):
        t = XArray([array.array('l', [10, 20, 30]), array.array('l', [40, 50, 60])])
        assert t.sum() == array.array('l', [50, 70, 90])

    def test_sum_list(self):
        t = XArray([[10, 20, 30], [40, 50, 60]])
        assert [50, 70, 90] == t.sum()

    def test_sum_dict(self):
        t = XArray([{'x': 1, 'y': 2}, {'x': 3, 'y': 4}])
        assert {'x': 4, 'y': 6} == t.sum()


# noinspection PyClassHasNoInit
class TestXArrayMean:
    """
    Tests XArray mean
    """
    def test_mean_empty(self):
        t = XArray([])
        assert None is t.mean()

    def test_mean_err(self):
        t = XArray(['a'])
        with pytest.raises(TypeError):
            t.mean()

    def test_mean_int(self):
        t = XArray([1, 2, 3])
        assert 2 == t.mean()

    def test_mean_float(self):
        t = XArray([1.0, 2.0, 3.0])
        assert 2.0 == t.mean()


# noinspection PyClassHasNoInit
class TestXArrayStd:
    """
    Tests XArray std
    """
    def test_std_empty(self):
        t = XArray([])
        assert None is t.std()

    def test_std_err(self):
        t = XArray(['a'])
        with pytest.raises(TypeError):
            t.std()

    def test_std_int(self):
        t = XArray([1, 2, 3])
        expect = math.sqrt(2.0 / 3.0)
        assert expect == t.std()

    def test_std_float(self):
        t = XArray([1.0, 2.0, 3.0])
        expect = math.sqrt(2.0 / 3.0)
        assert expect == t.std()


# noinspection PyClassHasNoInit
class TestXArrayVar:
    """
    Tests XArray var
    """
    def test_var_empty(self):
        t = XArray([])
        assert None is t.var()

    def test_var_err(self):
        t = XArray(['a'])
        with pytest.raises(TypeError):
            t.var()

    def test_var_int(self):
        t = XArray([1, 2, 3])
        expect = 2.0 / 3.0
        assert expect == t.var()

    def test_var_float(self):
        t = XArray([1.0, 2.0, 3.0])
        expect = 2.0 / 3.0
        assert expect == t.var()


# noinspection PyClassHasNoInit
class TestXArrayNumMissing:
    """
    Tests XArray num_missing
    """
    def test_num_missing_empty(self):
        t = XArray([])
        assert 0 == t.num_missing()

    def test_num_missing_zero(self):
        t = XArray([1, 2, 3])
        assert 0 == t.num_missing()

    def test_num_missing_int_none(self):
        t = XArray([1, 2, None])
        assert 1 == t.num_missing()

    def test_num_missing_int_all(self):
        t = XArray([None, None, None], dtype=int)
        assert 3 == t.num_missing()

    def test_num_missing_float_none(self):
        t = XArray([1.0, 2.0, None])
        assert 1 == t.num_missing()

    def test_num_missing_float_nan(self):
        t = XArray([1.0, 2.0, float('nan')])
        assert 1 == t.num_missing()


# noinspection PyClassHasNoInit
class TestXArrayNumNonzero:
    """
    Tests XArray nnz
    """
    def test_nnz_empty(self):
        t = XArray([])
        assert 0 == t.nnz()

    def test_nnz_zero_int(self):
        t = XArray([0, 0, 0])
        assert 0 == t.nnz()

    def test_nnz_zero_float(self):
        t = XArray([0.0, 0.0, 0.0])
        assert 0 == t.nnz()

    def test_nnz_int_none(self):
        t = XArray([1, 2, None])
        assert 2 == t.nnz()

    def test_nnz_int_all(self):
        t = XArray([None, None, None], dtype=int)
        assert 0 == t.nnz()

    def test_nnz_float_none(self):
        t = XArray([1.0, 2.0, None])
        assert 2 == t.nnz()

    def test_nnz_float_nan(self):
        t = XArray([1.0, 2.0, float('nan')])
        assert 2 == t.nnz()


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
        assert str is res.dtype()
        assert '2015 08 21' == res[0]
        assert '2016 09 22' == res[1]
        assert '2017 10 23' == res[2]

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
        assert datetime.datetime is res.dtype()
        assert datetime.datetime(2015, 8, 21) == res[0]
        assert datetime.datetime(2015, 8, 22) == res[1]
        assert datetime.datetime(2015, 8, 23) == res[2]

    def test_str_to_datetime_parse(self):
        t = XArray(['2015 8 21', '2015 Aug 22', '23 Aug 2015', 'Aug 24 2015'])
        res = t.str_to_datetime()
        assert datetime.datetime is res.dtype()
        assert datetime.datetime(2015, 8, 21) == res[0]
        assert datetime.datetime(2015, 8, 22) == res[1]
        assert datetime.datetime(2015, 8, 23) == res[2]
        assert datetime.datetime(2015, 8, 24) == res[3]

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
        assert int is res.dtype()

    def test_astype_int_int(self):
        t = XArray([1, 2, 3])
        res = t.astype(int)
        assert int is res.dtype()
        assert 1 == res[0]

    def test_astype_int_float(self):
        t = XArray([1, 2, 3])
        res = t.astype(float)
        assert float is res.dtype()
        assert 1.0 == res[0]

    def test_astype_float_float(self):
        t = XArray([1.0, 2.0, 3.0])
        res = t.astype(float)
        assert float is res.dtype()
        assert 1.0 == res[0]

    def test_astype_float_int(self):
        t = XArray([1.0, 2.0, 3.0])
        res = t.astype(int)
        assert int is res.dtype()
        assert 1 == res[0]

    def test_astype_int_str(self):
        t = XArray([1, 2, 3])
        res = t.astype(str)
        assert str is res.dtype()
        assert '1' == res[0]

    def test_astype_str_list(self):
        t = XArray(['[1, 2, 3]', '[4, 5, 6]'])
        res = t.astype(list)
        assert list is res.dtype()
        assert [1, 2, 3] == res[0]

    def test_astype_str_dict(self):
        t = XArray(['{"a": 1, "b": 2}', '{"x": 3}'])
        res = t.astype(dict)
        assert dict is res.dtype()
        assert {'a': 1, 'b': 2} == res[0]

    # noinspection PyTypeChecker
    def test_astype_str_array(self):
        t = XArray(['[1, 2, 3]', '[4, 5, 6]'])
        res = t.astype(array)
        assert array is res.dtype()
        assert [1, 2, 3] == list(res[0])

    def test_astype_str_datetime(self):
        t = XArray(['Aug 23, 2015', '2015 8 24'])
        res = t.astype(datetime.datetime)
        assert datetime.datetime is res.dtype()
        assert datetime.datetime(2015, 8, 23) == res[0]
        assert datetime.datetime(2015, 8, 24) == res[1]


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
        assert [1, 2, 3] == list(res)

    def test_clip_int_none(self):
        t = XArray([1, 2, 3])
        res = t.clip(None, None)
        assert [1, 2, 3] == list(res)

    def test_clip_int_def(self):
        t = XArray([1, 2, 3])
        res = t.clip()
        assert [1, 2, 3] == list(res)

    # noinspection PyTypeChecker
    def test_clip_float_nan(self):
        nan = float('nan')
        t = XArray([1.0, 2.0, 3.0])
        res = t.clip(nan, nan)
        assert [1.0, 2.0, 3.0] == list(res)

    def test_clip_float_none(self):
        t = XArray([1.0, 2.0, 3.0])
        res = t.clip(None, None)
        assert [1.0, 2.0, 3.0] == list(res)

    def test_clip_float_def(self):
        t = XArray([1.0, 2.0, 3.0])
        res = t.clip()
        assert [1.0, 2.0, 3.0] == list(res)

    def test_clip_int_all(self):
        t = XArray([1, 2, 3])
        res = t.clip(1, 3)
        assert [1.0, 2.0, 3.0] == list(res)

    # noinspection PyTypeChecker
    def test_clip_float_all(self):
        t = XArray([1.0, 2.0, 3.0])
        res = t.clip(1.0, 3.0)
        assert [1.0, 2.0, 3.0] == list(res)

    def test_clip_int_clip(self):
        t = XArray([1, 2, 3])
        res = t.clip(2, 2)
        assert [2, 2, 2] == list(res)

    # noinspection PyTypeChecker
    def test_clip_float_clip(self):
        t = XArray([1.0, 2.0, 3.0])
        res = t.clip(2.0, 2.0)
        assert [2.0, 2.0, 2.0] == list(res)

    # noinspection PyTypeChecker
    def test_clip_list_nan(self):
        nan = float('nan')
        t = XArray([[1, 2, 3]])
        res = t.clip(nan, nan)
        assert [[1, 2, 3]] == list(res)

    def test_clip_list_none(self):
        t = XArray([[1, 2, 3]])
        res = t.clip(None, None)
        assert [[1, 2, 3]] == list(res)

    def test_clip_list_def(self):
        t = XArray([[1, 2, 3]])
        res = t.clip()
        assert [[1, 2, 3]] == list(res)

    def test_clip_list_all(self):
        t = XArray([[1, 2, 3]])
        res = t.clip(1, 3)
        assert [[1, 2, 3]] == list(res)

    def test_clip_list_clip(self):
        t = XArray([[1, 2, 3]])
        res = t.clip(2, 2)
        assert [[2, 2, 2]] == list(res)


# noinspection PyClassHasNoInit
class TestXArrayClipLower:
    """
    Tests XArray clip_lower
    """
    def test_clip_lower_int_all(self):
        t = XArray([1, 2, 3])
        res = t.clip_lower(1)
        assert [1, 2, 3] == list(res)

    def test_clip_int_clip(self):
        t = XArray([1, 2, 3])
        res = t.clip_lower(2)
        assert [2, 2, 3] == list(res)

    def test_clip_lower_list_all(self):
        t = XArray([[1, 2, 3]])
        res = t.clip_lower(1)
        assert [[1, 2, 3]] == list(res)

    def test_clip_lower_list_clip(self):
        t = XArray([[1, 2, 3]])
        res = t.clip_lower(2)
        assert [[2, 2, 3]] == list(res)


# noinspection PyClassHasNoInit
class TestXArrayClipUpper:
    """
    Tests XArray clip_upper
    """
    def test_clip_upper_int_all(self):
        t = XArray([1, 2, 3])
        res = t.clip_upper(3)
        assert [1, 2, 3] == list(res)

    def test_clip_int_clip(self):
        t = XArray([1, 2, 3])
        res = t.clip_upper(2)
        assert [1, 2, 2] == list(res)

    def test_clip_upper_list_all(self):
        t = XArray([[1, 2, 3]])
        res = t.clip_upper(3)
        assert [[1, 2, 3]] == list(res)

    def test_clip_upper_list_clip(self):
        t = XArray([[1, 2, 3]])
        res = t.clip_upper(2)
        assert [[1, 2, 2]] == list(res)


# noinspection PyClassHasNoInit
class TestXArrayTail:
    """
    Tests XArray tail
    """
    def test_tail(self):
        # noinspection PyTypeChecker
        t = XArray(range(1, 100))
        res = t.tail(10)
        assert range(90, 100) == res

    def test_tail_all(self):
        # noinspection PyTypeChecker
        t = XArray(range(1, 100))
        res = t.tail(100)
        assert range(1, 100) == res


# noinspection PyClassHasNoInit
class TestXArrayCountna:
    """
    Tests XArray countna
    """
    def test_countna_not(self):
        t = XArray([1, 2, 3])
        res = t.countna()
        assert 0 == res

    def test_countna_none(self):
        t = XArray([1, 2, None])
        res = t.countna()
        assert 1 == res

    def test_countna_nan(self):
        t = XArray([1.0, 2.0, float('nan')])
        res = t.countna()
        assert 1 == res

    def test_countna_float_none(self):
        t = XArray([1.0, 2.0, None])
        res = t.countna()
        assert 1 == res


# noinspection PyClassHasNoInit
class TestXArrayDropna:
    """
    Tests XArray dropna
    """
    def test_dropna_not(self):
        t = XArray([1, 2, 3])
        res = t.dropna()
        assert [1, 2, 3] == list(res)

    def test_dropna_none(self):
        t = XArray([1, 2, None])
        res = t.dropna()
        assert [1, 2] == list(res)

    def test_dropna_nan(self):
        t = XArray([1.0, 2.0, float('nan')])
        res = t.dropna()
        assert [1.0, 2.0] == list(res)

    def test_dropna_float_none(self):
        t = XArray([1.0, 2.0, None])
        res = t.dropna()
        assert [1.0, 2.0] == list(res)


# noinspection PyClassHasNoInit
class TestXArrayFillna:
    """
    Tests XArray fillna
    """
    def test_fillna_not(self):
        t = XArray([1, 2, 3])
        res = t.fillna(10)
        assert [1, 2, 3] == list(res)

    def test_fillna_none(self):
        t = XArray([1, 2, None])
        res = t.fillna(10)
        assert [1, 2, 10] == list(res)

    def test_fillna_none_cast(self):
        t = XArray([1, 2, None])
        res = t.fillna(10.0)
        assert [1, 2, 10] == list(res)

    def test_fillna_nan(self):
        t = XArray([1.0, 2.0, float('nan')])
        res = t.fillna(10.0)
        assert [1.0, 2.0, 10.0] == list(res)

    def test_fillna_float_none(self):
        t = XArray([1.0, 2.0, None])
        res = t.fillna(10.0)
        assert [1.0, 2.0, 10.0] == list(res)

    def test_fillna_nan_cast(self):
        t = XArray([1.0, 2.0, float('nan')])
        res = t.fillna(10)
        assert [1.0, 2.0, 10.0] == list(res)

    def test_fillna_none_float_cast(self):
        t = XArray([1.0, 2.0, None])
        res = t.fillna(10)
        assert [1.0, 2.0, 10.0] == list(res)


# noinspection PyClassHasNoInit
class TestXArrayTopkIndex:
    """
    Tests XArray topk_index
    """
    def test_topk_index_0(self):
        t = XArray([1, 2, 3])
        res = t.topk_index(0)
        assert [0, 0, 0] == list(res)

    def test_topk_index_1(self):
        t = XArray([1, 2, 3])
        res = t.topk_index(1)
        assert [0, 0, 1] == list(res)

    def test_topk_index_2(self):
        t = XArray([1, 2, 3])
        res = t.topk_index(2)
        assert [0, 1, 1] == list(res)

    def test_topk_index_3(self):
        t = XArray([1, 2, 3])
        res = t.topk_index(3)
        assert [1, 1, 1] == list(res)

    def test_topk_index_4(self):
        t = XArray([1, 2, 3])
        res = t.topk_index(4)
        assert [1, 1, 1] == list(res)

    def test_topk_index_float_1(self):
        t = XArray([1.0, 2.0, 3.0])
        res = t.topk_index(1)
        assert [0, 0, 1] == list(res)

    def test_topk_index_str_1(self):
        t = XArray(['a', 'b', 'c'])
        res = t.topk_index(1)
        assert [0, 0, 1] == list(res)

    def test_topk_index_list_1(self):
        t = XArray([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        res = t.topk_index(1)
        assert [0, 0, 1] == list(res)

    def test_topk_index_reverse_int(self):
        t = XArray([1, 2, 3])
        res = t.topk_index(1, reverse=True)
        assert [1, 0, 0] == list(res)

    def test_topk_index_reverse_float(self):
        t = XArray([1.0, 2.0, 3.0])
        res = t.topk_index(1, reverse=True)
        assert [1, 0, 0] == list(res)

    def test_topk_index_reverse_str(self):
        t = XArray(['a', 'b', 'c'])
        res = t.topk_index(1, reverse=True)
        assert [1, 0, 0] == list(res)

    def test_topk_index_reverse_list(self):
        t = XArray([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        res = t.topk_index(1, reverse=True)
        assert [1, 0, 0] == list(res)


# noinspection PyClassHasNoInit
class TestXArraySketchSummary:
    """
    Tests XArray sketch_summary
    """
    def test_sketch_summary_size(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert 5 == ss.size()

    def test_sketch_summary_min(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert 1 == ss.min()

    def test_sketch_summary_max(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert 5 == ss.max()

    def test_sketch_summary_mean(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert 3.0 == ss.mean()

    def test_sketch_summary_sum(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert 15 == ss.sum()

    def test_sketch_summary_var(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert 2.0 == ss.var()

    def test_sketch_summary_std(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert almost_equal(math.sqrt(2.0), ss.std())

    def test_sketch_summary_num_undefined(self):
        t = XArray([1, None, 3, None, 5])
        ss = t.sketch_summary()
        assert 2 == ss.num_undefined()

    def test_sketch_summary_num_unique(self):
        t = XArray([1, 3, 3, 3, 5])
        ss = t.sketch_summary()
        assert 3 == ss.num_unique()

    # TODO files on multiple workers
    # probably something wrong with combiner
    def test_sketch_summary_frequent_items(self):
        t = XArray([1, 3, 3, 3, 5])
        ss = t.sketch_summary()
        assert {1: 1, 3: 3, 5: 1} == ss.frequent_items()

    def test_sketch_summary_frequency_count(self):
        t = XArray([1, 3, 3, 3, 5])
        ss = t.sketch_summary()
        assert 1 == ss.frequency_count(1)
        assert 3 == ss.frequency_count(3)
        assert 1 == ss.frequency_count(5)


# noinspection PyClassHasNoInit
class TestXArrayAppend:
    """
    Tests XArray append
    """
    def test_append(self):
        t = XArray([1, 2, 3])
        u = XArray([10, 20, 30])
        res = t.append(u)
        assert [1, 2, 3, 10, 20, 30] == list(res)

    def test_append_empty_t(self):
        t = XArray([], dtype=int)
        u = XArray([10, 20, 30])
        res = t.append(u)
        assert [10, 20, 30] == list(res)

    def test_append_empty_u(self):
        t = XArray([1, 2, 3])
        u = XArray([], dtype=int)
        res = t.append(u)
        assert [1, 2, 3] == list(res)

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
        assert [1, 2, 3] == sorted(list(res))

    def test_unique_float_noop(self):
        t = XArray([1.0, 2.0, 3.0])
        res = t.unique()
        assert 3 == len(res)
        assert [1.0, 2.0, 3.0] == sorted(list(res))

    def test_unique_str_noop(self):
        t = XArray(['1', '2', '3'])
        res = t.unique()
        assert 3 == len(res)
        assert ['1', '2', '3'] == sorted(list(res))

    def test_unique_int(self):
        t = XArray([1, 2, 3, 1, 2])
        res = t.unique()
        assert 3 == len(res)
        assert [1, 2, 3] == sorted(list(res))

    def test_unique_float(self):
        t = XArray([1.0, 2.0, 3.0, 1.0, 2.0])
        res = t.unique()
        assert 3 == len(res)
        assert [1.0, 2.0, 3.0] == sorted(list(res))

    def test_unique_str(self):
        t = XArray(['1', '2', '3', '1', '2'])
        res = t.unique()
        assert 3 == len(res)
        assert ['1', '2', '3'] == sorted(list(res))


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
        assert [1, 2, 3] == list(res)
        assert res.dtype() is int

    def test_item_length_list(self):
        t = XArray([[1], [1, 2], [1, 2, 3]])
        res = t.item_length()
        assert [1, 2, 3] == list(res)
        assert res.dtype() is int

    def test_item_length_dict(self):
        t = XArray([{1: 'a'}, {1: 'a', 2: 'b'}, {1: 'a', 2: 'b', 3: '3'}])
        res = t.item_length()
        assert [1, 2, 3] == list(res)
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
        assert ['date.year'] == res.column_names()
        assert [int] == res.column_types()
        assert 3 == len(res)
        assert [2011, 2012, 2013] == list(res['date.year'])

    def test_split_datetime_year_mo(self):
        t = XArray([datetime.datetime(2011, 1, 1),
                    datetime.datetime(2012, 2, 2),
                    datetime.datetime(2013, 3, 3)])
        res = t.split_datetime('date', limit=['year', 'month'])
        assert isinstance(res, XFrame)
        assert ['date.year', 'date.month'] == res.column_names()
        assert [int, int] == res.column_types()
        assert 3 == len(res)
        assert [2011, 2012, 2013] == list(res['date.year'])
        assert [1, 2, 3] == list(res['date.month'])

    def test_split_datetime_all(self):
        t = XArray([datetime.datetime(2011, 1, 1, 1, 1, 1),
                    datetime.datetime(2012, 2, 2, 2, 2, 2),
                    datetime.datetime(2013, 3, 3, 3, 3, 3)])
        res = t.split_datetime('date')
        assert isinstance(res, XFrame)
        assert res.column_names() == ['date.year', 'date.month', 'date.day',
                                      'date.hour', 'date.minute', 'date.second']
        assert res.column_types() == [int, int, int, int, int, int]
        assert 3 == len(res)
        assert [2011, 2012, 2013] == list(res['date.year'])
        assert [1, 2, 3] == list(res['date.month'])
        assert [1, 2, 3] == list(res['date.day'])
        assert [1, 2, 3] == list(res['date.hour'])
        assert [1, 2, 3] == list(res['date.minute'])
        assert [1, 2, 3] == list(res['date.second'])

    def test_split_datetime_year_no_prefix(self):
        t = XArray([datetime.datetime(2011, 1, 1),
                    datetime.datetime(2012, 2, 2),
                    datetime.datetime(2013, 3, 3)])
        res = t.split_datetime(limit='year')
        assert isinstance(res, XFrame)
        assert ['X.year'] == res.column_names()
        assert [int] == res.column_types()
        assert 3 == len(res)
        assert [2011, 2012, 2013] == list(res['X.year'])

    def test_split_datetime_year_null_prefix(self):
        t = XArray([datetime.datetime(2011, 1, 1),
                    datetime.datetime(2012, 2, 2),
                    datetime.datetime(2013, 3, 3)])
        res = t.split_datetime(column_name_prefix=None, limit='year')
        assert isinstance(res, XFrame)
        assert ['year'] == res.column_names()
        assert [int] == res.column_types()
        assert 3 == len(res)
        assert [2011, 2012, 2013] == list(res['year'])

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
        assert ['X.0', 'X.1', 'X.2'] == res.column_names()
        assert {'X.0': 1, 'X.1': 0, 'X.2': 1} == res[0]
        assert {'X.0': 1, 'X.1': 1, 'X.2': 1} == res[1]
        assert {'X.0': 0, 'X.1': 1, 'X.2': None} == res[2]

    def test_unpack_list_limit(self):
        t = XArray([[1, 0, 1],
                    [1, 1, 1],
                    [0, 1]])
        res = t.unpack(limit=[1])
        assert ['X.1'] == res.column_names()
        assert {'X.1': 0} == res[0]
        assert {'X.1': 1} == res[1]
        assert {'X.1': 1} == res[2]

    def test_unpack_list_na_values(self):
        t = XArray([[1, 0, 1],
                    [1, 1, 1],
                    [0, 1]])
        res = t.unpack(na_value=0)
        assert ['X.0', 'X.1', 'X.2'] == res.column_names()
        assert {'X.0': 1, 'X.1': 0, 'X.2': 1} == res[0]
        assert {'X.0': 1, 'X.1': 1, 'X.2': 1} == res[1]
        assert {'X.0': 0, 'X.1': 1, 'X.2': 0} == res[2]

    def test_unpack_list_na_values_col_types(self):
        t = XArray([[1, 0, 1],
                    [1, 1, 1],
                    [0, 1]])
        res = t.unpack(column_types=[int, int, int], na_value=0)
        assert ['X.0', 'X.1', 'X.2'] == res.column_names()
        assert {'X.0': 1, 'X.1': 0, 'X.2': 1} == res[0]
        assert {'X.0': 1, 'X.1': 1, 'X.2': 1} == res[1]
        assert {'X.0': 0, 'X.1': 1, 'X.2': 0} == res[2]

    def test_unpack_list_cast_str(self):
        t = XArray([[1, 0, 1],
                    [1, 1, 1],
                    [0, 1]])
        res = t.unpack(column_types=[str, str, str])
        assert ['X.0', 'X.1', 'X.2'] == res.column_names()
        assert {'X.0': '1', 'X.1': '0', 'X.2': '1'} == res[0]
        assert {'X.0': '1', 'X.1': '1', 'X.2': '1'} == res[1]
        assert {'X.0': '0', 'X.1': '1', 'X.2': None} == res[2]

    def test_unpack_list_no_prefix(self):
        t = XArray([[1, 0, 1],
                    [1, 1, 1],
                    [0, 1]])
        res = t.unpack(column_name_prefix='')
        assert ['0', '1', '2'] == res.column_names()
        assert {'0': 1, '1': 0, '2': 1} == res[0]
        assert {'0': 1, '1': 1, '2': 1} == res[1]
        assert {'0': 0, '1': 1, '2': None} == res[2]

    def test_unpack_dict_limit(self):
        t = XArray([{'word': 'a', 'count': 1},
                    {'word': 'cat', 'count': 2},
                    {'word': 'is', 'count': 3},
                    {'word': 'coming', 'count': 4}])
        res = t.unpack(limit=['word', 'count'], column_types=[str, int])
        assert ['X.count', 'X.word'] == sorted(res.column_names())
        assert {'X.word': 'a', 'X.count': 1} == res[0]
        assert {'X.word': 'cat', 'X.count': 2} == res[1]
        assert {'X.word': 'is', 'X.count': 3} == res[2]
        assert {'X.word': 'coming', 'X.count': 4} == res[3]

    def test_unpack_dict_limit_word(self):
        t = XArray([{'word': 'a', 'count': 1},
                    {'word': 'cat', 'count': 2},
                    {'word': 'is', 'count': 3},
                    {'word': 'coming', 'count': 4}])
        res = t.unpack(limit=['word'])
        assert ['X.word'] == res.column_names()
        assert {'X.word': 'a'} == res[0]
        assert {'X.word': 'cat'} == res[1]
        assert {'X.word': 'is'} == res[2]
        assert {'X.word': 'coming'} == res[3]

    def test_unpack_dict_limit_count(self):
        t = XArray([{'word': 'a', 'count': 1},
                    {'word': 'cat', 'count': 2},
                    {'word': 'is', 'count': 3},
                    {'word': 'coming', 'count': 4}])
        res = t.unpack(limit=['count'])
        assert ['X.count'] == res.column_names()
        assert {'X.count': 1} == res[0]
        assert {'X.count': 2} == res[1]
        assert {'X.count': 3} == res[2]
        assert {'X.count': 4} == res[3]

    def test_unpack_dict_incomplete(self):
        t = XArray([{'word': 'a', 'count': 1},
                    {'word': 'cat', 'count': 2},
                    {'word': 'is'},
                    {'word': 'coming', 'count': 4}])
        res = t.unpack(limit=['word', 'count'], column_types=[str, int])
        assert ['X.count', 'X.word'] == sorted(res.column_names())
        assert {'X.count': 1, 'X.word': 'a'} == res[0]
        assert {'X.count': 2, 'X.word': 'cat'} == res[1]
        assert {'X.count': None, 'X.word': 'is'} == res[2]
        assert {'X.count': 4, 'X.word': 'coming'} == res[3]

    def test_unpack_dict(self):
        t = XArray([{'word': 'a', 'count': 1},
                    {'word': 'cat', 'count': 2},
                    {'word': 'is', 'count': 3},
                    {'word': 'coming', 'count': 4}])
        res = t.unpack()
        assert ['X.count', 'X.word'] == sorted(res.column_names())
        assert {'X.count': 1, 'X.word': 'a'} == res[0]
        assert {'X.count': 2, 'X.word': 'cat'} == res[1]
        assert {'X.count': 3, 'X.word': 'is'} == res[2]
        assert {'X.count': 4, 'X.word': 'coming'} == res[3]

    def test_unpack_dict_no_prefix(self):
        t = XArray([{'word': 'a', 'count': 1},
                    {'word': 'cat', 'count': 2},
                    {'word': 'is', 'count': 3},
                    {'word': 'coming', 'count': 4}])
        res = t.unpack(column_name_prefix=None)
        assert ['count', 'word'] == sorted(res.column_names())
        assert {'count': 1, 'word': 'a'} == res[0]
        assert {'count': 2, 'word': 'cat'} == res[1]
        assert {'count': 3, 'word': 'is'} == res[2]
        assert {'count': 4, 'word': 'coming'} == res[3]


# noinspection PyClassHasNoInit
class TestXArraySort:
    """
    Tests XArray sort
    """
    def test_sort_int(self):
        t = XArray([3, 2, 1])
        res = t.sort()
        assert [1, 2, 3] == list(res)

    def test_sort_float(self):
        t = XArray([3, 2, 1])
        res = t.sort()
        assert [1.0, 2.0, 3.0] == list(res)

    def test_sort_str(self):
        t = XArray(['c', 'b', 'a'])
        res = t.sort()
        assert ['a', 'b', 'c'] == list(res)

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
        assert [3, 2, 1] == list(res)

    def test_sort_float_desc(self):
        t = XArray([1.0, 2.0, 3.0])
        res = t.sort(ascending=False)
        assert [3.0, 2.0, 1.0] == list(res)

    def test_sort_str_desc(self):
        t = XArray(['a', 'b', 'c'])
        res = t.sort(ascending=False)
        assert ['c', 'b', 'a'] == list(res)


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
        assert [{'a': 0}, {}] == list(res)

    def test_dict_trim_by_keys_exclude(self):
        t = XArray([{'a': 0, 'b': 1, 'c': 2}, {'x': 1}])
        res = t.dict_trim_by_keys(['a'])
        assert [{'b': 1, 'c': 2}, {'x': 1}] == list(res)


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
        assert [{'b': 1, 'c': 2}, {'x': 1}] == list(res)


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
        assert 2 == len(res)
        assert ['X.0', 'X.1', 'X.2'] == sorted(list(res[0].keys()))
        assert ['a', 'b', 'c'] == sorted(list(res[0].values()))
        assert ['X.0', 'X.1', 'X.2'] == sorted(list(res[1].keys()))
        assert ['x', 'y', 'z'] == sorted(list(res[1].values()))


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
        assert 2 == len(res)
        assert ['X.0', 'X.1', 'X.2'] == sorted(list(res[0].keys()))
        assert [0, 1, 2] == sorted(list(res[0].values()))
        assert ['X.0', 'X.1', 'X.2'] == sorted(list(res[1].keys()))
        assert [10, 20, 30] == sorted(list(res[1].values()))


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
        assert [True, False] == list(res)


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
        assert [True, False] == list(res)
