from __future__ import absolute_import


# pytest testsketch.py
# pytest testsketch::TestSketchConstructor
# pytest testsketch::TestSketchConstructor::test_construct

from xframes.xarray import XArray


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
class TestSketchConstructor:
    """
    Tests sketch constructor
    """

    def test_construct(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert 5 == ss.size()
        assert 5 == ss.max()
        assert 1 == ss.min()
        assert 15 == ss.sum()
        assert 3 == ss.mean()
        assert almost_equal(1.4142135623730951, ss.std())
        assert almost_equal(2.0, ss.var())

    def test_avg_length_int(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert 1 == ss.avg_length()

    def test_avg_length_float(self):
        t = XArray([1.0, 2.0, 3.0, 4.0, 5.0])
        ss = t.sketch_summary()
        assert 1 == ss.avg_length()

    def test_avg_length_list(self):
        t = XArray([[1, 2, 3, 4], [5, 6]])
        ss = t.sketch_summary()
        assert 3 == ss.avg_length()

    def test_avg_length_dict(self):
        t = XArray([{1: 1, 2: 2, 3: 3, 4: 4}, {5: 5, 6: 6}])
        ss = t.sketch_summary()
        assert 3 == ss.avg_length()

    def test_avg_length_str(self):
        t = XArray(['a', 'bb', 'ccc', 'dddd', 'eeeee'])
        ss = t.sketch_summary()
        assert 3 == ss.avg_length()

    def test_avg_length_empty(self):
        t = XArray([])
        ss = t.sketch_summary()
        assert 0 == ss.avg_length()

    def test_num_undefined(self):
        t = XArray([1, 2, 3, 4, 5, None])
        ss = t.sketch_summary()
        assert 1 == ss.num_undefined()

    def test_num_unique(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert 5 == ss.num_unique()

    def test_frequent_items(self):
        t = XArray([1, 2, 3, 2])
        ss = t.sketch_summary()
        assert {1: 1, 2: 2, 3: 1} == ss.frequent_items()

    def test_tf_idf_list(self):
        t = XArray([['this', 'is', 'a', 'test'], ['another', 'test']])
        ss = t.sketch_summary()
        tf_idf = ss.tf_idf()
        assert {'this': 0.4054651081081644,
                'a': 0.4054651081081644,
                'is': 0.4054651081081644,
                'test': 0.0} == tf_idf[0]
        assert {'test': 0.0,
                'another': 0.4054651081081644} == tf_idf[1]

    def test_tf_idf_str(self):
        t = XArray(['this is a test', 'another test'])
        ss = t.sketch_summary()
        tf_idf = ss.tf_idf()
        assert {'this': 0.4054651081081644,
                'a': 0.4054651081081644,
                'is': 0.4054651081081644,
                'test': 0.0} == tf_idf[0]
        assert {'test': 0.0,
                'another': 0.4054651081081644} == tf_idf[1]

    def test_quantile(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert almost_equal(3, ss.quantile(0.5), places=1)
        assert almost_equal(4, ss.quantile(0.8), places=1)
        assert almost_equal(5, ss.quantile(0.9), places=1)
        assert almost_equal(5, ss.quantile(0.99), places=1)

    def test_frequency_count(self):
        t = XArray([1, 2, 3, 4, 5, 3])
        ss = t.sketch_summary()
        assert 2 == ss.frequency_count(3)

    def test_missing(self):
        t = XArray([None], dtype=int)
        ss = t.sketch_summary()
        assert None is ss.min()
        assert None is ss.max()
        assert 0.0 == ss.mean()
        assert 0.0 == ss.sum()
        assert None is ss.var()
        assert None is ss.std()
        assert None is ss.max()
        assert 0 == ss.avg_length()
