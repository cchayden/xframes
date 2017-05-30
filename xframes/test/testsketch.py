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
        assert ss.size() == 5
        assert ss.max() == 5
        assert ss.min() == 1
        assert ss.sum() == 15
        assert ss.mean() == 3
        assert almost_equal(ss.std(), 1.4142135623730951)
        assert almost_equal(ss.var(), 2.0)

    def test_avg_length_int(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert ss.avg_length() == 1

    def test_avg_length_float(self):
        t = XArray([1.0, 2.0, 3.0, 4.0, 5.0])
        ss = t.sketch_summary()
        assert ss.avg_length() == 1

    def test_avg_length_list(self):
        t = XArray([[1, 2, 3, 4], [5, 6]])
        ss = t.sketch_summary()
        assert ss.avg_length() == 3

    def test_avg_length_dict(self):
        t = XArray([{1: 1, 2: 2, 3: 3, 4: 4}, {5: 5, 6: 6}])
        ss = t.sketch_summary()
        assert ss.avg_length() == 3

    def test_avg_length_str(self):
        t = XArray(['a', 'bb', 'ccc', 'dddd', 'eeeee'])
        ss = t.sketch_summary()
        assert ss.avg_length() == 3

    def test_avg_length_empty(self):
        t = XArray([])
        ss = t.sketch_summary()
        assert ss.avg_length() == 0

    def test_num_undefined(self):
        t = XArray([1, 2, 3, 4, 5, None])
        ss = t.sketch_summary()
        assert ss.num_undefined() == 1

    def test_num_unique(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert ss.num_unique() == 5

    def test_frequent_items(self):
        t = XArray([1, 2, 3, 2])
        ss = t.sketch_summary()
        assert ss.frequent_items() == {1: 1, 2: 2, 3: 1}

    def test_tf_idf_list(self):
        t = XArray([['this', 'is', 'a', 'test'], ['another', 'test']])
        ss = t.sketch_summary()
        tf_idf = ss.tf_idf()
        assert tf_idf[0] == {'this': 0.4054651081081644,
                             'a': 0.4054651081081644,
                             'is': 0.4054651081081644,
                             'test': 0.0}
        assert tf_idf[1] == {'test': 0.0,
                             'another': 0.4054651081081644}

    def test_tf_idf_str(self):
        t = XArray(['this is a test', 'another test'])
        ss = t.sketch_summary()
        tf_idf = ss.tf_idf()
        assert tf_idf[0] == {'this': 0.4054651081081644,
                             'a': 0.4054651081081644,
                             'is': 0.4054651081081644,
                             'test': 0.0}
        assert tf_idf[1] == {'test': 0.0,
                             'another': 0.4054651081081644}

    def test_quantile(self):
        t = XArray([1, 2, 3, 4, 5])
        ss = t.sketch_summary()
        assert almost_equal(ss.quantile(0.5), 3, places=1)
        assert almost_equal(ss.quantile(0.8), 4, places=1)
        assert almost_equal(ss.quantile(0.9), 5, places=1)
        assert almost_equal(ss.quantile(0.99), 5, places=1)

    def test_frequency_count(self):
        t = XArray([1, 2, 3, 4, 5, 3])
        ss = t.sketch_summary()
        assert ss.frequency_count(3) == 2

    def test_missing(self):
        t = XArray([None], dtype=int)
        ss = t.sketch_summary()
        assert ss.min() is None
        assert ss.max() is None
        assert ss.mean() == 0.0
        assert ss.sum() == 0.0
        assert ss.var() is None
        assert ss.std() is None
        assert ss.max() is None
        assert ss.avg_length() == 0
