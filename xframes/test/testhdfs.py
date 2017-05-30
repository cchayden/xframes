import os
import pickle

from xframes import XArray, XFrame
from xframes import fileio

hdfs_prefix = 'hdfs://localhost:8020'

# Needs to be tested
# XArray(saved file w/ _metadata)
# XArray save as text
# XArray save as csv
# XFrame(saved file w/ _metadata)
# XFrame save
# XFrame save as csv


# noinspection PyClassHasNoInit
class TestXArrayConstructorLoad:
    """
    Tests XArray constructors that loads from file.
    """

    def test_construct_file_int(self):
        path = '{}/user/xframes/files/test-array-int'.format(hdfs_prefix)
        t = XArray(path)
        assert len(t) == 4
        assert t.dtype() is int
        assert t[0] == 1

    def test_construct_local_file_float(self):
        t = XArray('{}/user/xframes/files/test-array-float'.format(hdfs_prefix))
        assert len(t) == 4
        assert t.dtype() is float
        assert t[0] == 1.0

    def test_construct_local_file_str(self):
        t = XArray('{}/user/xframes/files/test-array-str'.format(hdfs_prefix))
        assert len(t) == 4
        assert t.dtype() is str
        assert t[0] == 'a'

    def test_construct_local_file_list(self):
        t = XArray('{}/user/xframes/files/test-array-list'.format(hdfs_prefix))
        assert len(t) == 4
        assert t.dtype() is list
        assert t[0] == [1, 2]

    def test_construct_local_file_dict(self):
        t = XArray('{}/user/xframes/files/test-array-dict'.format(hdfs_prefix))
        assert len(t) == 4
        assert t.dtype() is dict
        assert t[0] == {1: 'a', 2: 'b'}


# noinspection PyClassHasNoInit
class TestXArraySaveCsv:
    """
    Tests XArray save csv format
    """
    def test_save(self):
        t = XArray([1, 2, 3])
        path = '{}/tmp/array-csv.csv'.format(hdfs_prefix)
        t.save(path)
        with fileio.open_file(path) as f:
            assert f.readline().strip() == '1'
            assert f.readline().strip() == '2'
            assert f.readline().strip() == '3'
        fileio.delete(path)

    def test_save_format(self):
        t = XArray([1, 2, 3])
        path = '{}/tmp/array-csv'.format(hdfs_prefix)
        t.save(path, format='csv')
        with fileio.open_file(path) as f:
            assert f.readline().strip() == '1'
            assert f.readline().strip() == '2'
            assert f.readline().strip() == '3'
        fileio.delete(path)


# noinspection PyClassHasNoInit
class TestXArraySaveText:
    """
    Tests XArray save text format
    """
    def test_save(self):
        t = XArray([1, 2, 3])
        path = '{}/tmp/array-csv'.format(hdfs_prefix)
        t.save(path)
        success_path = os.path.join(path, '_SUCCESS')
        assert fileio.is_file(success_path)
        fileio.delete(path)

    def test_save_format(self):
        t = XArray([1, 2, 3])
        path = '{}/tmp/array-csv'.format(hdfs_prefix)
        t.save(path, format='text')
        success_path = os.path.join(path, '_SUCCESS')
        assert fileio.is_file(success_path)
        fileio.delete(path)


# noinspection PyClassHasNoInit
class TestXFrameConstructor:
    """
    Tests XFrame constructors that create data from local sources.
    """

    def test_construct_auto_dataframe(self):
        path = '{}/user/xframes/files/test-frame-auto.csv'.format(hdfs_prefix)
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
        path = '{}/user/xframes/files/test-frame.csv'.format(hdfs_prefix)
        res = XFrame(path)
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_construct_auto_str_tsv(self):
        path = '{}/user/xframes/files/test-frame.tsv'.format(hdfs_prefix)
        res = XFrame(path)
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_construct_auto_str_psv(self):
        path = '{}/user/xframes/files/test-frame.psv'.format(hdfs_prefix)
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
        path = '{}/user/xframes/files/test-frame.txt'.format(hdfs_prefix)
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
        path = '{}/user/xframes/files/test-frame'.format(hdfs_prefix)
        res = XFrame(path)
        res = res.sort('id')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_construct_auto_str_xframe(self):
        # construct an XFrame given a file with unrecognized file extension
        path = '{}/user/xframes/files/test-frame'.format(hdfs_prefix)
        res = XFrame(path)
        res = res.sort('id')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_construct_str_csv(self):
        # construct and XFrame given a text file
        # interpret as csv
        path = '{}/user/xframes/files/test-frame.txt'.format(hdfs_prefix)
        res = XFrame(path, format='csv')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}

    def test_construct_str_xframe(self):
        # construct and XFrame given a saved xframe
        path = '{}/user/xframes/files/test-frame'.format(hdfs_prefix)
        res = XFrame(path, format='xframe')
        res = res.sort('id')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}


# noinspection PyClassHasNoInit
class TestXFrameReadCsv:
    """
    Tests XFrame read_csv
    """

    def test_read_csv(self):
        path = '{}/user/xframes/files/test-frame.csv'.format(hdfs_prefix)
        res = XFrame.read_csv(path)
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}


# noinspection PyClassHasNoInit
class TestXFrameReadText:
    """
    Tests XFrame read_text
    """

    def test_read_text(self):
        path = '{}/user/xframes/files/test-frame-text.txt'.format(hdfs_prefix)
        res = XFrame.read_text(path)
        assert len(res) == 3
        assert res.column_names() == ['text']
        assert res.column_types() == [str]
        assert res[0] == {'text': 'This is a test'}
        assert res[1] == {'text': 'of read_text.'}
        assert res[2] == {'text': 'Here is another sentence.'}


# noinspection PyClassHasNoInit
class TestXFrameReadParquet:
    """
    Tests XFrame read_parquet
    """

    def test_read_parquet_str(self):
        t = XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        path = '{}/tmp/frame-parquet'.format(hdfs_prefix)
        t.save(path, format='parquet')

        res = XFrame('{}/tmp/frame-parquet.parquet'.format(hdfs_prefix))
        # results may not come back in the same order
        res = res.sort('id')
        assert len(res) == 3
        assert res.column_names() == ['id', 'val']
        assert res.column_types() == [int, str]
        assert res[0] == {'id': 1, 'val': 'a'}
        assert res[1] == {'id': 2, 'val': 'b'}
        assert res[2] == {'id': 3, 'val': 'c'}
        fileio.delete(path)


# noinspection PyClassHasNoInit
class TestXFrameSaveBinary:
    """
    Tests XFrame save binary format
    """

    def test_save(self):
        t = XFrame({'id': [30, 20, 10], 'val': ['a', 'b', 'c']})
        path = '{}/tmp/frame'.format(hdfs_prefix)
        t.save(path, format='binary')
        with fileio.open_file(os.path.join(path, '_metadata')) as f:
            metadata = pickle.load(f)
            assert metadata == [['id', 'val'], [int, str]]
        # TODO find some way to check the data
        fileio.delete(path)


# noinspection PyClassHasNoInit
class TestXFrameSaveCsv:
    """
    Tests XFrame save csv format
    """

    def test_save(self):
        t = XFrame({'id': [30, 20, 10], 'val': ['a', 'b', 'c']})
        path = '{}/tmp/frame-csv'.format(hdfs_prefix)
        t.save(path, format='csv')

        with fileio.open_file(path + '.csv') as f:
            heading = f.readline().rstrip()
            assert heading == 'id,val'
            assert f.readline().rstrip() == '30,a'
            assert f.readline().rstrip() == '20,b'
            assert f.readline().rstrip() == '10,c'
        fileio.delete(path + '.csv')


# noinspection PyClassHasNoInit
class TestXFrameSaveParquet:
    """
    Tests XFrame save for parquet files
    """
    def test_save(self):
        t = XFrame({'id': [30, 20, 10], 'val': ['a', 'b', 'c']})
        path = '{}/tmp/frame-parquet'.format(hdfs_prefix)
        t.save(path, format='parquet')
        # TODO verify
        fileio.delete(path + '.parquet')
