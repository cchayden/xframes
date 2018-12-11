from xframes import fileio


# run with pytest
class TestFileioLength:
    """
    Tests length function
    """

    def test_length_file(self):
        path = 'files/test-frame-auto.csv'
        length = fileio.length(path)
        assert 166 == length

    def test_length_fdir(self):
        path = 'files/test-frame'
        length = fileio.length(path)
        assert 1182 == length
