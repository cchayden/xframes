"""
This object implements the base of the xframes inheritance hierarchy.
"""
import os


from xframes.spark_context import CommonSparkContext
import xframes.fileio as fileio
import xframes.version


class UnimplementedException(Exception):
    pass

FOOTER_STRS = ['Note: Only the head of the XFrame is printed.  You can use ',
               'print_rows(num_rows=m, num_columns=n) to print more rows and columns.']

LAZY_FOOTER_STRS = ['Note: Only the head of the XFrame is printed. This XFrame is lazily ',
                    'evaluated.  You can use len(xf) to force materialization.']

MAX_ROW_WIDTH = 70
HTML_MAX_ROW_WIDTH = 120


def version():
    return xframes.version.__version__


def wrap_rdd(rdd):
    from pyspark import RDD
    from xframes.xrdd import XRdd
    if rdd is None:
        return None
    if isinstance(rdd, RDD):
        return XRdd(rdd)
    if isinstance(rdd, XRdd):
        return rdd
    raise TypeError('Type is not RDD')


def _wrap_dstream(dstream):
    from pyspark.streaming import DStream
    from xframes.xstream import XStream
    if dstream is None:
        return None
    if isinstance(dstream, DStream):
        return XStream(dstream)
    if isinstance(dstream, XStream):
        return dstream
    raise TypeError('Type is not DStream')

def check_input_uri(uri):
    if ',' in uri:
        uri_list = uri.split(',')
    else:
        uri_list = [uri]
    for path in uri_list:
        if not fileio.exists(path):
            raise ValueError('Input file does not exist: {}'.format(path))

def check_output_uri(uri):
    dirname = os.path.dirname(uri)
    if not fileio.exists(dirname):
        fileio.make_dir(dirname)
        if not fileio.exists(dirname):
            raise ValueError('Output directory does not exist: {}'.format(dirname))
