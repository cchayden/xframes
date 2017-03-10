"""
This object implements the base of the xframes inheritance hierarchy.
"""
import os

from pyspark import RDD
from pyspark.streaming import DStream

from xframes.spark_context import CommonSparkContext
import xframes.fileio as fileio


class UnimplementedException(Exception):
    pass


class XObjectImpl(object):
    """ Implementation for XObject. """

    @staticmethod
    def _wrap_rdd(rdd):
        from xframes.xrdd import XRdd
        if rdd is None:
            return None
        if isinstance(rdd, RDD):
            return XRdd(rdd)
        if isinstance(rdd, XRdd):
            return rdd
        raise TypeError('Type is not RDD')

    @staticmethod
    def _wrap_dstream(dstream):
        from xframes.xstream import XStream
        if dstream is None:
            return None
        if isinstance(dstream, DStream):
            return XStream(dstream)
        if isinstance(dstream, XStream):
            return dstream
        raise TypeError('Type is not DStream')

    @staticmethod
    def spark_context():
        return CommonSparkContext.spark_context()

    @staticmethod
    def spark_sql_context():
        return CommonSparkContext.spark_sql_context()

    @staticmethod
    def hive_context():
        return CommonSparkContext.hive_context()

    @staticmethod
    def streaming_context():
        return CommonSparkContext.streaming_context()

    @staticmethod
    def check_input_uri(uri):
        if ',' in uri:
            uri_list = uri.split(',')
        else:
            uri_list = [uri]
        for path in uri_list:
            if not fileio.exists(path):
                raise ValueError('Input file does not exist: {}'.format(path))

    @staticmethod
    def check_output_uri(uri):
        dirname = os.path.dirname(uri)
        if not fileio.exists(dirname):
            fileio.make_dir(dirname)
            if not fileio.exists(dirname):
                raise ValueError('Output directory does not exist: {}'.format(dirname))
