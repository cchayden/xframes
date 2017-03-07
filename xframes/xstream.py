"""
This module defines the XStream class which provides the
ability to process streaming operations.
"""

import array
from textwrap import wrap
import inspect
import time
import itertools
from dateutil import parser as date_parser
import datetime
import copy
import ast
import logging
import types
from sys import stderr

import pyspark

from xframes.deps import pandas, HAS_PANDAS
from xframes.prettytable import PrettyTable
from xframes.deps import dataframeplus, HAS_DATAFRAME_PLUS
from xframes.xobject import XObject
from xframes.xstream_impl import XStreamImpl
from xframes.xplot import XPlot
from xframes.xarray_impl import infer_type_of_list
from xframes.util import make_internal_url, classify_type, classify_auto
from xframes.xarray import XArray
import xframes
import util

"""
Copyright (c) 2017, Charles Hayden, Inc.
All rights reserved.
"""

__all__ = ['XStream']


class XStream(XObject):
    """
    Provides for streams of XFrames.
    """

    def __init__(self, column_names=None, impl=None, verbose=False):

        if impl:
            self._impl = impl
            return

        self._impl = XStreamImpl(column_names)

    def create_from_kafka_topic(self, topics, kafka_servers=None):
        if isinstance(topics, basestring):
            topics = [topics]
        if not isinstance(topics, list):
            raise TypeError('Topics must be string or list.')
        if kafka_servers is None:
            kafka_servers = 'localhost:9092'
        elif isinstance(topics, list):
            kafka_servers = ','.join(kafka_servers)

        self._impl.create_from_kafka_topic(topics, kafka_servers)

    def start(self):
        self._impl.start()

    def stop(self, stop_spark_context=True, stop_gracefully=False):
        self._impl.stop(stop_spark_context, stop_gracefully)

    def process_stream(self, row_fn, init_fn=None):
        self._impl.process_stream(row_fn, init_fn)
