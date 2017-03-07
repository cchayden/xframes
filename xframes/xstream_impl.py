"""
This module provides an implementation of XStream using pySpark RDDs.
"""
import os
import json
import random
import array
import pickle
import csv
import StringIO
import ast
import shutil
import re
import copy
from datetime import datetime
from dateutil import parser as date_parser
import logging


import xframes.fileio as fileio
from xframes.xobject_impl import XObjectImpl
from xframes.traced_object import TracedObject
from pyspark.streaming.kafka import KafkaUtils
from xframes.xframe import XFrame
from xframes.spark_context import CommonSparkContext
from xframes.util import infer_type_of_rdd
from xframes.util import cache, uncache, persist, unpersist
from xframes.util import is_missing, is_missing_or_empty
from xframes.util import to_ptype, to_schema_type, hint_to_schema_type, pytype_from_dtype, safe_cast_val
from xframes.util import distribute_seed
from xframes.xrdd import XRdd

class XStreamImpl(XObjectImpl, TracedObject):
    """ Implementation for XFrame. """

    def __init__(self, column_names):
        """ Instantiate a XStream implementation.
        """
        self._entry()
        super(XStreamImpl, self).__init__(None)
        self._raw_stream = None
        self._stream = None
        self.column_names = column_names


    def create_from_kafka_topic(self, topics, kafka_servers):
        ssc = self.streaming_context()
        self._raw_stream = KafkaUtils.createDirectStream(ssc, topics, {'bootstrap.servers': kafka_servers})
        # loads the kafka value; drop the topic; assume it is in json format
        self._stream = self._raw_stream.map(lambda x: json.loads(x[1]))

    @staticmethod
    def start():
        ssc = CommonSparkContext.streaming_context()
        ssc.start()

    @staticmethod
    def stop(stop_spark_context, stop_gracefully):
        ssc = CommonSparkContext.streaming_context()
        ssc.stop(stop_spark_context, stop_gracefully)

    def process_stream(self, row_fn, init_fn):
        def process_rdd(rdd):
            xf = XFrame.from_rdd(rdd, column_names=self.column_names)
            xf.foreach(row_fn, init_fn)

        self._stream.foreachRDD(process_rdd)

