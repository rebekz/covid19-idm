# -*- coding: utf-8 -*-
import os
import numpy
import pytest
import logging
import findspark

# current is ./spark/tests
cur_dir = os.path.dirname(os.path.realpath(__file__))

# workaround to use locally downloaded spark
default_spark_home = os.path.join(cur_dir, os.pardir, os.pardir, os.pardir, 'spark')
findspark.init(os.getenv('SPARK_HOME', default_spark_home))
from pyspark.sql import SparkSession  # noqa


def quiet():
    """Turn down logging / warning for the test context"""
    logger = logging.getLogger('py4j')
    logger.setLevel(logging.WARN)
    numpy.warnings.filterwarnings('ignore')


@pytest.fixture(scope='session')
def spark(request):
    cur_dir = os.path.dirname(os.path.realpath(__file__))

    """Setup spark session"""
    session = (SparkSession.builder
               .master('local')
               # workaround to avoid snappy library issue
               .config('spark.sql.parquet.compression.codec', 'uncompressed')
               .config('spark.sql.warehouse.dir', cur_dir)
               # scala patch for model persistence
               .appName('idm-test')
               .enableHiveSupport()
               .getOrCreate())

    def cleanup():
        session.stop()
        session._jvm.System.clearProperty('spark.driver.port')

    request.addfinalizer(cleanup)

    quiet()
    return session

