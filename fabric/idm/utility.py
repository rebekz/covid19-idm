import findspark
findspark.init() # noqa

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import pandas as pd
import yaml


class SparkJob:
    """
    Initiate a SparkSession

    Parameters
    ----------
    appName : str
        app name for SparkSession
    master : str
        chosen Spark master either yarn or local. default is local.
    """

    def __init__(self, appName, master="local[*]"):
        self.spark = SparkSession \
            .builder.master(master).appName(appName).getOrCreate()

    def __enter__(self):
        return self.spark

    def __exit__(self, *args):
        self.spark.stop()

def load_data(metadata, metric="CONFIRMED"):
    """
    Load data from Hive table. Transforming db_disease_outbreak-like structure to
    date, geo_id, metric format.

    Parameters
    ----------
    metadata : dict
        input table metadata
    metric : str
        metric column name about cases

    Returns
    -------
    cases : pd.DataFrame
        Reported cases DataFrame indexed by date with columns: `geo_id, cumulative_cases`
    """

    input_table = metadata["input"]["input_table"]
    loc_granularity = metadata["input"]["loc_granularity"]
    geoid_col = metadata["input"]["geoid_col"]
    metricval_col = metadata["input"]["metric_val_col"]
    metricname_col = metadata["input"]["metric_name_col"]
    metricgranularity_col = metadata["input"]["metric_granularity_col"]
    date_col = metadata["input"]["date_col"]
    hourid_col = metadata["input"]["hour_id_col"]

    with SparkJob("ai-idm-load-input-data") as spark:
        df_filtered = spark.read.table(input_table) \
                .filter(F.col(metricgranularity_col).isin(loc_granularity)) \
                .filter(F.col(metricname_col) == metric) \
                .filter((F.col(geoid_col).isNotNull()) & (F.col(geoid_col) != ""))

        window_spec = Window.partitionBy(F.col(geoid_col), F.col(date_col)).orderBy(F.col(hourid_col).desc())

        active_cases = df_filtered.sort(F.asc(date_col)) \
               .withColumn("r_number", F.row_number().over(window_spec)) \
               .filter(F.col("r_number") == 1) \
               .groupby(geoid_col, metricname_col, date_col).agg(F.sum(metricval_col).alias(metricval_col)) \
               .select(geoid_col, date_col, metricname_col, metricval_col)

        active_cases_pd = active_cases.toPandas()

    confirmed_cases = active_cases_pd[active_cases_pd[metricname_col] == metric]
    confirmed_cases_sorted = confirmed_cases.set_index(confirmed_cases[date_col]).sort_index()

    _metric = f"{metric.lower()}"
    confirmed_cases_sorted[_metric] = confirmed_cases_sorted[metricval_col].apply(lambda x: abs(x))
    confirmed_cases_sorted = confirmed_cases_sorted[[geoid_col, _metric]].reset_index()
    confirmed_cases_sorted["increment"] = confirmed_cases_sorted.groupby(by=geoid_col)[_metric].diff()
    cases = confirmed_cases_sorted.set_index(date_col)
    cases.columns = ["geo_id", _metric, "increment"]

    # create increment case for capturing series of new cases
    cases.index = pd.to_datetime(cases.index)

    return cases
