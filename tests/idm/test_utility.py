import os
import pytest
import textwrap
import yaml
from pyspark.sql import SparkSession  # noqa

import pandas as pd
from fabric.idm.utility import load_data

db = "idm"
input_table = "db_disease_outbreak_two"
GEO_ID_COL = "province"
METRIC_GRANULARITY_COL = "metric_granularity"
METRIC_NAME_COL = "metric_name"
DATE_COL = "date_id"
HOUR_ID_COL = "hour_id"
METRIC_VALUE_COL = "metric_value"

@pytest.fixture
def input_data(spark, request, tmpdir):
    spark.sql(f"DROP DATABASE IF EXISTS {db} CASCADE")
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db}")

    (spark
        .read
        .parquet("tests/resources/cases.parquet")
        .write
        .saveAsTable(f"{db}.{input_table}"))

    yaml_string = textwrap.dedent(f"""\
    ---
    ForecastModel:
      input:
        input_table: {db}.{input_table}
        date_col: {DATE_COL}
        geoid_col: {GEO_ID_COL}
        loc_granularity:
          - province
        metric_val_col: {METRIC_VALUE_COL}
        metric_granularity_col: {METRIC_GRANULARITY_COL}
        metric_name_col: {METRIC_NAME_COL}
        hour_id_col: {HOUR_ID_COL}
      output:
        output_path: {tmpdir}
      model_params:
        bayesian:
          c0_mu: 10
          c0_sigma: 1
          r_mu: 0.2
          r_mu_sigma: 0.1
          r_sigma: 0.5
          k_mu: 30
          k_mu_sigma: 1
          k_sigma: 1
          forecast_days: 365
          model_path: {tmpdir}
          cores: 2
          chains: 2
          tune: 5
          draws: 5
          target_accept: 0.8
        curvefit:
          maxfev: 100
          forecast_days: 30
      with_train: true
      methods:
        - curvefit
        - bayesian
      """)

    yaml_dict = yaml.safe_load(yaml_string)["ForecastModel"]

    def cleanup():
        spark.sql(f'DROP DATABASE {db} CASCADE')
        spark.stop()

    request.addfinalizer(cleanup)

    return yaml_dict

def test_load_data(input_data):

    data = load_data(metadata=input_data, metric="CONFIRMED")

    expected_columns = ["geo_id", "confirmed", "increment"]

    assert data.columns.tolist() == expected_columns
    assert data.index.name == "date_id"
