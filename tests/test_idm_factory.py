import os
import pytest
import textwrap
import yaml
import pandas as pd
from shutil import copyfile

from fabric.utility import parse_file
from fabric.idm.utility import load_data
from fabric.idm.models import ForecastModel
from fabric.idm_factory import main as idm_factory

db = "idm"
input_table = "db_disease_outbreak"
GEO_ID_COL = "geo_id"
METRIC_GRANULARITY_COL = "metric_granularity"
METRIC_NAME_COL = "metric_name"
DATE_COL = "date_id"
HOUR_ID_COL = "hour_id"
METRIC_VALUE_COL = "metric_value"

@pytest.fixture
def input_df(spark, request, tmpdir):

    spark.sql(f"DROP DATABASE IF EXISTS {db} CASCADE")
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db}")

    input_data = pd.DataFrame(data=[
        ("JAKARTA", "province", "CONFIRMED", 0, "20200401", "23"),
        ("JAKARTA", "province", "CONFIRMED", 10, "20200402", "23"),
        ("JAKARTA", "province", "CONFIRMED", 20, "20200402", "23"),
        ("JAKARTA", "province", "CONFIRMED", 30, "20200402", "23"),
        ("JAKARTA", "province", "CONFIRMED", 10, "20200403", "23"),
        ("JAKARTA", "province", "CONFIRMED", 25, "20200404", "23"),
        ("JAKARTA", "province", "CONFIRMED", 11, "20200405", "23"),
        ("JAKARTA", "province", "CONFIRMED", 12, "20200406", "23"),
        ("JAKARTA", "province", "CONFIRMED", 10, "20200407", "23"),
        ("JAKARTA", "province", "CONFIRMED", 10, "20200408", "23"),
        ("JAKARTA", "province", "CONFIRMED", 10, "20200409", "23"),
        ("JAKARTA", "province", "CONFIRMED", 10, "20200410", "23"),
        ("JAKARTA", "province", "CONFIRMED", 10, "20200411", "23"),
        ("JAKARTA", "province", "CONFIRMED", 10, "20200412", "23"),
        ("JAKARTA", "province", "CONFIRMED", 10, "20200413", "23"),
        ("JAKARTA", "province", "DEATHS", 0, "20200401", "23"),
        ("JAKARTA", "province", "DEATHS", 10, "20200402", "23"),
        ("JAKARTA", "province", "DEATHS", 20, "20200402", "23"),
        ("JAKARTA", "province", "DEATHS", 30, "20200402", "23"),
        ("JAKARTA", "province", "DEATHS", 10, "20200403", "23"),
        ("JAKARTA", "province", "DEATHS", 25, "20200404", "23"),
        ("JAKARTA", "province", "DEATHS", 11, "20200405", "23"),
        ("JAKARTA", "province", "DEATHS", 12, "20200406", "23"),
        ("JAKARTA", "province", "DEATHS", 10, "20200407", "23"),
        ("JAKARTA", "province", "DEATHS", 10, "20200408", "23"),
        ("JAKARTA", "province", "DEATHS", 10, "20200409", "23"),
        ("JAKARTA", "province", "DEATHS", 10, "20200410", "23"),
        ("JAKARTA", "province", "DEATHS", 10, "20200411", "23"),
        ("JAKARTA", "province", "DEATHS", 10, "20200412", "23"),
        ("JAKARTA", "province", "DEATHS", 10, "20200413", "23"),
        ("BANTEN", "province", "CONFIRMED", 0, "20200401", "23"),
        ("BANTEN", "province", "CONFIRMED", 10, "20200402", "23"),
        ("BANTEN", "province", "CONFIRMED", 20, "20200402", "23"),
        ("BANTEN", "province", "CONFIRMED", 30, "20200402", "23"),
        ("BANTEN", "province", "CONFIRMED", 10, "20200403", "23"),
        ("BANTEN", "province", "CONFIRMED", 25, "20200404", "23"),
        ("BANTEN", "province", "CONFIRMED", 11, "20200405", "23"),
        ("BANTEN", "province", "CONFIRMED", 12, "20200406", "23"),
        ("BANTEN", "province", "CONFIRMED", 10, "20200407", "23"),
        ("BANTEN", "province", "CONFIRMED", 10, "20200408", "23"),
        ("BANTEN", "province", "CONFIRMED", 10, "20200409", "23"),
        ("BANTEN", "province", "CONFIRMED", 10, "20200410", "23"),
        ("BANTEN", "province", "CONFIRMED", 10, "20200411", "23"),
        ("BANTEN", "province", "CONFIRMED", 10, "20200412", "23"),
        ("BANTEN", "province", "CONFIRMED", 10, "20200413", "23"),
        ("BANTEN", "province", "DEATHS", 0, "20200401", "23"),
        ("BANTEN", "province", "DEATHS", 10, "20200402", "23"),
        ("BANTEN", "province", "DEATHS", 20, "20200402", "23"),
        ("BANTEN", "province", "DEATHS", 30, "20200402", "23"),
        ("BANTEN", "province", "DEATHS", 10, "20200403", "23"),
        ("BANTEN", "province", "DEATHS", 25, "20200404", "23"),
        ("BANTEN", "province", "DEATHS", 11, "20200405", "23"),
        ("BANTEN", "province", "DEATHS", 12, "20200406", "23"),
        ("BANTEN", "province", "DEATHS", 10, "20200407", "23"),
        ("BANTEN", "province", "DEATHS", 10, "20200408", "23"),
        ("BANTEN", "province", "DEATHS", 10, "20200409", "23"),
        ("BANTEN", "province", "DEATHS", 10, "20200410", "23"),
        ("BANTEN", "province", "DEATHS", 10, "20200411", "23"),
        ("BANTEN", "province", "DEATHS", 10, "20200412", "23"),
        ("BANTEN", "province", "DEATHS", 10, "20200413", "23")
    ], columns=[GEO_ID_COL, METRIC_GRANULARITY_COL, METRIC_NAME_COL, METRIC_VALUE_COL, DATE_COL, HOUR_ID_COL])

    (spark
        .createDataFrame(input_data, f"{GEO_ID_COL}:string, {METRIC_GRANULARITY_COL}:string, {METRIC_NAME_COL}:string, {METRIC_VALUE_COL}:int, {DATE_COL}:string, {HOUR_ID_COL}:string")
        .write
        .saveAsTable(f"{db}.{input_table}")
    )

    def cleanup():
        spark.sql(f'DROP DATABASE {db} CASCADE')
        spark.stop()

    request.addfinalizer(cleanup)

@pytest.fixture
def input_config(tmpdir):
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
        output_path: data
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
          model_path: model
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
      """)

    try:
        os.mkdir(os.path.join(tmpdir, "data"))
        os.mkdir(os.path.join(tmpdir, "model"))
    except:
        os.remove(os.path.join(tmpdir, "data"))
        os.remove(os.path.join(tmpdir, "model"))
        os.mkdir(os.path.join(tmpdir, "data"))
        os.mkdir(os.path.join(tmpdir, "model"))

    config_loc = os.path.join(tmpdir, "config.yml")
    with open(config_loc, "w") as f:
        f.write(yaml_string)

    return config_loc

@pytest.fixture
def input_src_file(tmpdir):
    src = "tests/resources/prepare_data_two.parquet"
    dest = os.path.join(tmpdir, "data", "src_confirmed.parquet")

    copyfile(src, dest)
    return dest

@pytest.fixture
def input_config_two(tmpdir):
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
      """)

    config_loc = os.path.join(tmpdir, "config.yml")
    with open(config_loc, "w") as f:
        f.write(yaml_string)

    return config_loc

@pytest.fixture
def input_src_file_two(tmpdir):
    src = "tests/resources/prepare_data_two.parquet"
    dest = os.path.join(tmpdir, "src_confirmed.parquet")

    copyfile(src, dest)
    return dest



def test_load_data(input_df, input_config, tmpdir):

    idm_factory(pipeline="load",
                config=input_config,
                run_mode="local",
                selected_metric="CONFIRMED",
                load_from="db",
                base_path=tmpdir)

    artefacts = os.listdir(os.path.join(tmpdir, "data"))
    artefact_to_assert = ['src_confirmed.parquet']

    for i in artefact_to_assert:
        assert i in artefacts

def test_forecasting_confirmed_cases(input_config, input_src_file, tmpdir):

    idm_factory(pipeline="forecast",
                config=input_config,
                run_mode="local",
                selected_metric="CONFIRMED",
                start_date="2020-04-01",
                load_from="local",
                base_path=tmpdir)

    artefacts = os.listdir(os.path.join(tmpdir, "data"))

    artefact_to_assert = ["src_confirmed.parquet", 'cumulative_confirmed_forecast.parquet']

    # check model and prediction both are generated
    for i in artefact_to_assert:
        assert i in artefacts

    forecast = pd.read_parquet(os.path.join(tmpdir, "data", "cumulative_confirmed_forecast.parquet"))

    # assert column names
    expected_columns = ['date_id', 'geo_id', 'cumulative_confirmed_curvefit_forecast',
            'cumulative_confirmed_curvefit_credible_interval_low', 'cumulative_confirmed_curvefit_credible_interval_high',
            'cumulative_confirmed']

    assert forecast.columns.tolist() == expected_columns

    # should producing forecast for next 30 days
    next_30_days = pd.date_range(start="2020-04-13", periods=30, freq="D").tolist()[-1]

    # forecast should not empty
    forecast_curvefit = forecast[forecast.index == next_30_days]["cumulative_confirmed_curvefit_forecast"]

    assert forecast_curvefit[0] > 0

def test_infer_rt(input_config, input_src_file, tmpdir):

    idm_factory(pipeline="infer_rt",
                config=input_config,
                run_mode="local",
                selected_metric="CONFIRMED",
                start_date="2020-04-01",
                load_from="local",
                base_path=tmpdir)

    artefacts = os.listdir(os.path.join(tmpdir, "data"))
    artefact_to_assert = ["config.yml", "src_confirmed.parquet", "confirmed_rt.parquet"]
    r_t = pd.read_parquet(os.path.join(tmpdir, "data", "confirmed_rt.parquet"))

    expected_columns = ["date_id", "geo_id", "r_t_most_likely", "r_t_ci_5", "r_t_ci_95", "case_growth_class"]

    assert r_t.columns.tolist() == expected_columns

def test_run_pipeline_without_basepath(input_config_two, input_src_file_two, tmpdir):

    idm_factory(pipeline="forecast",
                config=input_config_two,
                run_mode="local",
                selected_metric="CONFIRMED",
                start_date="2020-04-01",
                load_from="local")

    idm_factory(pipeline="infer_rt",
                config=input_config_two,
                run_mode="local",
                selected_metric="CONFIRMED",
                start_date="2020-04-01",
                load_from="local")

    artefacts = os.listdir(tmpdir)
    artefact_to_assert = ["config.yml", "src_confirmed.parquet", "confirmed_rt.parquet", "cumulative_confirmed_forecast.parquet"]

    # check model and prediction both are generated
    for i in artefact_to_assert:
        assert i in artefacts
