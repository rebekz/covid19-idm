import pdb
import os
import pytest
import textwrap
import yaml

import pandas as pd
from fabric.idm.models import ForecastModel

GEO_ID_COL = "geo_id"
METRIC_GRANULARITY_COL = "metric_granularity"
METRIC_NAME_COL = "metric_name"
DATE_COL = "date_id"
HOUR_ID_COL = "hour_id"
METRIC_VALUE_COL = "metric_value"

@pytest.fixture
def input_data(tmpdir):
    input_data = pd.read_parquet("tests/resources/prepare_data_two.parquet")
    yaml_string = textwrap.dedent(f"""\
    ---
    ForecastModel:
      input:
        input_table: db_disease_outbreak
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
      with_train: true""")

    yaml_dict = yaml.safe_load(yaml_string)["ForecastModel"]

    return input_data, yaml_dict

def test_curvefit_forecast(input_data, tmpdir):

    data = input_data[0]
    config = input_data[1]

    # Input data params
    date_col = config["input"]["date_col"]
    geoid_col = "geo_id"

    model_params = config["model_params"]["curvefit"]

    forecast_model = ForecastModel(date_col=date_col,
                                    geoid_col=geoid_col,
                                    start_date="2020-04-01")

    curvefit_result = forecast_model.curvefitting_forecast(data, model_params)

    expected_columns = ['date_id', 'geo_id', 'cumulative_confirmed_curvefit_forecast',
            'cumulative_confirmed_curvefit_credible_interval_low', 'cumulative_confirmed_curvefit_credible_interval_high',
            'cumulative_confirmed']

    assert curvefit_result.columns.tolist() == expected_columns

    # should producing forecast for next 30 days
    next_30_days = pd.date_range(start="2020-04-13", periods=30, freq="D").tolist()[-1]

    # forecast should not empty
    forecast_curvefit = curvefit_result.loc[curvefit_result.index == next_30_days, "cumulative_confirmed_curvefit_forecast"]

    assert forecast_curvefit[1] > 0

def test_bayesian_forecast(input_data, tmpdir):

    data = input_data[0]
    config = input_data[1]

    # Input data params
    date_col = config["input"]["date_col"]
    geoid_col = "geo_id"

    model_params = config["model_params"]["bayesian"]

    forecast_model = ForecastModel(date_col=date_col,
                                   geoid_col=geoid_col,
                                   start_date="2020-04-01")

    forecast_model.bayesian_learning(data, model_params, tmpdir)
    bayesian_result = forecast_model.bayesian_forecast(data, model_params, tmpdir)

    expected_columns = ['date_id', 'geo_id', 'cumulative_confirmed_bayesian_forecast',
            'cumulative_confirmed_bayesian_credible_interval_low', 'cumulative_confirmed_bayesian_credible_interval_high', "cumulative_confirmed"]

    assert bayesian_result.columns.tolist() == expected_columns
    # should producing forecast for next 30 days
    next_30_days = pd.date_range(start="2020-04-13", periods=30, freq="D").tolist()[-1]

    # forecast should not empty
    forecast_bayesian = bayesian_result.loc[bayesian_result.index == next_30_days, "cumulative_confirmed_bayesian_forecast"]

    assert forecast_bayesian[1] > 0

def test_combine_forecast(input_data, tmpdir):

    data = input_data[0]
    config = input_data[1]

    # Input data params
    date_col = config["input"]["date_col"]
    geoid_col = "geo_id"

    model_params_bayesian = config["model_params"]["bayesian"]
    model_params_curvefit = config["model_params"]["curvefit"]

    forecast_model = ForecastModel(date_col=date_col,
                                   geoid_col=geoid_col,
                                   start_date="2020-04-01")

    forecast_model.bayesian_learning(data, model_params_bayesian, tmpdir)
    bayesian_result = forecast_model.bayesian_forecast(data, model_params_bayesian, tmpdir)
    curvefit_result = forecast_model.curvefitting_forecast(data, model_params_curvefit)

    forecast = forecast_model.combine_forecast([curvefit_result, bayesian_result])

    expected_columns = ['date_id', 'geo_id', 'cumulative_confirmed_curvefit_forecast',
            'cumulative_confirmed_curvefit_credible_interval_low', 'cumulative_confirmed_curvefit_credible_interval_high',
            'cumulative_confirmed', \
            'cumulative_confirmed_bayesian_forecast', 'cumulative_confirmed_bayesian_credible_interval_low', \
            'cumulative_confirmed_bayesian_credible_interval_high']

    assert forecast.columns.tolist() == expected_columns

def test_prepare_data(input_data):

    data = input_data[0]
    config = input_data[1]

    # Input data params
    date_col = config["input"]["date_col"]
    geoid_col = "geo_id"

    forecast_model = ForecastModel(date_col, geoid_col)

    result = forecast_model.prepare_data(data)
    result_df = result[0]

    expected_columns = ['geo_id', "confirmed", "increment", 'cumulative_confirmed', 'x_cases', 'days_since_x_cases']

    assert result_df.columns.tolist() == expected_columns
