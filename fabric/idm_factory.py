import logging
import sys
import os
from datetime import timedelta

from fabric.utility import parse_file
from fabric.idm.models import ForecastModel, RTModel
from fabric.idm.utility import load_data as idm_load_data

from prefect import task, Flow, Parameter, unmapped
from prefect.engine.executors import DaskExecutor
import pandas as pd
import findspark

"""
Infectious Disease Modeling (IDM) Factory
-----------------------------------------

This code is meant for defining IDM pipeline. Currently pipeline that available:
    - load
    - forecast
    - infer_rt
"""

# needed for load_data method since currently using a Spark job
findspark.init()

@task(max_retries=3, retry_delay=timedelta(seconds=15))
def load_data(metadata, selected_metric, load_from="db"):
    """
    Loading data from Hive table, which one contains reported cases.

    Parameters
    ----------
    metadata : dict
        table metadata
    selected_metric : str
        metric to be forecasted
    load_from : str, optional
        load data from database or local file

    Returns
    -------
    cases : Pandas DataFrame
        DataFrame contains cumulative metrics
    """

    if load_from == "db":
        return idm_load_data(metadata, selected_metric)
    elif load_from == "local":
        data_path = os.path.join(metadata["output"]["output_path"], f"src_{selected_metric.lower()}.parquet")
        loaded_data = pd.read_parquet(data_path)

        return loaded_data

@task
def parse_config(config, base_path):
    """
    Parse config file to dict

    Parameters
    ----------
    config : str
        config file path
    base_path : str
        base_path application

    Returns
    -------
    vars : dict
    """

    config_vars = parse_file(config)
    if base_path is not None:
        config_vars["output"]["output_path"] = os.path.join(base_path, config_vars["output"]["output_path"])
        if "bayesian" in config_vars["model_params"]:
            config_vars["model_params"]["bayesian"]["model_path"] = os.path.join(base_path, config_vars["model_params"]["bayesian"]["model_path"])

    return config_vars

@task
def get_forecast_methods(config):
    """
    Get forcast method in config dict

    Parameters
    ----------
    config : dict
        config vars

    Returns
    -------
    methods : list
        list of forecast methods
    """

    return config["methods"]

@task
def load_forecast_model(metadata, metric, start_date, minimum_cases):
    """
    Initiate forecast model

    Parameters
    ----------
    metadata : dict
        config vars
    metric : str
        target timeseries metric to be forecasted
    start_date : str, "yyyy-MM-dd" format
        starting date filter
    minimum_cases : int
        minimal number of cases

    Returns
    -------
    forecast_model : ForecastModel object
        forecast model object
    """

    date_col = metadata["input"]["date_col"]
    geoid_col = "geo_id"

    model = ForecastModel(date_col=date_col,
                         geoid_col=geoid_col,
                         start_date=start_date,
                         min_cases=minimum_cases,
                         metric=metric)

    return model

@task
def print_data(data):
    """
    Dummy function, just for debugging

    Parameters
    ----------
    data : DataFrame
    """

    print(data.head())

@task
def forecasting(method, model, data, hyper_params):
    """
    Generate forecast data based on chosen method

    Parameters
    ----------
    method : str
        chosen method for generate forecast
    model : ForecastModel object
        forecast model object
    data : Pandas DataFrame
        data about cumulative cases
    hyper_params : dict
        model hyper parameters

    Returns
    -------
    data with forecast : list of Pandas DataFrame
        list of DataFrame with forecast result
    """

    if method == "curvefit":
        curvefit_params = hyper_params["model_params"]["curvefit"]
        return model.curvefitting_forecast(data, curvefit_params)
    elif method == "bayesian":
        bayesian_params = hyper_params["model_params"]["bayesian"]
        with_train = hyper_params["with_train"]
        model_path = bayesian_params["model_path"]
        if with_train:
            model.bayesian_learning(data, bayesian_params, model_path)

        return model.bayesian_forecast(data, bayesian_params, model_path)

@task
def combine_forecast(model, data):
    """
    Combining forecast data to one

    Parameters
    ----------
    model : class object
        Forecast model object
    data : array-like
        Array of DataFrame

    Returns
    -------
    forecast : pd.DataFrame
        combined forecast
    """

    return model.combine_forecast(data)

@task
def save_data(data, config, metric, app="forecast"):
    """
    Saving data

    Parameters
    ----------
    data : Pandas DataFrame
        forecast data
    config : dict
        config variables
    metric : str
    app : str
        name of idm task
    """

    save_path = config["output"]["output_path"]
    if app is None:
        filename = f"{metric}.parquet"
    else:
        filename = f"{metric}_{app}.parquet"

    full_save_path = os.path.join(save_path, filename)
    data.to_parquet(full_save_path)

@task
def compute_rt(data, config, start_date, minimum_cases):
    """
    Running r_t inference

    Parameters
    ----------
    data : pd.DataFrame
        timeseries data about covid19 cases
    config : dict
        config vars
    start_date : str with "yyyy-MM-dd" format
        starting date filter
    minimum_cases : int
        minimal number of cases

    Returns
    -------
    r_t : pd.DataFrame
        predicted infection rate (r_t) for given location and time
    """
    rt_model = RTModel(start_date=start_date,
                       min_cases=minimum_cases)

    return rt_model.run(data)

def main(pipeline, config, run_mode="local", start_date=None, selected_metric="CONFIRMED", minimum_cases=None, load_from="db", base_path=None):
    """
    Main function

    Parameters
    ----------
    pipeline : str
        what pipeline that chosen
    config : str
        configuration file being setup
    run_mode : str, optional
        run mode currently local or agent. Local mode the pipeline initiate by local machine. Agent mode the pipeline will initiate by prefect scheduler.
    start_date : str, optional
        if defined, will only selecting date greater than this
    selected_metric : str, optional
        target column name
    minimum_cases : int, optional
    load_from : str, optional
    base_path : str, optional
    """

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(__name__)

    if pipeline == "load":
        pipeline_params = {
                            "config_file": config,
                            "metric": selected_metric,
                            "load_from": load_from,
                            "base_path": base_path
                          }

        with Flow("Loading cases") as flow:
            config_file = Parameter("config_file", default="conf/config.yml")
            metric = Parameter("metric", default="CONFIRMED")
            load_from = Parameter("load_from", default="db")
            base_path = Parameter("base_path", default=None)

            # parse config
            metadata = parse_config(config_file, base_path)

            # load data
            df = load_data(metadata, metric, load_from)

            save_data(df, metadata, f"src_{metric.run().lower()}", None)

    elif pipeline == "forecast":
        pipeline_params = {
                            "config_file": config,
                            "metric": selected_metric,
                            "load_from": load_from,
                            "start_date": start_date,
                            "minimum_cases": minimum_cases,
                            "base_path": base_path
                          }


        with Flow("Forecasting") as flow:
            config_file = Parameter("config_file", default="conf/config.yml")
            metric = Parameter("metric", default="CONFIRMED")
            load_from = Parameter("load_from", default="local")
            start_date = Parameter("start_date", default=None)
            minimum_cases = Parameter("minimum_cases", default=None)
            base_path = Parameter("base_path", default=None)


            # parse config
            metadata = parse_config(config_file, base_path)

            # Load data
            df = load_data(metadata, metric, load_from)

            # initiate forecast model
            model = load_forecast_model(metadata, metric.run().lower(), start_date, minimum_cases)

            # getting forecast methods
            methods = get_forecast_methods(metadata)

            # Forecasting
            list_forecast = forecasting.map(method=methods,
                                            model=unmapped(model),
                                            data=unmapped(df),
                                            hyper_params=unmapped(metadata))

            # Combine forecast results
            forecast = combine_forecast(model, list_forecast)

            # print
            print_data(forecast)

            # save forecast result
            save_data(forecast, metadata, f"cumulative_{metric.run().lower()}")

    elif pipeline == "infer_rt":

        pipeline_params = {
                            "config_file": config,
                            "metric": selected_metric,
                            "load_from": load_from,
                            "start_date": start_date,
                            "minimum_cases": minimum_cases,
                            "base_path": base_path
                          }

        with Flow("Infer R_t") as flow:
            config_file = Parameter("config_file", default="conf/config.yml")
            metric = Parameter("metric", default="CONFIRMED")
            load_from = Parameter("load_from", default="local")
            start_date = Parameter("start_date", default=None)
            minimum_cases = Parameter("minimum_cases", default=None)
            base_path = Parameter("base_path", default=None)

            # parse config
            metadata = parse_config(config_file, base_path)

            # Load data
            df = load_data(metadata, metric, load_from)

            # compute rt
            r_t = compute_rt(df, metadata, start_date, minimum_cases)

            # print
            print_data(r_t)

            # save
            save_data(r_t, metadata, f"{metric.run().lower()}", "rt")

    if run_mode == "local":
        flow.run(executor=DaskExecutor(debug=True), **pipeline_params)

    elif run_mode == "agent":
        flow.register()
        logger.info("Please start pipeline from prefect UI or cli")
