# Infectious Disease Modeling Platform

AI-powered infectious disease modeling tools for spread forecast, intervention model, non-pharmaceutical interventions (NPIs).

**Disclaimer:**
As we're not Epidemiologist thus methodology being used here are based on published literature and best practices.

# Dependencies

* Python >= 3.6.8
* Poetry >= 1.0.9
* Docker

# Building

```bash
$ make clean package
```

This will create the package file in `dist/idm-x.x.x-x.tar.gz`

# Testing

```bash
$ make docker-test
```

# Main component

`pipeline:` component for executing modeling. Currently pipeline that has been established:

* `forecast`: pipeline for generate forecast for chosen cumulative metric such as cumulative cases. We're using two method for generating forecast: 1). statistical curve-fitting method 2). bayesian modeling
* `infer_rt`: pipeline for generate R(t) or infection rate. We're followed the steps from Kevin Systrom's method, featured at rt.live, for inferring R(t).

`api`: RESTful api component use for consuming metric provided by the platform or play with the model such as SEIR model.

# Showcase

* `api` url: http://10.1.0.11:8000/v1/docs
* `dashboard` url: http://10.1.0.11:8501

#  Usage

## Running forecast pipeline

This command for launch `forecast` pipeline. You need to download data first using `Download data from db_disease_outbreak` command

```bash
$ python manage.py idm pipeline forecast conf/config.yml --src-from local
```

## Running infer_rt pipeline

This command for launch `infer_rt` pipeline. You need to download data first using `Download data from db_disease_outbreak` command

```bash
$ python manage.py idm pipeline infer_rt conf/config.yml --src-from local
```

## Running api

```bash
$ python manage.py api start conf/config.yml --port 8001
```

or using docker

```bash
$ WITH_DOCKER=1 bin/erk-idm start-api --port 8001
```

It will start the service using port 8001

## Configurations

To executing idm pipeline you need to provide parameter for the job written in config file. Here are the example configuration accepted by the application:

```yaml
---
ForecastModel:
  input:
    input_table: db_disease_outbreak
    date_col: date_id
    geoid_col: geo_id
    loc_granularity:
      - province
    metric_val_col: metric_val
    metric_granularity_col: metric_granularity
    metric_name_col: metric_name
    hour_id_col: hour_id
  output:
    output_path: /path/to/forecast/output
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
      model_path: /path/to/save/model 
      cores: 2
      chains: 2
      tune: 1000
      draws: 1000
      target_accept: 0.8
    curvefit:
      maxfev: 100
      forecast_days: 3
  with_train: true
  methods:
    - curvefit
    - bayesian
API:
  backend_cors_origin:
    - http://localhost:8000
    - http://10.1.0.11:8000
  forecast_file_path: "data/cumulative_confirmed_forecast.parquet"
  r_t_file_path: "data/confirmed_rt.parquet"
```

# Contributing

## Getting started

Make sure you are using python >= 3.6.8

Checking your python version

```bash
$ python --version
```

Install `poetry` via pip or others. Please check https://python-poetry.org/ for how to installing `poetry` for your machine

```
$ pip install poetry
```

Install all dependencies

```bash
$ poetry config --local virtualenvs.in-project true
$ poetry install
```

Initiate virtual environment created by `poetry`

```bash
$ poetry shell
```

## Collaboration is open

**Pull requests are welcome**. Please add your tests when you are adding new features and improvements. For major changes, please open an issue first to discuss what you would like to change.

Please put reviewers for your pull requests:

* Fitra Kacamarga (<fitra.19@gmail.com>)
