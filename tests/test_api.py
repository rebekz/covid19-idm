import os
import pytest
import textwrap
import yaml

from fabric.api import main
import pandas as pd
from fastapi.testclient import TestClient

@pytest.fixture(scope='session', autouse=True)
def init_api(request):
    backend_cors_origin = ["http://localhost:8000"]
    mock_forecast_file = os.path.join(str(request.config.rootdir), "tests/resources", "cumulative_confirmed_forecast.parquet")
    mock_rt_file = os.path.join(str(request.config.rootdir), "tests/resources", "confirmed_rt.parquet")

    config = {"backend_cors_origin": backend_cors_origin, "forecast_file_path": mock_forecast_file, "rt_file_path": mock_rt_file}
    app = main(config)

    return app

def test_forecast_api(init_api):

    client = TestClient(init_api)
    # get list of geo-id
    response = client.get("/v1/forecast/geo_id")
    assert response.status_code == 200
    assert response.json() == ["BANTEN", "JAKARTA"]

    # test query Jakarta forecast
    response = client.post("/v1/forecast/getMetric",
            json={"geo_id": "JAKARTA", "metric": "cumulative_confirmed", "method": "curvefit", "time_from": "2020-04-01T00:00:00", "time_to": "2020-04-30T00:00:00"}
            )

    forecast_df = pd.DataFrame(response.json()["result"]["series"])

    # check dataframe is not empty
    assert forecast_df.empty == False

    # should producing forecast at this time
    assert forecast_df[forecast_df.date_id == "2020-04-30T00:00:00"]["forecast"].values[0] > 0

def test_seir_api(init_api):

    client = TestClient(init_api)
    response = client.post("v1/seir/getMetric",
            json={"params": {
                "initN":1000000,
                "beta": 0.14,
                "incubPeriod": 5.2,
                "durInf": 11,
                "initI": 100
                },
                "start_date": "2020-02-01",
                "simulation_days": 200,
                "interventions": {
                    "t": ["2020-04-01", "2020-05-01"],
                    "beta": [0.1, 0.2]
                    }
                })
    seir_result = pd.DataFrame(response.json()["result"])
    init_params = response.json()["init_params"]

    expected_columns = ["date_id", "S", "E", "I", "D_E", "D_I", "R", "F"]

    assert seir_result.columns.tolist() == expected_columns
    assert len(seir_result.index.tolist()) == 201
    assert init_params is not None

def test_seir_api_no_params(init_api):

    client = TestClient(init_api)
    response = client.post("v1/seir/getMetric", json={})
    seir_result = pd.DataFrame(response.json()["result"])
    init_params = response.json()["init_params"]

    expected_columns = ["date_id", "S", "E", "I", "D_E", "D_I", "R", "F"]

    assert seir_result.columns.tolist() == expected_columns
    assert len(seir_result.index.tolist()) == 201
    assert init_params is not None

def test_rt_api(init_api):

    client = TestClient(init_api)

    # test query Jakarta forecast
    response = client.post("/v1/infection_rate/getMetric",
            json={"geo_id": "BALI", "time_from": "2020-01-01T00:00:00", "time_to": "2020-05-01T00:00:00"}
            )

    r_t = pd.DataFrame(response.json()["result"]["series"])
    geo_id = response.json()["result"]["geo_id"]

    assert r_t.empty == False
    assert geo_id == "BALI"
