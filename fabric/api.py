from datetime import datetime
import logging
import os
import yaml
from typing import List, Optional
from enum import Enum

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from pydantic import BaseModel

from fabric.utility import parse_file
from fabric.idm.models import SEIRModel


def main(config_vars):
    """
    Starting api service

    Parameters
    ----------
    config_vars : dict
        api settings

    Returns
    -------
    app : fastapi class
    """

    v1_path = "/v1"

    app = FastAPI(
            title="Forecast Services API",
            description="This API provide forecast services",
            version="0.1.0",
            openapi_url=f"{v1_path}/openapi.json",
            docs_url=f"{v1_path}/docs",
            redoc_url=f"{v1_path}/redoc"
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config_vars["backend_cors_origin"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    """
    Forecast part
    """

    class ForecastCases(BaseModel):
        date_id: datetime
        forecast: Optional[float]
        actual: Optional[float]
        credible_interval_low: Optional[float]
        credible_interval_high: Optional[float]

        def __init__(self, **data) -> None:
            """Custom init to parse datetime64"""
            if isinstance(data["date_id"], pd.Timestamp):
                data["date_id"] = data["date_id"].to_pydatetime()
            super().__init__(**data)

    class ForecastContent(BaseModel):
        geo_id: str
        metric: str
        method: str
        series: List[ForecastCases]

    class ForecastOut(BaseModel):
        result: Optional[ForecastContent]
        class Config:
            arbitrary_types_allowed = True

    class ForecastMetric(str, Enum):
        confirmed = "cumulative_confirmed"

    class ForecastMethod(str, Enum):
        curvefit = "curvefit"
        bayesian = "bayesian"

    class RequestForecastMetric(BaseModel):
        geo_id: str
        metric: ForecastMetric
        method: ForecastMethod
        time_from: datetime = datetime(2019, 12, 1)
        time_to: datetime = datetime(2022, 1, 1)

    """
    SEIR Part
    """

    class SEIRParam(BaseModel):
        initN: int = 1000000
        beta: float = 0.2
        incubPeriod: float = 5.2
        durInf: float = 16.39
        xi: float = 0.0
        mu_I: float = 0.01
        mu_0: float = 0.0
        nu: float = 0.0
        p: float = 0.0
        beta_D: float = None
        sigma_D: float = None
        gamma_D: float = None
        mu_D: float = None
        theta_E: float = 0.0
        theta_I: float = 0.02
        psi_E: float = 1.0
        psi_I: float = 1.0
        q: float = 0.0
        initE: int = 0.0
        initI: int = 100
        initD_E: int = 0
        initD_I: int = 0
        initR: int = 0
        initF: int = 0

    class ValueAndDesc(BaseModel):
        value: Optional[float]
        description: str

    class SEIRInitParam(BaseModel):
        beta: ValueAndDesc
        incubPeriod: ValueAndDesc
        durInf: ValueAndDesc
        xi: ValueAndDesc
        mu_I: ValueAndDesc
        mu_0: ValueAndDesc
        nu: ValueAndDesc
        p: ValueAndDesc
        beta_D: ValueAndDesc
        sigma_D: ValueAndDesc
        gamma_D: ValueAndDesc
        mu_D: ValueAndDesc
        theta_E: ValueAndDesc
        theta_I: ValueAndDesc
        psi_E: ValueAndDesc
        psi_I: ValueAndDesc
        q: ValueAndDesc

    class RequestSEIRMetric(BaseModel):
        params: Optional[SEIRParam]
        start_date: Optional[str] = "2020-02-01"
        simulation_days: Optional[int] = 200
        interventions: Optional[dict] = None

    class SEIRResult(BaseModel):
        date_id: datetime
        S: np.float64
        E: np.float64
        I: np.float64
        D_E: np.float64
        D_I: np.float64
        R: np.float64
        F: np.float64

        def __init__(self, **data) -> None:
            if isinstance(data["date_id"], pd.Timestamp):
                data["date_id"] = data["date_id"].to_pydatetime()
            super().__init__(**data)

    class SEIROut(BaseModel):
        result: List[SEIRResult]
        init_params: SEIRInitParam

    """
    R_t part
    """

    class RtCases(BaseModel):
        date_id: datetime
        value: float
        credible_interval_low: float
        credible_interval_high: float

        def __init__(self, **data) -> None:
            """Custom init to parse datetime64"""
            if isinstance(data["date_id"], pd.Timestamp):
                data["date_id"] = data["date_id"].to_pydatetime()
            super().__init__(**data)

    class RtContent(BaseModel):
        geo_id: str
        case_growth_status: str
        case_growth_description: str
        current_infection_rate: float
        metric: str
        series: List[RtCases]

    class RtOut(BaseModel):
        result: Optional[RtContent]
        class Config:
            arbitrary_types_allowed = True

    class RequestRt(BaseModel):
        geo_id: str
        time_from: datetime = datetime(2019, 12, 1)
        time_to: datetime = datetime(2020, 1, 1)

    @app.get(f"{v1_path}/forecast/geo_id")
    async def get_list_geo_id():
        forecast_file_path = os.getenv("FORECAST_PATH", config_vars["forecast_file_path"])
        forecast = pd.read_parquet(forecast_file_path)
        return forecast["geo_id"].unique().tolist()

    @app.post(v1_path + "/forecast/getMetric")
    async def get_forecast(
            request: RequestForecastMetric
    ) -> ForecastOut:
        if request.metric == ForecastMetric.confirmed:
            forecast_file_path = os.getenv("FORECAST_PATH", config_vars["forecast_file_path"])
            forecast = pd.read_parquet(forecast_file_path)

            df = forecast.copy()
            df = df[df["geo_id"] == request.geo_id]
            df = df[(df.index >= request.time_from) & (df.index <= request.time_to)]
            df = df.where(pd.notnull(df), None)  # Convert null to None so it encode properly
            df = df.sort_index()

            forecast_result = None
            if df.empty is False:
                if request.method == ForecastMethod.curvefit:
                    forecast_result = ForecastContent(
                        geo_id=request.geo_id,
                        metric=ForecastMetric.confirmed,
                        method=ForecastMethod.curvefit,
                        series=list(map(lambda x: ForecastCases(date_id=x["date_id"], \
                                forecast=x["cumulative_confirmed_curvefit_forecast"], \
                                actual=x["cumulative_confirmed"], \
                                credible_interval_low=x["cumulative_confirmed_curvefit_credible_interval_low"], \
                                credible_interval_high=x["cumulative_confirmed_curvefit_credible_interval_high"] \
                            ), df.to_dict(orient="records"))),
                    )

                elif request.method == ForecastMethod.bayesian:
                    forecast_result = ForecastContent(
                        geo_id=request.geo_id,
                        metric=ForecastMetric.confirmed,
                        method=ForecastMethod.bayesian,
                        series=list(map(lambda x: ForecastCases(date_id=x["date_id"], \
                                forecast=x["cumulative_confirmed_bayesian_forecast"], \
                                actual=x["cumulative_confirmed"], \
                                credible_interval_low=x["cumulative_confirmed_bayesian_credible_interval_low"], \
                                credible_interval_high=x["cumulative_confirmed_bayesian_credible_interval_high"] \
                            ), df.to_dict(orient="records"))),
                    )

            return ForecastOut(result=forecast_result)
        else:
            return ForecastOut(result=None)

    @app.post(v1_path + "/seir/getMetric")
    async def get_seir(
            request: RequestSEIRMetric
    ) -> SEIROut:

        if request.params is not None:
            seir_model = SEIRModel(**vars(request.params))
        else:
            seir_default_params = SEIRParam()
            seir_model = SEIRModel(**seir_default_params.dict())

        seir_result, init_params = seir_model.run(start_date=request.start_date, t=request.simulation_days, checkpoints=request.interventions)

        return SEIROut(
                result=list(map(lambda x: SEIRResult(**x), seir_result.to_dict(orient="records"))),
                init_params=SEIRInitParam(**init_params)
                )

    @app.post(v1_path + "/infection_rate/getMetric")
    async def get_rt(
            request: RequestRt
            ) -> RtOut:
        rt_file_path = os.getenv("RT_PATH", config_vars["rt_file_path"])
        r_t = pd.read_parquet(rt_file_path)

        rt_df = r_t.copy()
        rt_df = rt_df.loc[rt_df["geo_id"] == request.geo_id, :]
        rt_df = rt_df.loc[(rt_df.index >= request.time_from) & (rt_df.index <= request.time_to), :]
        rt_df = rt_df.where(pd.notnull(rt_df), None)  # Convert null to None so it encode properly
        rt_df = rt_df.sort_index()

        rt_result = None
        if rt_df.empty is False:

            last_rt = rt_df.iloc[-1]
            case_growth_status = last_rt.case_growth_class
            current_rt = last_rt.r_t_most_likely
            case_growth_description = f"On average, each person with COVID is infecting {current_rt} other people."

            rt_result = RtContent(
                    geo_id=request.geo_id,
                    case_growth_status=case_growth_status,
                    case_growth_description=case_growth_description,
                    current_infection_rate=current_rt,
                    metric="Infection rate",
                    series=list(map(lambda x: RtCases(date_id=x["date_id"], \
                            value=x["r_t_most_likely"], \
                            credible_interval_low=x["r_t_ci_5"], \
                            credible_interval_high=x["r_t_ci_95"] \
                        ), rt_df.to_dict(orient="records"))),
                    )

        return RtOut(result=rt_result)

    return app
