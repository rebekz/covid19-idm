"""
forecast model using methods:
    * parametric curve-fitting
    * bayesian modeling
"""
import logging
import sys
import os
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import lmfit
from functools import reduce

class ForecastConstant:
    cumulative = "cumulative"


class ForecastModel:
    """
    Forecasting Model

    Parameters
    ----------
    date_col : str
        date column name
    geoid_col : str
        identifier for location of reported cases
    looger : logging object, optional
        logging object, if not available it will initiate one.
    """

    def __init__(self,
                 date_col,
                 geoid_col,
                 start_date=None,
                 min_cases=None,
                 metric="confirmed",
                 logger=None):

        self.date_col = date_col
        self.geoid_col = geoid_col
        self.start_date = start_date
        self.min_cases = min_cases
        self.metric = metric
        self.cumulative_metric = f"{ForecastConstant.cumulative}_{self.metric}"

        if logger is None:
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    def prepare_data(self, data):
        """
        Preparing data for forecasting. Create new columns:
            - `days_since_x_cases` which are telling you number of day after the day of the cases reach number X cases.
            - `x_cases` the date when the case is reach `x` number

        Parameters
        ----------
        data : Pandas DataFrame
            input data

        Returns
        -------
        data : Pandas DataFrame
            transformed DataFrame
        list_geo : list
            list of available `geo_id` from the data
        n_geo : int
            number of `geo_id` from the data
        idx_geo : array-like
            array of index number of data for each `geo_id`
         geo_first_dates : dict
            first date recorded for each `geo_id`
        """

        data[f"{ForecastConstant.cumulative}_{self.metric}"] = data[self.metric]

        if self.start_date is not None:
            data = data.loc[data.index >= self.start_date, :]
        if self.min_cases is not None:
            data = data.loc[data[self.metric] >= self.min_cases, :]

        list_geo = data[self.geoid_col].unique()
        n_geo = len(list_geo)

        idx_geo = pd.Index(list_geo).get_indexer(data[self.geoid_col])
        geo_first_dates = {c: data[data[self.geoid_col] == c].index.min() for c in list_geo}
        data.loc[:, "x_cases"] = data.apply(lambda x: geo_first_dates[x[self.geoid_col]], axis=1)
        data.loc[:, "days_since_x_cases"] = (data.index - data["x_cases"]).apply(lambda x: x.days)

        return data, list_geo, n_geo, idx_geo, geo_first_dates

    def print_data(self, data):
        """
        dummy function
        """

        print(data.head())

    def logistic(self, K, r, t, C_0):
        """
        Logistic function used for model the Bayesian forecast method

        Parameters
        ----------
        K : int
            capacity
        r : float
            growth rate
        t : int
            time
        C_0 : int
            initial number of cases at time x

        Returns
        -------
        number : float
        """

        A = (K-C_0)/C_0
        return K / (1 + A * np.exp(-r * t))

    def bayesian_model(self, x, y, index, n, model_params):
        """
        Bayesian model

        Parameters
        ----------
        x : array-like
            time
        y : array-like
            observed number at time t
        index : array-lke
            geo_id index in observed data
        n : int
            number of observations
        model_params : dict
            model hyper parameters
        """

        c0_mu = model_params["c0_mu"]
        c0_sigma = model_params["c0_sigma"]
        r_mu = model_params["r_mu"]
        r_mu_sigma = model_params["r_mu_sigma"]
        r_sigma = model_params["r_sigma"]
        k_mu = model_params["k_mu"]
        k_mu_sigma = model_params["k_mu_sigma"]
        k_sigma = model_params["k_sigma"]

        model = pm.Model()
        with model:
            BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
            t = pm.Data("x_data", x)
            confirmed_cases = pm.Data("y_data", y)

            # Intercept
            C_0 = pm.Normal("C_0", mu=c0_mu, sigma=c0_sigma)

            # Growth rate
            r_mu = pm.Normal("r_mu", mu=r_mu, sigma=r_mu_sigma)
            r_sigma = pm.HalfNormal("r_sigma", r_sigma)
            r = BoundedNormal("r", mu=r_mu, sigma=r_sigma, shape=n)

            # Total number of cases
            K_mu = pm.Normal("K_mu", mu=k_mu, sigma=k_mu_sigma)
            K_sigma = pm.HalfNormal("K_sigma", k_sigma)
            K = pm.Gamma("K", mu=K_mu, sigma=K_sigma, shape=n)

            # Logistic regression
            growth = self.logistic(K[index], r[index], t, C_0)

            # Likelihood error
            eps = pm.HalfNormal("eps")

            # Likelihood
            pm.Lognormal("cases", mu=np.log(growth), sigma=eps, observed=confirmed_cases)

        return model

    def bayesian_learning(self, data, model_params, save_path="model"):
        """
        Forecasting using Bayesian model.
        Learning priors from all locations about the cases

        Parameters
        ----------
        data : Pandas DataFrame
            observation
        model_params : dict
            model parameters
        save_path : str
            path to save learned model
        """

        self.logger.info("Creating bayesian model")

        cases = self.prepare_data(data[[self.geoid_col, self.metric]].dropna())
        cases_data = cases[0]
        n_geo = cases[2]
        idx_geo = cases[3]
        draws = model_params["draws"]
        tune = model_params["tune"]
        chains = model_params["chains"]
        cores = model_params["cores"]
        target_accept = model_params["target_accept"]

        # Sampling prior cases from all locations
        train_model = self.bayesian_model(
            x=cases_data["days_since_x_cases"],
            y=cases_data[self.cumulative_metric],
            index=idx_geo,
            n=n_geo,
            model_params=model_params
        )
        self.logger.info("Training...")
        with train_model:
            trace = pm.sample(draws=draws, tune=tune, chains=chains,
                              cores=cores, target_accept=target_accept)

        # save trace
        self.logger.info("Saving model")
        model_save_path = os.path.join(save_path, f"{self.cumulative_metric}_bayesian_forecast.trace")
        pm.save_trace(trace, model_save_path, overwrite=True)

    def bayesian_forecast(self, data, model_params, model_path="model"):
        """
        Forecasting using Bayesian model.
        Inferencing for all locations using posterior distribution

        Parameters
        ----------
        data : Pandas DataFrame
            observation
         : str
             column table forecasted
        model_params : dict
            model parameters
        model_path : str
            location for learned model

        Returns
        -------
        forecast : Pandas DataFrame
            forecast result
        """

        self.logger.info("Using bayesian model")

        cases = self.prepare_data(data[[self.geoid_col, self.metric]])
        cases_data = cases[0]
        list_geo = cases[1]
        n_geo = cases[2]
        geo_first_dates = cases[4]

        n_days = model_params["forecast_days"]

        # Construct new vector with forecast target
        time_index = np.arange(0, n_days, 1)
        time_index = np.repeat(time_index, n_geo)

        # Construct geo vector
        geo_index = np.arange(n_geo)
        geo_index = np.tile(geo_index, n_days)
        dummy_y = np.zeros(len(time_index))

        # Generate the inference model
        inference_model = self.bayesian_model(x=time_index, y=dummy_y, index=geo_index, n=n_geo, model_params=model_params)

        # Sampling from posterior
        self.logger.info("Inferencing...")

        # load model
        model_path = os.path.join(model_path, f"{self.cumulative_metric}_bayesian_forecast.trace")

        with inference_model:
            trace = pm.load_trace(model_path)
            posterior = pm.sample_posterior_predictive(trace)

        # Calculate credible interval
        credible_interval = az.hdi(
            posterior["cases"], hdi_prob=.95
        )

        # Calculate dates
        start = [geo_first_dates[x] for x in list_geo[geo_index].tolist()]
        offset = [pd.DateOffset(x) for x in time_index]
        dates = list(
           map(lambda x: (x[0] + x[1]).to_pydatetime(), zip(start, offset))
        )

        # Create result dataframe
        forecast = pd.DataFrame(
            {
                self.date_col: dates,
                self.geoid_col: list_geo[geo_index],
                f"{self.cumulative_metric}_bayesian_forecast": np.mean(posterior["cases"], axis=0),
                f"{self.cumulative_metric}_bayesian_credible_interval_low": credible_interval[:, 0],
                f"{self.cumulative_metric}_bayesian_credible_interval_high": credible_interval[:, 1]
            },
            index=dates,
        ).rename_axis("index")

        # Merge with ground truth
        forecast = pd.merge(
            forecast.rename_axis("index").reset_index(),
            cases_data[[self.geoid_col, self.cumulative_metric]].rename_axis("index").reset_index(),
            on=["index", self.geoid_col],
            how="outer"
        ).set_index("index")

        return forecast

    def curve_function(self, x, c, k, m):
        """
        Curve function. We're model the distribution of cumulative is follow logistic function.

        Parameters
        ----------
        x : array-like
            time
        c : float
        k : float
        m : float
        """

        y = c / (1 + np.exp(-k*(x-m))) # noqa
        return y

    def curvefitting_forecast(self, data, model_params):
        """
        Forecasting by curve-fitting the cases

        Parameters
        ----------
        data : Pandas DataFrame
            observation
        target : str
            column to be forecasted
        model_params : dict
            model parameters

        Returns
        -------
        forecast : Pandas DataFrame
            forecast result
        """

        cases = self.prepare_data(data[[self.geoid_col, self.metric]].dropna())
        cases_data = cases[0]
        list_geo = cases[1]

        fit_result = []
        self.logger.info("Creating Curve-fitting model")
        maxfev = model_params["maxfev"]
        n_days = model_params["forecast_days"]

        # Curve-fitting for all locations
        for geo in list_geo:
            self.logger.info("Fitting %s", geo)
            df = cases_data.loc[cases_data[self.geoid_col] == geo, :] # noqa

            x = df.loc[:, "days_since_x_cases"].values # noqa
            y = df.loc[:, self.cumulative_metric].values # noqa

            # set capacity is today's number
            p_0 = [np.max(y), 1, 1]

            # Here we are assume the distribution is follow logistic function
            model = lmfit.Model(self.curve_function)
            model.set_param_hint("c", value=p_0[0])
            model.set_param_hint("k", value=p_0[1])
            model.set_param_hint("m", value=p_0[2])
            params = model.make_params()

            # fitting the curve
            curvefit = model.fit(y, params, method="leastsq", x=x, max_nfev=maxfev)

            # Forecast for next days
            max_days = np.max(x)
            forecast_days = max_days + n_days
            forecast_time_index = np.arange(max_days + 1, forecast_days, 1)
            time_index = np.concatenate([x, forecast_time_index])
            fitted = curvefit.best_fit
            best_values = curvefit.best_values
            preds = self.curve_function(forecast_time_index,
                                        best_values["c"], best_values["k"], best_values["m"])

            temp = df[[self.cumulative_metric]]
            temp.loc[:, ("curvefit_forecast")] = fitted
            index = pd.date_range(start=df.index[-1], periods=n_days, freq="D")
            index = index[1:]
            temp = temp.append(pd.DataFrame(data=preds, index=index, columns=["curvefit_forecast"]))

            # getting 95% confidence interval
            dely = curvefit.eval_uncertainty(x=time_index, sigma=2)

            # checking are we getting sensible confidence interval
            first_dely = dely[0]
            first_forecast = temp.iloc[0]["curvefit_forecast"]

            # if first value of low confidence is non-negative then add the records, otherwise set all NaN
            if (first_forecast - first_dely) > 0:
                temp.loc[:, "credible_interval_high"] = temp.loc[:, "curvefit_forecast"] + dely
                temp.loc[:, "credible_interval_low"] = temp.loc[:, "curvefit_forecast"] - dely
            else:
                temp.loc[:, "credible_interval_high"] = np.NaN
                temp.loc[:, "credible_interval_low"] = np.NaN

            dates = temp.index.to_pydatetime()

            # Create result dataframe
            forecast = pd.DataFrame(
                {
                    self.date_col: dates,
                    self.geoid_col: geo,
                    f"{self.cumulative_metric}_curvefit_forecast": \
                        temp.loc[:, "curvefit_forecast"].values,
                    f"{self.cumulative_metric}_curvefit_credible_interval_low": \
                        temp.loc[:, "credible_interval_low"],
                    f"{self.cumulative_metric}_curvefit_credible_interval_high":  \
                        temp.loc[:, "credible_interval_high"],
                },
                index=dates
                )

            # Merge with ground truth
            forecast = pd.merge(
                forecast.rename_axis("index").reset_index(),
                df[[self.geoid_col, self.cumulative_metric]].rename_axis("index").reset_index(),
                on=["index", self.geoid_col],
                how="outer"
            ).set_index("index")

            fit_result.append(forecast)

        result_df = reduce((lambda x, y: pd.concat([x, y])), fit_result)

        return result_df


    def combine_forecast(self, dataframes):
        """
        Combine all forecast result from different methods

        Parameters
        ----------
        dataframes : array-like of DataFrame
        """

        forecast = reduce(lambda left, right: pd.merge(left, right, \
                        on=["index", self.date_col, self.geoid_col, self.cumulative_metric], \
                        how="outer"), dataframes)

        return forecast
