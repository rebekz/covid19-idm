import sys
import logging
import math
import pandas as pd
import numpy as np
from functools import reduce
from scipy import stats as sps


class RTConstant:
    DATE_COL = "date"
    GEO_ID_COL = "geo_id"
    METRIC_COL = "positive"
    INCREMENT_COL = "increment"


class RTModel:
    """
    This class extends the analysis of Kevin Systrom about estimating covid-19's R_t in real-time
    https://github.com/k-sys/covid-19/blob/master/Realtime%20R0.ipynb and code from
    https://github.com/covid-projections/covid-data-model/

    Parameters
    ----------
    start_date : str
        starting date
    min_cases : int
        minimum number of cases required to run inference
    window_size : int
        size of the sliding Gaussian window to compute
    r_list : array-like
        array of r_t to compute posteriors over. Doesn't really need to be configured
    serial_interval : float
        estimation interval time between infection to subsequent transmission
    process_sigma : float
        stdev of process model. Increasing this allows for larger
        instant deltas in R_t, shrinking it smooths things, but allows for less rapid
        change. Can be interpreted as the std of the allowed shift in R_t day-to-day
    confidence_intervals : list(float)
        confidence interval to computed. .95 would be 95% credible intervals from 5% to 95%
    critical_threshold : array-like
        upper and lower bound for r_t that classify as critical
    high_threshold : array-like
        upper and lower bound for r_t that classify as high
    medium_threshold : array-like
       upper and lower bound for r_t that classify as medium
    low_threshold : array-like
       upper and lower bound for r_t that classify as low
    """

    def __init__(self,
                 start_date=None,
                 min_cases=5,
                 window_size=14,
                 kernel_std=5,
                 r_list=np.linspace(0, 10, 501),
                 serial_interval=3.92,
                 process_sigma=0.25,
                 confidence_intervals=[0.95],
                 critical_threshold=[1.4],
                 high_threshold=[1.1, 1.4],
                 medium_threshold=[0.9, 1.1],
                 low_threshold=[0.0, 0.9]
                 ):

        self.window_size = window_size
        self.kernel_std = kernel_std
        self.r_list = r_list
        self.serial_interval = serial_interval
        self.process_sigma = process_sigma
        self.confidence_intervals = confidence_intervals
        self.min_cases = min_cases
        self.critical_threshold = critical_threshold
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.low_threshold = low_threshold
        self.start_date = start_date

        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_timeseries(self, cases):
        """
        Reformat data from db_disease_outbreak-like format to proper timeseries data

        Parameters
        ----------
        cases : pd.DataFrame
            dataframe about covid19's cases
        """

        if self.start_date is not None:
            cases = cases.loc[cases.index >= self.start_date, :]

        cases = cases[[RTConstant.GEO_ID_COL, RTConstant.INCREMENT_COL]].reset_index()
        cases.columns = [RTConstant.DATE_COL, RTConstant.GEO_ID_COL, RTConstant.METRIC_COL]
        cases = cases.set_index([RTConstant.GEO_ID_COL, RTConstant.DATE_COL]).sort_index()

        if self.min_cases is not None:
            cases = cases.loc[cases[RTConstant.METRIC_COL] >= self.min_cases, :]

        return cases

    def replace_outliers(self,
        x,
        local_lookback_window=14,
        z_threshold=10,
        min_mean_to_consider=5):
        """
        Take a pandas.Series, apply an outlier filter, and return a pandas.Series.
        This outlier detector looks at the z score of the current value compared to the mean and std
        derived from the previous N samples, where N is the local_lookback_window.
        For points where the z score is greater than z_threshold, a check is made to make sure the mean
        of the last N samples is at least min_mean_to_consider. This makes sure we don't filter on the
        initial case where values go from all zeros to a one. If that threshold is met, the value is
        then replaced with the linear interpolation between the two nearest neighbors.
        Parameters
        ----------
        x
            Input pandas.Series with the values to analyze
        log
            Logger instance
        local_lookback_window
            The length of the rolling window to look back and calculate the mean and std to baseline the
            z score. NB: We require the window to be full before returning any result.
        z_threshold
            The minimum z score needed to trigger the replacement
        min_mean_to_consider
            Threshold to skip low n cases, especially the degenerate case where a long list of zeros
            becomes a 1. This requires that the rolling mean of previous values must be greater than
            or equal to min_mean_to_consider to be replaced.
        Returns
        -------
        x
            pandas.Series with any triggered outliers replaced
        """

        # Calculate Z Score
        r = x.rolling(window=local_lookback_window, min_periods=local_lookback_window, center=False)
        m = r.mean().shift(1)
        s = r.std(ddof=0).shift(1)
        z_score = (x - m) / s

        possible_changes_idx = np.where(z_score > z_threshold)[0]
        changed_idx = []
        changed_value = []
        changed_snippets = []

        for idx in possible_changes_idx:
            if m[idx] > min_mean_to_consider:
                changed_idx.append(idx)
                changed_value.append(int(x[idx]))
                slicer = slice(idx - local_lookback_window, idx + local_lookback_window)
                changed_snippets.append(x[slicer].astype(int).tolist())
                try:
                    x[idx] = np.mean([x[idx - 1], x[idx + 1]])
                except IndexError:  # Value to replace can be newest and fail on x[idx+1].
                    # If so, just use previous.
                    x[idx] = x[idx - 1]

        return x

    def apply_gaussian_smoothing(self, ts, max_threshold=5):
        """
        Apply a rolling Gaussian window to smooth the data

        Parameters
        ----------
        ts : pd.DataFrame
            timeseries data
        max_threshold : int
            this parameter allows you to filter out entire series
            when they do not contain high enough numeric values.

        Returns
        -------
        smoothed : pd.Series
            Gaussian smoothed data
        """

        new_cases = ts.diff()
        index = new_cases.index
        series = self.replace_outliers(x=pd.Series(new_cases.values.squeeze()))
        smoothed = (
                series.rolling(
                    self.window_size, win_type="gaussian", min_periods=self.kernel_std, center=True
                    )
                    .mean(std=self.kernel_std)
                    .round())

        nonzeroes = [idx for idx, val in enumerate(smoothed) if val != 0]

        if smoothed.empty:
            idx_start = 0
        elif max(smoothed) < max_threshold:
            idx_start = len(smoothed)
        else:
            idx_start = nonzeroes[0]

        smoothed = smoothed.iloc[idx_start:]

        return smoothed

    def get_posteriors(self, ts):
        """
        Generate posteriors for R_t

        Parameters
        ----------
        ts : pd.Series
            timeseries data

        Returns
        -------
        posteriors : array-like
        start_idx : int
        """

        if len(ts) == 0:
            return None, None

        GAMMA = 1/self.serial_interval

        # (1) Calculate Lambda (the Poisson likelihood given the data) based on
        # the observed increase from t-1 cases to t cases.
        lam = ts[:-1].values * np.exp(GAMMA * (self.r_list[:, None] - 1))

        # (2) Calculate each day's likelihood over R_t
        likelihoods = pd.DataFrame(
            data=sps.poisson.pmf(ts[1:].values, lam),
            index=self.r_list,
            columns=ts.index[1:],
        )

        # (3) Create the Gaussian Matrix
        process_matrix = sps.norm(loc=self.r_list, scale=self.process_sigma).pdf(
            self.r_list[:, None]
        )

        # (3a) Normalize all rows to sum to 1
        process_matrix /= process_matrix.sum(axis=0)

        # (4) Calculate the initial prior. Gamma mean of "a" with mode of "a-1".
        prior0 = sps.gamma(a=2.5).pdf(self.r_list)
        prior0 /= prior0.sum()

        reinit_prior = sps.gamma(a=2).pdf(self.r_list)
        reinit_prior /= reinit_prior.sum()

        # Create a DataFrame that will hold our posteriors for each day
        # Insert our prior as the first posterior.
        posteriors = pd.DataFrame(
            index=self.r_list, columns=ts.index, data={ts.index[0]: prior0}
        )

        # We said we'd keep track of the sum of the log of the probability
        # of the data for maximum likelihood calculation.
        log_likelihood = 0.0

        # (5) Iteratively apply Bayes' rule
        for previous_day, current_day in zip(ts.index[:-1], ts.index[1:]):
            # (5a) Calculate the new prior
            current_prior = process_matrix @ posteriors[previous_day]

            # (5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
            numerator = likelihoods[current_day] * current_prior

            # (5c) Calculate the denominator of Bayes' Rule P(k)
            denominator = np.sum(numerator)

            # Execute full Bayes' Rule
            if denominator == 0:
                # Restart the baysian learning for the remaining series.
                # This is necessary since otherwise NaN values
                # will be inferred for all future days, after seeing
                # a single (smoothed) zero value.
                #
                # We understand that restarting the posteriors with the
                # re-initial prior may incur a start-up artifact as the posterior
                # restabilizes, but we believe it's the current best
                # solution for municipalities that have smoothed cases and
                # deaths that dip down to zero, but then start to increase
                # again.

                posteriors[current_day] = reinit_prior
            else:
                posteriors[current_day] = numerator / denominator

            # Add to the running sum of log likelihoods
            log_likelihood += np.log(denominator)

        self.log_likelihood = log_likelihood
        start_idx = -len(posteriors.columns)

        return posteriors, start_idx


    def highest_density_interval(self, posteriors, ci):
        """
        Given a PMF, generate the confidence bands

        Parameters
        ----------
        posteriors : pd.DataFrame
            probability mass function to compute intervals for
        ci : float
            confidence interval. value of .95 will compute the upper
            and lower bounds

        Returns
        -------
        ci_low : np.array
            low confidence intervals
        ci_high : np.array
            high confidence intervals
        """

        posterior_cdfs = posteriors.values.cumsum(axis=0)
        low_idx_list = np.argmin(np.abs(posterior_cdfs - (1 - ci)), axis=0)
        high_idx_list = np.argmin(np.abs(posterior_cdfs - ci), axis=0)
        ci_low = self.r_list[low_idx_list]
        ci_high = self.r_list[high_idx_list]
        return ci_low, ci_high

    def classify_case_growth(self, r_t, r_t_col="r_t_most_likely"):
        """
        Classify r_t_most_likely to critical, high, medium, low.
        we're using the defination from https://blog.covidactnow.org/modeling-metrics-critical-to-reopen-safely/

        Parameters
        ----------
        r_t : pd.DataFrame
            location with infection rate derived from r_t inference

        Returns
        -------
        r_t : pd.DataFrame
            location's infection rate with added case growth class
        """

        def classifier(rt):
            if rt > self.critical_threshold[0]:
                return "critical"
            elif self.high_threshold[0] < rt <= self.high_threshold[1]:
                return "high"
            elif self.medium_threshold[0] < rt <= self.medium_threshold[1]:
                return "medium"
            elif self.low_threshold[0] < rt <= self.low_threshold[1]:
                return "low"

        r_t["case_growth_class"] = r_t[r_t_col].apply(lambda x: classifier(x))

        return r_t

    def run(self, input_data):
        """
        Run Rt inference

        Parameters
        ----------
        input_data : pd.DataFrame
            timeseries data

        Returns
        -------
        r_t : pd.DataFrame
            infection rate (r_t) for each given location
        """

        cases = self.get_timeseries(input_data)
        results = []

        for geo_id, cases in cases.groupby(level=RTConstant.GEO_ID_COL):
            self.logger.info("Computing R_t for %s", geo_id)

            smoothed = self.apply_gaussian_smoothing(ts=cases)
            posteriors, start_idx = self.get_posteriors(smoothed)
            if posteriors is not None:
                index = cases.index.get_level_values("date")[start_idx:]
                r_t = pd.DataFrame(
                        {
                            "index": index,
                            "date_id": index,
                            RTConstant.GEO_ID_COL: geo_id,
                            "r_t_most_likely": posteriors.idxmax().values,
                        }
                    )

                for ci in self.confidence_intervals:
                    ci_low, ci_high = self.highest_density_interval(posteriors, ci)
                    low_band = 1 - ci
                    high_band = ci
                    r_t[f"r_t_ci_{int(math.floor(101 * low_band))}"] = ci_low
                    r_t[f"r_t_ci_{int(math.floor(100 * high_band))}"] = ci_high

                r_t = r_t.set_index("index")

                results.append(r_t)
            else:
                self.logger.info("Not able to compute R_t for %s", geo_id)

        result_df = reduce((lambda x, y: pd.concat([x, y])), results)
        result_df = self.classify_case_growth(result_df)

        return result_df
