import logging
import sys
from seirsplus.models import *
import networkx
from datetime import datetime
import pandas as pd


class SEIRModel:
    """
    This class implements a SEIR-like model using SEIRS+ model from
    https://github.com/ryansmcgee/seirsplus#usage-instal

    Parameters
    ----------
    beta : float
       Rate of transmission (exposure)
    incubPeriod : float
       Duration of incubation period (days)
    durInf : float
       Duration of infections (days)
    xi : float, optional
        Rate of re-susceptiobility (upon recovery)
    mu_I : float, optional
        Rate of infection-related death
    mu_0 : float, optional
        Rate of baseline death
    nu : float, optional
        Rate of baseline birth
    p : float, optional
        probability of global interactions
    beta_D : float, optional
        Rate of transmission (exposure) for individuals with detected infections
    sigma_D : float, optional
        Rate of infection (upon exposure) for individuals with detected infections
    gamma_D : float, optional
        Rate of recovery (upon infection) for individuals with detected infections
    mu_D : float, optional
        Rate of infection-related death for individuals with detected infections
    theta_E : float, optional
        Rate of baseline testing for exposed individuals
    theta_I : float, optional
        Rate of baseline testing for infectious individuals
    psi_E : float, optional
        Probability of positive test results for exposed individuals
    psi_I : float, optional
        Probability of positive test results for exposed individuals
    q : float, optional
        Probability of quarantined individuals interaction with others
    initE : int, optional
        init number for exposed individuals
    initI : int, optional
        init number for infectious individuals
    initD_E : int, optional
        init number of detected exposed individuals
    initD_I : int, optional
        init number of detected infectious individuals
    initR : int, optional
        init number of recovered individuals
    initF : int, optional
        init number of infection-related fatalities
    logger : logging object, optional
        logging object, if not available it will initiate one
    """

    def __init__(self,
                 initN=1000000,
                 beta=0.2,
                 incubPeriod=5.2,
                 durInf=16.39,
                 xi=0,
                 mu_I=0.01,
                 mu_0=0,
                 nu=0,
                 p=0,
                 beta_D=None,
                 sigma_D=None,
                 gamma_D=None,
                 mu_D=None,
                 theta_E=0,
                 theta_I=0.02,
                 psi_E=1.0,
                 psi_I=1.0,
                 q=0,
                 initE=0,
                 initI=100,
                 initD_E=0,
                 initD_I=0,
                 initR=0,
                 initF=0,
                 logger=None):

        if logger is  None:
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        self.model = SEIRSModel(initN,
                                beta,
                                1/incubPeriod,
                                1/durInf,
                                xi=xi,
                                mu_I=mu_I,
                                mu_0=mu_0,
                                nu=nu,
                                p=p,
                                beta_D=beta_D,
                                sigma_D=sigma_D,
                                gamma_D=gamma_D,
                                mu_D=mu_D,
                                theta_E=theta_E,
                                theta_I=theta_I,
                                psi_E=psi_E,
                                psi_I=psi_I,
                                q=q,
                                initE=initE,
                                initI=initI,
                                initD_E=initD_E,
                                initD_I=initD_I,
                                initR=initR,
                                initF=initF)

        self.beta = beta
        self.beta_desc = "Rate of transmission"
        self.incubPeriod = incubPeriod
        self.incubPeriod_desc = "Duration of incubation period"
        self.durInf = durInf
        self.durInf_desc = "Duration of infection"
        self.xi = xi
        self.xi_desc = "Rate of re-susceptibility"
        self.mu_I = mu_I
        self.mu_I_desc = "Rate of infection-related mortality"
        self.mu_0 = mu_0
        self.mu_0_desc = "Rate of baseline mortality"
        self.nu = nu
        self.nu_desc = "Rate of baseline birth"
        self.p = p
        self.p_desc = "Probability of global interactions"
        self.beta_D = beta_D
        self.beta_D_desc = "Rate of transmission for detected cases"
        self.sigma_D = sigma_D
        self.sigma_D_desc = "Rate of progression for detected cases"
        self.gamma_D = gamma_D
        self.gamma_D_desc = "Rate of recovery for detected cases"
        self.mu_D = mu_D
        self.mu_D_desc = "Rate of infection-related mortality for detected cases"
        self.theta_E = theta_E
        self.theta_E_desc = "Rate of testing for exposed individuals"
        self.theta_I = theta_I
        self.theta_I_desc = "Rate of testing for infectious individuals"
        self.psi_E = psi_E
        self.psi_E_desc = "Probability of positive tests for exposed individuals"
        self.psi_I = psi_I
        self.psi_I_desc = "Probability of positive tests for infectious individuals"
        self.q = q
        self.q_desc = "probability of global interactions for quarantined individuals"

    def _get_checkpoint_days(self, date, start_date, end_date):
        """
        Convert date to days since start_date

        Parameters
        ----------
        date : str
            target date
        start_date : datetime
            start date
        end_date : datetime
            end date boundary

        Returns
        -------
        days : int
            number of days after start_date
        """

        dt = datetime.strptime(date, "%Y-%m-%d")
        if dt < start_date:
            raise KeyError("target date must after start date")
        if dt > end_date:
            raise KeyError("target date must before end_date")

        return (dt - start_date).days

    def run(self, start_date="2020-02-01", t=200, checkpoints=None, verbose=False):
        """
        Run SEIR model

        Paramters
        ---------
        start_date : str
            starting date of simulation
        t : int
            number of days of simulation
        checkpoints : dict
            place an intervention by override parameters during of simulation
            example:
            checkpoints = {'t': ["2020-04-01", "2020-06-01"], # starting date of intervention
                           # changing parameters
                           'p':       [0.1, 0.5],
                           'theta_E': [0.02, 0.02],
                           'theta_I': [0.02, 0.02],
                           'phi_E':   [0.2, 0.2],
                           'phi_I':   [0.2, 0.2]}
        """

        index = pd.date_range(start=start_date, periods=t+1, freq="D")
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = index[-1]
        if checkpoints is not None:
            checkpoints["t"] = [self._get_checkpoint_days(i, start_date_dt, end_date_dt) for i in checkpoints["t"]]


        self.model.run(t, checkpoints=checkpoints, verbose=verbose)

        result = pd.DataFrame(
                {
                    "date_id": index,
                    "S": self.model.numS[::10],
                    "E": self.model.numE[::10],
                    "I": self.model.numI[::10],
                    "D_E": self.model.numD_E[::10],
                    "D_I": self.model.numD_I[::10],
                    "R": self.model.numR[::10],
                    "F": self.model.numF[::10]
                },
                index=index)

        init_params = {"beta": {"value": self.beta, "description": self.beta_desc},
                       "incubPeriod": {"value": self.incubPeriod, "description": self.incubPeriod_desc},
                       "durInf": {"value": self.durInf, "description": self.durInf_desc},
                       "xi": {"value": self.xi, "description": self.xi_desc},
                       "mu_I": {"value": self.mu_I, "description": self.mu_I},
                       "mu_0": {"value": self.mu_0, "description": self.mu_0_desc},
                       "nu": {"value": self.nu, "description": self.nu_desc},
                       "p": {"value": self.p, "description": self.p_desc},
                       "beta_D": {"value": self.beta_D, "description": self.beta_D_desc},
                       "sigma_D": {"value": self.sigma_D, "description": self.sigma_D_desc},
                       "gamma_D": {"value": self.gamma_D, "description": self.gamma_D_desc},
                       "mu_D": {"value": self.mu_D, "description": self.mu_D_desc},
                       "theta_E": {"value": self.theta_E, "description": self.theta_E_desc},
                       "theta_I": {"value": self.theta_I, "description": self.theta_I_desc},
                       "psi_E": {"value": self.psi_E, "description": self.psi_E_desc},
                       "psi_I": {"value": self.psi_I, "description": self.psi_I_desc},
                       "q": {"value": self.q, "description": self.q_desc}}

        return result, init_params
