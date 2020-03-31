# import of modules
import numpy as np
import scipy as sp
from scipy import linalg
from scipy import optimize
from scipy import special
from scipy import stats

"""
---------------------------------------------------------------------------
Generation of distribution objects
---------------------------------------------------------------------------
Developed by:
Sebastian Geyer (s.geyer@tum.de),
Felipe Uribe
Iason Papaioannou
Daniel Straub

Assistant Developers:
Alexander von Ramm
Matthias Willer
Peter Kaplan

Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
New Version 2019-01:
* Automatic import of required scipy subpackages
* Fixing of bugs in the lognormal and exponential distribution
* Optimization and fixing of minor bugs
Version 2018-01:
* Fixing of bugs in the gumbel,gumbelmin and gamma distribution
---------------------------------------------------------------------------
This software generates distribution objects according to the parameters
and definitions used in the distribution table of the ERA Group of TUM.
They can be defined either by their parameters, the first and second
moment or by data, given as a vector.
---------------------------------------------------------------------------
"""


class distributions(object):  # pragma: no cover
    """
    Generation of distribution objects
      construct distribution object with Obj = distributions(name,opt,val) with
      opt = "PAR", if you want to specify the distibution by its parameters:
      Binomial:                   Obj = distributions('binomial','PAR',[n,p])
      Geometric:                  Obj = distributions('geometric','PAR',[p])
      Negative binomial:          Obj = distributions('negativebinomial','PAR',[k,p])
      Poisson:                    Obj = distributions('poisson','PAR',[lambda,t])
      Uniform:                    Obj = distributions('uniform','PAR',[lower,upper])
      Normal:                     Obj = distributions('normal','PAR',[mean,std])
      Standard normal:            Obj = distributions('standardnormal','PAR',[])
      Log-normal:                 Obj = distributions('lognormal','PAR',[mu_lnx,sig_lnx])
      Exponential:                Obj = distributions('exponential','PAR',[lambda])
      Gamma:                      Obj = distributions('gamma','PAR',[lambda,k])
      Beta:                       Obj = distributions('beta','PAR',[r,s,lower,upper])
      Gumbel (to model minima):   Obj = distributions('gumbelMin','PAR',[a_n,b_n])
      Gumbel (to model maxima):   Obj = distributions('gumbel','PAR',[a_n,b_n])
      Fréchet:                    Obj = distributions('frechet','PAR',[a_n,k])
      Weibull:                    Obj = distributions('weibull','PAR',[a_n,k])
      GEV (to model maxima):      Obj = distributions('GEV','PAR',[beta,alpha,epsilon])
      GEV (to model minima):      Obj = distributions('GEVMin','PAR',[beta,alpha,epsilon])
      Pareto:                     Obj = distributions('pareto','PAR',[x_m,alpha])
      Rayleigh:                   Obj = distributions('rayleigh','PAR',[alpha])
      Chi-squared:                Obj = distributions('chisquare','PAR',[k])


      opt = "MOM", if you want to specify the distibution by its moments:
      Binomial:                   Obj = distributions('binomial','MOM',[mean,std])
      Geometric:                  Obj = distributions('geometric','MOM',[mean])
      Negative binomial:          Obj = distributions('negativebinomial','MOM',[mean,std])
      Poisson:                    Obj = distributions('poisson','MOM',[mean])
      Uniform:                    Obj = distributions('uniform','MOM',[mean,std])
      Normal:                     Obj = distributions('normal','MOM',[mean,std])
      Standard normal:            Obj = distributions('standardnormal','MOM',[])
      Log-normal:                 Obj = distributions('lognormal','MOM',[mean,std])
      Exponential:                Obj = distributions('exponential','MOM',[mean])
      Gamma:                      Obj = distributions('gamma','MOM',[mean,std])
      Beta:                       Obj = distributions('beta','MOM',[mean,std,lower,upper])
      Gumbel (to model minima):   Obj = distributions('gumbel','MOM',[mean,std])
      Gumbel (to model maxima):   Obj = distributions('gumbelMax','MOM',[mean,std])
      Fréchet:                    Obj = distributions('frechet','MOM',[mean,std])
      Weibull:                    Obj = distributions('weibull','MOM',[mean,std])
      GEV (to model minima):      Obj = distributions('GEVMin','MOM',[mean,std,epsilon])
      GEV (to model maxima):      Obj = distributions('GEV','MOM',[mean,std,epsilon])
      Pareto:                     Obj = distributions('pareto','MOM',[mean,std])
      Rayleigh:                   Obj = distributions('rayleigh','MOM',[mean])
      Chi-squared:                Obj = distributions('chisquare','MOM',[mean])
    """

    def __init__(self, name, opt, val=[0, 1]):
        # constructor
        self.Name = name.lower()
        val = np.array(val, ndmin=1, dtype=float)
        # definition of the distribution by its parameters
        if opt.upper() == "PAR":
            self.Par = val
            if name.lower() == "binomial":
                if (val[1] >= 0) and (val[1] <= 1) and (val[0] % 1 == 0):
                    self.Dist = sp.stats.binom(n=self.Par[0], p=self.Par[1])
                else:
                    raise RuntimeError(
                        "The Binomial distribution is not "
                        "defined for your parameters"
                    )

            elif name.lower() == "geometric":
                val = val[0]
                if val > 0 and val <= 1:
                    self.Dist = sp.stats.geom(p=val)
                else:
                    raise RuntimeError(
                        "The Geometric distribution is not "
                        "defined for your parameters"
                    )

            elif name.lower() == "negativebinomial":
                if (
                    (val[1] > 0)
                    and (val[1] <= 1)
                    and (val[0] > 0)
                    and (val[0] % 1 == 0)
                ):
                    self.Dist = sp.stats.nbinom(n=val[0], p=val[1])
                else:
                    raise RuntimeError(
                        "The Negative Binomial distribution "
                        "is not defined for your parameters"
                    )

            elif name.lower() == "poisson":
                n = len(val)
                if n == 1:
                    val = val[0]
                    if val > 0:
                        self.Dist = sp.stats.poisson(mu=val)
                    else:
                        raise RuntimeError(
                            "The Poisson distribution is not "
                            "defined for your parameters"
                        )
                if n == 2:
                    if val[0] > 0 and val[1] > 0:
                        self.Par = val[0] * val[1]
                        self.Dist = sp.stats.poisson(mu=val[0] * val[1])
                    else:
                        raise RuntimeError(
                            "The Poisson distribution is not "
                            "defined for your parameters"
                        )

            elif name.lower() == "exponential":
                val = val[0]
                if val[0] > 0:
                    self.Dist = sp.stats.expon(scale=1 / val[0])
                else:
                    raise RuntimeError(
                        "The Exponential distribution is not "
                        "defined for your parameters"
                    )

            # parameters of the gamma distribution: a = k, scale = 1/lambda
            elif name.lower() == "gamma":
                if val[0] > 0 and val[1] > 0:
                    self.Par[0] = val[0]
                    self.Par[1] = 1 / val[1]
                    self.Dist = sp.stats.gamma(a=val[0], scale=1 / val[1])
                else:
                    raise RuntimeError(
                        "The Gamma distribution is not defined " "for your parameters"
                    )

            elif name.lower() == "beta":
                """
                beta distribution in lecture notes can be shifted in order to
                account for ranges [a,b] -> this is not implemented yet
                """
                if (val[0] > 0) and (val[1] > 0) and (val[2] < val[3]):
                    self.Par[0] = val[0]
                    self.Par[1] = val[1]
                    self.Par[2] = val[2]
                    self.Par[3] = val[3]
                    self.Dist = sp.stats.beta(a=val[0], b=val[1])
                else:
                    raise RuntimeError(
                        "The Beta distribution is not defined " "for your parameters"
                    )

            elif name.lower() == "gumbelmin":
                """
                this distribution can be used to model minima
                """
                if val[1] > 0:
                    """
                    sigma is the scale parameter
                    mu is the location parameter
                    """
                    self.Par[0] = val[0]
                    self.Par[1] = val[1]
                    self.Dist = sp.stats.genextreme(c=0, scale=val[1], loc=-val[0])
                else:
                    raise RuntimeError(
                        "The Gumbel distribution is not defined" " for your parameters"
                    )

            elif name.lower() == "gumbel":
                """
                mirror image of this distribution can be used to model maxima
                """
                if val[1] > 0:
                    """
                    sigma is the scale parameter
                    mu is the location parameter
                    """
                    self.Par[0] = val[0]
                    self.Par[1] = val[1]
                    self.Dist = sp.stats.genextreme(c=0, scale=val[1], loc=val[0])
                else:
                    raise RuntimeError(
                        "The Gumbel distribution is not defined" " for your parameters"
                    )

            elif name.lower() == "frechet":
                if (val[0] > 0) and (val[1] > 0):
                    self.Par[0] = val[0]
                    self.Par[1] = val[1]
                    self.Dist = sp.stats.genextreme(
                        c=-1 / val[1], scale=val[0] / val[1], loc=val[0]
                    )
                else:
                    raise RuntimeError(
                        "The Frechet distribution is not define" "d for your parameters"
                    )

            elif name.lower() == "weibull":
                if (val[0] > 0) and (val[1] > 0):
                    self.Par[0] = val[0]
                    self.Par[1] = val[1]
                    self.Dist = sp.stats.weibull_min(c=val[0], scale=val[1])
                else:
                    raise RuntimeError(
                        "The Weibull distribution is not "
                        "definied for your parameters"
                    )

            elif name.lower() == "gev":
                if val[1] > 0:
                    self.Par[0] = val[0]
                    self.Par[1] = val[1]
                    self.Par[2] = val[2]
                    self.Dist = sp.stats.genextreme(c=-val[0], scale=val[1], loc=val[2])
                else:
                    raise RuntimeError(
                        "The Generalized Extreme Value Distribu"
                        "tion is not defined for your parameter"
                        "s"
                    )

            elif name.lower() == "gevmin":  # double check definition
                if val[1] > 0:
                    self.Dist = sp.stats.genextreme(
                        c=-val[0], scale=val[1], loc=-val[2]
                    )
                else:
                    raise RuntimeError(
                        "The Generalized Extreme Value Distribu"
                        "tion is not defined for your parameter"
                        "s"
                    )

            elif name.lower() == "pareto":
                if val[0] > 0 and val[1] > 0:
                    self.Dist = sp.stats.genpareto(
                        c=1 / val[1], scale=val[0] / val[1], loc=val[0]
                    )
                else:
                    raise RuntimeError(
                        "The Pareto distribution is not def" "ined for your parameters"
                    )

            elif name.lower() == "rayleigh":
                val = val[0]
                if val > 0:
                    self.Dist = sp.stats.rayleigh(scale=val)
                else:
                    raise RuntimeError(
                        "The Rayleigh distribution is not "
                        "defined for your parameters"
                    )

            elif name.lower() == "chisquare":
                val = val[0]
                if val > 0 and val % 1 == 0:
                    self.Dist = sp.stats.gamma(a=val / 2.0, scale=2)
                else:
                    raise RuntimeError(
                        "The Chisquared distribution is not "
                        "defined for your parameters"
                    )

            elif name.lower() == "uniform":
                """
                the distribution defined in scipy is uniform between loc and
                loc + scale
                """
                self.Dist = sp.stats.uniform(loc=val[0], scale=val[1] - val[0])

            elif (name.lower() == "standardnormal") or (
                name.lower() == "standardgaussian"
            ):
                self.Dist = sp.stats.norm(loc=0, scale=1)

            elif name.lower() == "normal" or name.lower() == "gaussian":
                if val[1] > 0:
                    self.Dist = sp.stats.norm(loc=val[0], scale=val[1])
                else:
                    raise RuntimeError(
                        "The Normal distribution is not defined" " for your parameters"
                    )

            elif name.lower() == "lognormal":
                if val[1] > 0:
                    """
                    a parametrization in terms of the underlying normally
                    distributed variable corresponds to s = sigma,
                    scale = exp(mu)
                    """
                    self.Dist = sp.stats.lognorm(s=val[1], scale=np.exp(val[0]))
                else:
                    raise RuntimeError(
                        "The Lognormal distribution is not "
                        "defined for your parameters"
                    )

            else:
                raise RuntimeError("Distribution type not available")

        #       if the distribution is to be defined using the moments
        elif opt.upper() == "MOM":
            self.Par = [None, None]
            if val.size > 1 and val[1] < 0:
                raise RuntimeError("The standard deviation must not be " "negative")

            elif name.lower() == "binomial":
                # Solve system of two equations for the parameters
                self.Par[1] = 1 - (val[1]) ** 2 / val[0]
                self.Par[0] = val[0] / self.Par[1]
                # Evaluate if distribution can be defined on the parameters
                if self.Par[1] % 1 <= 10 ** (-4):
                    self.Par[1] = round(self.Par[1], 0)
                    if 0 <= self.Par[0] and self.Par[1] <= 1:  # OK
                        self.Dist = sp.stats.binom(n=self.Par[0], p=self.Par[1])
                    else:
                        raise RuntimeError("Please select other moments")
                else:
                    raise RuntimeError("Please select other moments")

            elif name.lower() == "geometric":
                # Solve Equation for the parameter based on the first moment
                self.Par = 1 / val[0]
                """
                Evaluate if distribution can be defined on the parameter and if
                the moments are well defined
                """
                if 0 <= self.Par and self.Par <= 1:
                    self.Dist = sp.stats.geom(p=self.Par)
                else:
                    raise RuntimeError("Please select other moments")

            elif name.lower() == "negativebinomial":
                # Solve System of two equations for the parameters
                self.Par[1] = val[0] / ((val[0] + val[1]) ** 2)
                self.Par[0] = val[1] * self.Par[1]
                # Evaluate if distribution can be defined on the parameters
                if self.Par[0] % 1 <= 10 ** (-4):
                    self.Par[0] = round(self.Par[0], 0)
                    if 0 <= self.Par[1] and self.Par[1] <= 1:
                        self.Dist = sp.stats.nbinom(n=self.Par[0], p=self.Par[1])
                    else:
                        raise RuntimeError("Please select other moments")
                else:
                    raise RuntimeError("Please select other moments")

            elif name.lower() == "poisson":
                self.Par = val[0]
                # Evaluluate if moments match
                if 0 <= self.Par:
                    self.Dist = sp.stats.poisson(mu=self.Par)
                else:
                    raise RuntimeError("Please select other moments")

            elif name.lower() == "exponential":
                """
                Solve Equation for the parameter of the distribution based on
                the first moment
                """
                try:
                    self.Par = 1 / val[0]
                except ZeroDivisionError:
                    raise RuntimeError("The first moment cannot be zero!")
                if 0 <= self.Par:
                    self.Dist = sp.stats.expon(scale=1 / self.Par)
                else:
                    raise RuntimeError("Please select other moments")

            elif name.lower() == "gamma":
                # Solve system of equations for the parameters
                self.Par[0] = val[0] / (val[1] ** 2)  # parameter lambda
                self.Par[1] = self.Par[0] * val[0]  # parameter k
                # Evaluate if distribution can be defined on the parameters
                if self.Par[0] > 0 and self.Par[1] > 0:
                    self.Dist = sp.stats.gamma(a=self.Par[1], scale=1 / self.Par[0])
                else:
                    raise RuntimeError("Please select other moments")

            #            if (name.lower() == 'beta')

            elif name.lower() == "gumbelmin":
                ne = 0.57721566490153  # euler constant
                # solve two equations for the parameters of the distribution
                self.Par[1] = val[1] * np.sqrt(6) / np.pi  # scale parameter
                self.Par[0] = val[0] - ne * self.Par[1]  # location parameter
                if self.Par[1] > 0:
                    self.Dist = sp.stats.gumbel_l(loc=self.Par[0], scale=self.Par[1])
                else:
                    raise RuntimeError("Please select other moments")

            elif name.lower() == "gumbel":
                ne = 0.57721566490153  # euler constant
                # solve two equations for the parameters of the distribution
                self.Par[1] = val[1] * np.sqrt(6) / np.pi  # scale parameter
                self.Par[0] = val[0] - ne * self.Par[1]  # location parameter
                if self.Par[1] > 0:
                    self.Dist = sp.stats.gumbel_r(loc=self.Par[0], scale=self.Par[1])
                else:
                    raise RuntimeError("Please select other moments")

            elif name.lower() == "frechet":
                par0 = 2.0001

                def equation(par):
                    return (
                        np.sqrt(
                            sp.special.gamma(1 - 2 / par)
                            - sp.special.gamma(1 - 1 / par) ** 2
                        )
                        / sp.special.gamma(1 - 1 / par)
                        - val[1] / val[0]
                    )

                sol = sp.optimize.fsolve(equation, x0=par0, full_output=True)
                if sol[2] == 1:
                    self.Par[1] = sol[0][0]
                    self.Par[0] = val[0] / sp.special.gamma(1 - 1 / self.Par[1])
                else:
                    raise RuntimeError(
                        "fsolve could not converge to a solutio"
                        "n for determining the parameters of th"
                        "e Frechet distribution"
                    )
                if self.Par[0] > 0 and self.Par[1] > 0:
                    c = 1 / self.Par[1]
                    scale = self.Par[0] / self.Par[1]
                    loc = self.Par[0]
                    self.Dist = sp.stats.genextreme(c=c, scale=scale, loc=loc)
                else:
                    raise RuntimeError("Please select other moments")

            elif name.lower() == "weibull":

                def equation(par):
                    return (
                        np.sqrt(
                            sp.special.gamma(1 + 2 / par)
                            - (sp.special.gamma(1 + 1 / par)) ** 2
                        )
                        / sp.special.gamma(1 + 1 / par)
                        - val[1] / val[0]
                    )

                sol = sp.optimize.fsolve(equation, x0=0.02, full_output=True)
                if sol[2] == 1:
                    self.Par[1] = sol[0][0]
                    self.Par[0] = val[0] / sp.special.gamma(1 + 1 / self.Par[1])
                else:
                    raise RuntimeError(
                        "fsolve could not converge to a solutio"
                        "n for determining the parameters of th"
                        "e Weibull distribution"
                    )
                if self.Par[0] > 0 and self.Par[1] > 0:
                    self.Dist = sp.stats.weibull_min(c=self.Par[1], scale=self.Par[0])
                else:
                    raise RuntimeError("Please select other moments")

            elif name.lower() == "gev":  # doublecheck fsolve convergence
                if val[0] == val[2]:
                    self.Par[0] = -1
                    self.Par[1] = val[1]
                    self.Par[2] = val[2]
                else:
                    if val[0] > val[2]:
                        par0 = 0.3
                    else:
                        par0 = -1.5

                    def equation(par):
                        return (
                            (
                                sp.special.gamma(1 - 2 * par)
                                - sp.special.gamma(1 - par) ** 2
                            )
                            / (sp.special.gamma(1 - par) - 1) ** 2
                            - (val[2] / (val[1] - val[2])) ** 2
                        )

                    sol = sp.optimize.fsolve(equation, x0=par0, full_output=True)
                    if sol[2] == 1:
                        self.Par[0] = sol[0][0]
                        self.Par[1] = (
                            (val[0] - val[3])
                            * self.Par[0]
                            / (sp.special.gamma(1 - self.Par[0]) - 1)
                        )
                    else:
                        raise RuntimeError(
                            "fsolve could not converge to a sol"
                            "ution for determining the paramete"
                            "rs of the GEV distribution"
                        )
                if self.Par[1] > 0:
                    self.Dist = sp.stats.genextreme(
                        c=-self.Par[0], scale=self.Par[1], loc=self.Par[2]
                    )
                else:
                    raise RuntimeError("Please select other moments")

            elif name.lower() == "pareto":
                self.Par[1] = 1 + np.sqrt(1 - (val[0] / val[1]) ** 2)
                self.Par[0] = val[0] * (self.Par[1] - 1) / self.Par[1]
                if self.Par[0] > 0 and self.Par[1] > 0:
                    c = 1 / self.Par[1]
                    scale = self.Par[0] / self.Par[1]
                    loc = self.Par[0]
                    self.Dist = sp.stats.genpareto(c=c, scale=scale, loc=loc)
                else:
                    raise RuntimeError("Please select other moments")

            elif name.lower() == "rayleigh":
                self.Par = val[0] / np.sqrt(np.pi / 2)
                if self.Par > 0:
                    self.Dist = sp.stats.rayleigh(scale=self.Par)
                else:
                    raise RuntimeError("Please select other moments")

            elif name.lower() == "chisquare":
                self.Par = val[0]
                if self.Par % 1 <= 10 ** (-4):
                    self.Par = round(self.Par, 0)
                else:
                    raise RuntimeError("Please select other moments")
                if self.Par > 0:
                    self.Dist = sp.stats.gamma(a=self.Par / 2.0, b=2.0)
                else:
                    raise RuntimeError("Please select other moments")

            elif name.lower() == "uniform":
                # compute parameters
                self.Par[0] = val[0] - np.sqrt(12) * val[1] / 2
                self.Par[1] = val[0] + np.sqrt(12) * val[1] / 2
                # Define distribution
                self.Dist = sp.stats.uniform(loc=val[0], scale=val[1] - val[0])

            elif (name.lower() == "standardnormal") or (
                name.lower() == "standardgaussian"
            ):
                self.Par = [0, 1]
                self.Dist = sp.stats.norm()

            elif (name.lower() == "normal") or (name.lower() == "gaussian"):
                self.Par = val
                self.Dist = sp.stats.norm(loc=self.Par[0], scale=self.Par[1])

            elif name.lower() == "lognormal":
                if val[0] <= 0:
                    raise RuntimeError(
                        "Please select other moments, first moment must be greater than zero"
                    )
                # solve two equations for the parameters of the distribution
                self.Par[0] = np.log(val[0] ** 2 / np.sqrt(val[1] ** 2 + val[0] ** 2))
                self.Par[1] = np.sqrt(np.log(1 + (val[1] / val[0]) ** 2))
                self.Dist = sp.stats.lognorm(s=self.Par[1], scale=np.exp(self.Par[0]))

        # if the distribution is to be fitted to a data vector
        elif opt.upper() == "DATA":
            if name.lower() == "binomial":
                raise RuntimeError(
                    "The binomial distribution is not supported" " in DATA"
                )

            elif name.lower() == "negativebinomial":
                raise RuntimeError(
                    "The negative binomial distribution is not " "supported in DATA"
                )

            elif name.lower() == "geometric":
                raise RuntimeError(
                    "The geometric distribution is not " "supported in DATA"
                )

            elif name.lower() == "poisson":
                raise RuntimeError(
                    "The poisson distribution is not supported" " in DATA"
                )

            elif name.lower() == "exponential":
                pars = sp.stats.expon.fit(val, floc=0)
                self.Par = 1 / pars[1]
                self.Dist = sp.stats.expon(scale=1 / self.Par)

            elif name.lower() == "gamma":
                pars = sp.stats.gamma.fit(val, floc=0)
                self.Par[0] = pars[0]
                self.Par[1] = 1 / pars[2]
                self.Dist = sp.stats.gamma(a=self.Par[0], scale=1 / self.Par[1])

            elif name.lower() == "beta":
                raise RuntimeError("The beta distribution is not supported " "in DATA")

            elif name.lower() == "gumbel":
                pars = sp.stats.gumbel_r.fit(val)
                self.Par[0] = pars[0]
                self.Par[1] = pars[1]
                self.Dist = sp.stats.gumbel_r(loc=self.Par[0], scale=self.Par[1])

            elif name.lower() == "gumbelmin":
                pars = sp.stats.gumbel_l.fit(val)
                self.Par[0] = pars[0]
                self.Par[1] = pars[1]
                self.Dist = sp.stats.gumbel_l(loc=self.Par[0], scale=self.Par[1])

            elif name.lower() == "frechet":
                raise RuntimeError(
                    "The frechet distribution is not supported " "in DATA"
                )

            elif name.lower() == "weibull":
                pars = sp.stats.weibull_min.fit(val, floc=0)
                self.Par[0] = pars[0]
                self.Par[1] = pars[2]
                self.Dist = sp.stats.weibull_min(c=self.Par[0], scale=self.Par[1])

            elif name.lower() == "normal" or name.lower() == "gaussian":
                pars = sp.stats.norm.fit(val)
                self.Par[0] = pars[0]
                self.Par[1] = pars[1]
                self.Dist = sp.stats.norm(loc=self.Par[0], scale=self.Par[1])

            elif name.lower() == "lognormal":
                pars = sp.stats.lognorm.fit(val, floc=0)
                self.Par[0] = pars[0]
                self.Par[1] = np.log(pars[2])
                self.Dist = sp.stats.lognorm(s=self.Par[0], scale=np.exp(self.Par[1]))

            elif name.lower() == "gev":
                pars = sp.stats.genextreme.fit(val)
                self.Par[0] = -pars[0]
                self.Par[1] = pars[2]
                self.Par[2] = pars[1]
                self.Dist = sp.stats.genextreme(
                    c=-self.Par[0], scale=self.Par[1], loc=self.Par[2]
                )

            elif name.lower() == "gevmin":
                pars = sp.stats.genextreme.fit(-val)
                self.Par[0] = -pars[0]
                self.Par[1] = pars[2]
                self.Par[2] = pars[1]
                self.Dist = sp.stats.genextreme(
                    c=-self.Par[0], scale=self.Par[1], loc=self.Par[2]
                )

            elif name.lower() == "pareto":
                raise RuntimeError(
                    "The pareto distribution is not supported " "in DATA"
                )

            elif name.lower() == "rayleigh":
                pars = sp.stats.rayleigh.fit(val, floc=0)
                self.Par = pars[1]
                self.Dist = sp.stats.rayleigh(scale=self.Par)

            elif name.lower() == "chisquare":
                raise RuntimeError(
                    "The Chisquare distribution is not " " supported in DATA"
                )
            else:
                raise RuntimeError("Distribution type not available")

        else:
            raise RuntimeError("Unknown option :" + opt)

    # %% ----------------------------------------------------------------------------
    def mean(self):
        if self.Name == "negativebinomial":
            return self.Dist.mean() + self.Par[0]

        if self.Name == "gumbel":
            ne = 0.57721566490153
            return self.Par[0] + self.Par[1] * ne

        if self.Name == "beta":
            return (self.Par[1] * self.Par[2] + self.Par[0] * self.Par[3]) / (
                self.Par[0] + self.Par[1]
            )

        if self.Name == "gevmin":
            return -self.Dist.mean()

        else:
            return self.Dist.mean()

    # %% ----------------------------------------------------------------------------
    def std(self):
        if self.Name == "beta":
            return self.Dist.std() * (self.Par[3] - self.Par[2])
        return self.Dist.std()

    # %% ----------------------------------------------------------------------------
    def pdf(self, x):

        if self.Name == "binomial":
            return self.Dist.pmf(x)

        elif self.Name == "geometric":
            return self.Dist.pmf(x)

        elif self.Name == "negativebinomial":
            return self.Dist.pmf(x - self.Par[0])

        elif self.Name == "poisson":
            return self.Dist.pmf(x)

        elif self.Name == "beta":
            """
            I believe there is a mistake in the matlab implementation of
            distributions since the pdf value in the center is the same no matter
            if the support is [0, 1] or [0, 2]
            """
            return self.Dist.pdf((x - self.Par[2]) / (self.Par[3] - self.Par[2])) / (
                self.Par[3] - self.Par[2]
            )

        elif self.Name == "gevmin":
            return self.Dist.pdf(-x)

        else:
            return self.Dist.pdf(x)

    # %% ----------------------------------------------------------------------------
    def cdf(self, x):
        if self.Name == "negativebinomial":
            return self.Dist.cdf(x - self.Par[0])

        if self.Name == "beta":
            return self.Dist.cdf((x - self.Par[2]) / (self.Par[3] - self.Par[2]))

        if self.Name == "gevmin":
            return self.Dist.cdf(-x)  # <-- this is not a proper cdf !

        else:
            return self.Dist.cdf(x)

    # %% ----------------------------------------------------------------------------
    def random(self, size=None):
        # Matlab distributions returns mxm samples if n isnt given, is this behaviour wanted?
        # Yes, this is the Matlab style for generating arrays/matrices, however python/numpy uses an list/tuple for the dimensions so let's stay consistent
        if self.Name == "binomial":
            samples = sp.stats.binom.rvs(int(self.Par[0]), p=self.Par[1], size=size)
            return samples

        elif self.Name == "negativebinomial":
            samples = self.Dist.rvs(size=size) + self.Par[0]
            return samples

        elif self.Name == "beta":
            samples = (
                self.Dist.rvs(size=size) * (self.Par[3] - self.Par[2]) + self.Par[2]
            )
            return samples

        elif self.Name == "gevmin":
            return self.Dist.rvs(size=size) * (-1)

        else:
            samples = self.Dist.rvs(size=size)
            return samples

    # %% ----------------------------------------------------------------------------
    def icdf(self, y):
        if self.Name == "negativebinomial":
            return self.Dist.ppf(y) + self.Par[0]

        elif self.Name == "beta":
            return self.Dist.ppf(y) * (self.Par[3] - self.Par[2]) + self.Par[2]

        elif self.Name == "gevmin":
            return -self.Dist.ppf(y)

        else:
            return self.Dist.ppf(y)
