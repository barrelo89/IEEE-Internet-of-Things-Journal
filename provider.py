import random
import numpy as np
from scipy.stats import truncnorm

class Provider():

    def __init__(self, max_mu, max_provider_bid, time_unit):

        self.mean_ = 0.1 + np.around(random.uniform(0, max_mu), decimals = 2)
        self.sigma_ = 2*self.mean_
        self.bid_ = 0.1 + np.around(random.uniform(0, max_provider_bid), decimals = 2)
        self.time_unit = time_unit
        #self.submission_time = 0

    def prob_distribution(self, task):

        x_axis = np.arange(0,task.expiry_+self.time_unit, self.time_unit)

        a = task.deadline_
        b = task.expiry_

        probability = truncnorm.pdf(x_axis, -b, b, loc = a*self.mean_, scale = self.sigma_)

        cdf = truncnorm.cdf(x_axis, -b, b, loc = a*self.mean_, scale = self.sigma_)

        probability = probability / ((cdf[-1] - cdf[0]))
        cdf = (cdf - cdf[0]) / ((cdf[-1] - cdf[0]))

        return probability, cdf, x_axis

    def submit(self, task):

        #if self.submission_time == 0:

        probability, cdf, x_axis = self.prob_distribution(task)

        denominator = sum(probability)
        t_sub = np.random.choice(x_axis, p = probability / denominator)

        #    self.submission_time = t_sub

        #else:
        #    t_sub = self.submission_time

        return t_sub
