import random
import numpy as np
from scipy.stats import truncnorm
class Provider():
  def __init__(self, max_mu, max_provider_bid, max_provider_skill) : 
    self.mean_ = 0.5 + np.around(random.uniform(0, max_mu), decimals = 2)
    self.sigma_ = self.mean_
    self.bid_ = 0.01 + np.around(random.uniform(0, max_provider_bid), decimals = 2)
    self.skill_ = 0.01 +np.around(random.uniform(0, max_provider_skill), decimals = 2)
    
  def prob_distribution(self, task, time_unit):
    
    x_axis = np.arange(0,task.expiry_+time_unit, time_unit)
    
    a = task.deadline_
    b = task.expiry_
    
    probability = truncnorm.pdf(x_axis, -b, b, loc = a*self.mean_, scale = self.sigma_)
    
    cdf = truncnorm.cdf(x_axis, -b, b, loc = a*self.mean_, scale = self.sigma_)
    
    probability = probability / ((cdf[-1] - cdf[0]))
    cdf = (cdf - cdf[0]) / ((cdf[-1] - cdf[0]))
    
    
    return probability, cdf, x_axis

  def submit(self, x_axis, probability):
    denominator = sum(probability)
    t_sub = np.random.choice(x_axis, p = probability / denominator)
    return t_sub   