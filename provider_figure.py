import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from sklearn.metrics import auc
import pandas as pd
import itertools

from task import Tasks
from provider import Provider 
from platforms import Platform
from functions import *
import multiprocessing

#--------------------------------PROVIDER.EPS-------------------------------------------------------------------------------------------------------------------------------------------------------------

time_unit = 0.01
max_value = 80
max_deadline = 100
max_alpha = 80 #나중에 max_alpha를 변화시키면서 한 번 보자
max_task_size = 10

max_provider_bid = 10 
max_provider_skill = 10
max_mu = 1 #이 값도 한 번 변화시키서 보자.

task1 = Tasks(max_value, max_alpha, max_deadline, max_task_size)
task1.original_value_ = max_value
task1.deadline_ = max_deadline/4
task1.expiry_ = 1.5*max_deadline/4

values1, X_axis = task1.TaskValueFunction(time_unit)

provider1 = Provider(max_mu, max_provider_bid, max_provider_skill)
provider2 = Provider(max_mu, max_provider_bid, max_provider_skill)
provider3 = Provider(max_mu, max_provider_bid, max_provider_skill)

provider1.mean_ = 0.5
provider1.sigma_ = 3*provider1.mean_

provider2.mean_ = 0.8
provider2.sigma_ = 3*provider2.mean_

provider3.mean_ = 1.2
provider3.sigma_ = 3*provider3.mean_

probability1, cdf1, x_axis1 = provider1.prob_distribution(task1, time_unit)
probability2, cdf2, x_axis2 = provider2.prob_distribution(task1, time_unit)
probability3, cdf3, x_axis3 = provider3.prob_distribution(task1, time_unit)

plt.plot(x_axis1, probability1, 'r', label = 'when mu is %f' %provider1.mean_, lw = 3)
plt.plot(x_axis1, probability2, 'g', label = 'when mu is %f' %provider2.mean_, lw = 3)
plt.plot(x_axis1, probability3, 'b', label = 'when mu is %f' %provider3.mean_, lw = 3)

plt.vlines(x = task1.expiry_, ymin = 0, ymax = max_value+10, colors = 'k', linestyles = 'dashed', label = 'expiry', lw = 2)
plt.vlines(x = task1.deadline_, ymin = 0, ymax = max_value+10, colors = 'k', linestyles = 'dotted', label = 'deadline', lw = 2)
plt.legend(loc = 'best')
plt.ylim(0,0.5)
plt.xlim(0,task1.expiry_+10)
plt.xlabel('Time Elapsed', fontsize = 20)
plt.ylabel('Task Completion Probability', fontsize = 20)
plt.xticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.tight_layout()
plt.show()
















