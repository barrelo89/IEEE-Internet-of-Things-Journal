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

time_unit = 1
max_value = 100
max_deadline = 100
max_alpha = 100 #나중에 max_alpha를 변화시키면서 한 번 보자
max_task_size = 10

max_provider_bid = 10
max_provider_skill = 10
max_mu = 1 #이 값도 한 번 변화시키서 보자.

markersize = 8

task1 = Tasks(max_value, max_alpha, max_deadline, max_task_size)
task2 = Tasks(max_value, max_alpha, max_deadline, max_task_size)
task3 = Tasks(max_value, max_alpha, max_deadline, max_task_size)
task4 = Tasks(max_value, max_alpha, max_deadline, max_task_size)

task1.original_value_ = max_value
task2.original_value_ = max_value
task3.original_value_ = max_value
task4.original_value_ = max_value

task1.deadline_ = max_deadline/2
task2.deadline_ = max_deadline/2
task3.deadline_ = max_deadline/2
task4.deadline_ = max_deadline/2

task1.expiry_ = 1.5*max_deadline/2
task2.expiry_ = 1.5*max_deadline/2
task3.expiry_ = 1.5*max_deadline/2
task4.expiry_ = 1.5*max_deadline/2

task1.alpha_ = 0.01
task2.alpha_ = 0.1
task3.alpha_ = 1
task4.alpha_ = 10

values1, X_axis = task1.TaskValueFunction(time_unit)
values2, X_axis = task2.TaskValueFunction(time_unit)
values3, X_axis = task3.TaskValueFunction(time_unit)
values4, X_axis = task4.TaskValueFunction(time_unit)

plt.plot(X_axis, values1, 'r', label = r'$\alpha$ = %.2f' %task1.alpha_, lw = 2)
plt.plot(X_axis, values2, 'g--*', markersize = markersize, label = r'$\alpha$ = %.1f' %task2.alpha_, lw = 1)
plt.plot(X_axis, values3, 'b--v', markersize = markersize, label = r'$\alpha$ = %.1f' %task3.alpha_, lw = 1)
plt.plot(X_axis, values4, 'y--^', markersize = markersize, label = r'$\alpha$ = %.1f' %task4.alpha_, lw = 1)
plt.vlines(x = task1.expiry_, ymin = 0, ymax = max_value+10, colors = 'k', linestyles = 'dashed', label = r'$t^{ex}=50$', lw = 2)
plt.vlines(x = task1.deadline_, ymin = 0, ymax = max_value+10, colors = 'k', linestyles = 'dotted', label = r'$t^d=75$', lw = 2)
plt.legend(loc = 'best', fontsize = 20)
plt.ylim(0,max_value+10)
plt.xlim(0,task1.expiry_+10)
plt.xlabel('Time Elapsed', fontsize = 20)
plt.ylabel(r'Task Valuation, $V_j(t)$', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.tight_layout()
plt.show()
