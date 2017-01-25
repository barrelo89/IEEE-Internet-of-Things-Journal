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

def TaskCreator(n,task_types, range, max_value, max_deadline, expiry):#done
  creatures=[]
  for _ in np.arange(n):
    creatures.append(Tasks(task_types, range, max_value, max_deadline, expiry))#types, type_range, max_value, max_deadline, expiry)
  return np.array(creatures)

def ProviderCreator(n, types, max_provider_bid, max_provider_skill):#done
  creatures =[]
  
  for i in np.arange(n):
    creatures.append(Provider(types, max_provider_bid, max_provider_skill))
  
  return np.array(creatures)  

def SystemExpectedSocialwelfare(task_creatures, provider_creatures, time_unit):#done
  system_social_welfare=[]
  for task, provider in zip(task_creatures, provider_creatures):
    system_social_welfare.append(auc(task.TaskValueFunction(time_unit)[1], task.TaskValueFunction(time_unit)[0]*provider.prob_distribution(task, time_unit)[0]))
  return np.sum(system_social_welfare)

def SocialWelfare(task_creatures, provider_creatures, time_unit, match):#done: proposed_expected_socialwelfare
  welfare = []
  for req_index, pro_index in list(match.items()):
    welfare.append(auc(task_creatures[req_index].TaskValueFunction(time_unit)[1], task_creatures[req_index].TaskValueFunction(time_unit)[0]*provider_creatures[pro_index].prob_distribution(task_creatures[req_index], time_unit)[0]))
  
  return np.sum(welfare)

def benchmark_satisfaction_level(requester_preference_ordering):
  rank = []
  for i in range(len(requester_preference_ordering)):
    rank.append(np.where(requester_preference_ordering.loc[i]==i)[0][0])
    
  return np.array(rank), np.mean(rank)

def bench_distribution(b_rank):
  pdf = []
  cdf = []
  
  for i in range(1,11):
    
      if i == 1:
        pdf.append(len(np.where(b_rank < len(b_rank)*i/10)[0]))
        cdf.append(len(np.where(b_rank < len(b_rank)*i/10)[0]))
      else:
      
        tmp = len(np.where(b_rank < len(b_rank)*(i-1)/10)[0])
        pdf.append(len(np.where(b_rank < len(b_rank)*i/10)[0])- tmp)    
        cdf.append(len(np.where(b_rank < len(b_rank)*i/10)[0]))
      
  return np.array(pdf) / sum(pdf), np.array(cdf) / len(b_rank)

def task_provider_plot(task, provider):
  fig, ax1 = plt.subplots()
  ax2 = ax1.twinx()
  
  values, X_axis = task.TaskValueFunction(0.01)
  probability, cdf, x_axis = provider.prob_distribution(task, 0.01)
  
  ax2.plot(x_axis, probability, color = 'b', label = 'pdf')
  ax2.plot(x_axis, cdf, color = 'r', label = 'cdf')
  ax1.plot(X_axis, values, marker = 'o', markersize = 4, color = 'g', label = 'Task Value')
  ax1.plot([task.deadline_,task.deadline_],[0, max([task.original_value_+task.original_value_*0.1, 1.1])], ls = '--', label = 'deadline')
  ax1.plot([task.expiry_,task.expiry_],[0, max([task.original_value_+task.original_value_*0.1, 1.1])], ls = '-.', label = 'expiry') 
  #plt.legend(loc = 'best', fontsize = 25)
  ax1.set_xlabel('Time(t)').set_fontsize(25)
  ax1.set_ylabel('Task Value').set_fontsize(25)
  ax2.set_ylabel('PDF/CDF').set_fontsize(25)
  ax1.tick_params(labelsize = 20)
  ax2.tick_params(labelsize = 20)

  ax1.set_xlim([0, task.expiry_+task.expiry_*0.1])
  ax1.set_ylim([0, max([task.original_value_+task.original_value_*0.1, 1.1])])
  plt.tight_layout()
  plt.show()

def plot_satisfaction_distribution(b_rank, b_mean_rank, rank, mean_rank, bar_width):
  #이 plot을 통해서 preference가 더 잘 반영됬다는 것을 알 수 있다. 
  b_pdf, b_cdf = bench_distribution(b_rank)
  pdf, cdf = auctioneer.satisfaction_distribution(rank)
  
  x_axis = np.array(range(len(pdf)))*10
  
  plt.bar(x_axis, pdf, width = bar_width, color = 'r', label = 'proposed')
  plt.bar(x_axis+bar_width, b_pdf, width = bar_width, color ='b', label = 'benchmark')
  plt.vlines(x = (mean_rank / len(b_rank))*100, ymin = 0, ymax = 0.5, color = 'k', linestyles = 'dashed', label = 'Average Rank of Proposed')
  plt.vlines(x = (b_mean_rank / len(b_rank))*100, ymin = 0, ymax = 0.5, color = 'g', linestyles = 'dashdot', label = 'Average Rank of Benchmark')
  plt.xlabel('Rank Range of Matched Providers')
  plt.ylabel('Distribution of Mathced Providers')
  plt.xticks(x_axis)
  plt.ylim([0, 0.5])
  plt.legend(loc = 'best')
  plt.show()      
  
def Proposed_SocialWelfare(W_requesters, W_providers, time_unit, match):
  
  welfare = []
  
  dic_t_sub = {}
  
  for requester, provider in list(match.items()):
    
    probability, cdf, x_axis = W_providers[provider].prob_distribution(W_requesters[requester], time_unit)
    t_sub = W_providers[provider].submit(x_axis, probability)
    
    dic_t_sub[provider] = t_sub
    
    value = W_requesters[requester].time_variant_value(t_sub, time_unit)
    welfare.append(value)

  return sum(welfare), dic_t_sub    

def Existing_SocialWelfare(W_requesters, W_providers, time_unit):
  
  welfare = []
  
  for requester, provider in zip(W_requesters, W_providers):
   
    probability, cdf, x_axis = provider.prob_distribution(requester, time_unit)
    t_sub = provider.submit(x_axis, probability) 
    
    value = requester.time_variant_value(t_sub, time_unit)
    welfare.append(value)

  return sum(welfare)


time_unit = 0.01
max_value = 80
max_deadline = 100
max_alpha = 80 #나중에 max_alpha를 변화시키면서 한 번 보자
max_task_size = 10

max_provider_bid = 10 
max_provider_skill = 10
max_mu = 1 #이 값도 한 번 변화시키서 보자.

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

plt.plot(X_axis, values1, 'r', label = 'when alpha is %f' %task1.alpha_, lw = 3)
plt.plot(X_axis, values2, 'g', label = 'when alpha is %f' %task2.alpha_, lw = 3)
plt.plot(X_axis, values3, 'b', label = 'when alpha is %f' %task3.alpha_, lw = 3)
plt.plot(X_axis, values4, 'y', label = 'when alpha is %f' %task4.alpha_, lw = 3)
plt.vlines(x = task1.expiry_, ymin = 0, ymax = max_value+10, colors = 'k', linestyles = 'dashed', label = 'expiry', lw = 2)
plt.vlines(x = task1.deadline_, ymin = 0, ymax = max_value+10, colors = 'k', linestyles = 'dotted', label = 'deadline', lw = 2)
plt.legend(loc = 'best')
plt.ylim(0,max_value+10)
plt.xlim(0,task1.expiry_+10)
plt.xlabel('Time Elapsed', fontsize = 20)
plt.ylabel('Task Valuation', fontsize = 20)
plt.xticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.tight_layout()
plt.show()































