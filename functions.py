import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from sklearn.metrics import auc
from task import Tasks
from provider import Provider 
from platforms import Platform
import pandas as pd
import itertools

def TaskCreator(n,max_value, max_alpha, max_deadline, max_task_size):
  creatures=[]
  for _ in np.arange(n):
    creatures.append(Tasks(max_value, max_alpha, max_deadline, max_task_size))#types, type_range, max_value, max_deadline, expiry)
  return np.array(creatures)

def ProviderCreator(n, max_mu, max_provider_bid, max_provider_skill):
  creatures =[]
  
  for i in np.arange(n):
    creatures.append(Provider(max_mu, max_provider_bid, max_provider_skill))
  
  return np.array(creatures)  

def SystemExpectedSocialwelfare(task_creatures, provider_creatures, time_unit):#done
  system_social_welfare=[]
  for task, provider in zip(task_creatures, provider_creatures):
    system_social_welfare.append(auc(task.TaskValueFunction(time_unit)[1], task.TaskValueFunction(time_unit)[0]*provider.prob_distribution(task, time_unit)[0]))
  return np.sum(system_social_welfare)

def SocialWelfare(task_creatures, provider_creatures, time_unit, match):#done: proposed_expected_socialwelfare
  welfare = []
  for req_index, pro_index in list(match.items()):
    welfare.append(auc(task_creatures[int(req_index)].TaskValueFunction(time_unit)[1], task_creatures[int(req_index)].TaskValueFunction(time_unit)[0]*provider_creatures[int(pro_index)].prob_distribution(task_creatures[int(req_index)], time_unit)[0]))
  
  return np.sum(welfare)

def benchmark_satisfaction_level(requester_preference_ordering):
  rank = []
  for i in range(len(requester_preference_ordering)):
    rank.append(np.where(requester_preference_ordering.loc[i]==i)[0][0])
    
  return np.array(rank), np.mean(rank)

#def bench_distribution(b_rank):
#  pdf = []
#  cdf = []
#  
#  for i in range(1,11):
#    
#      if i == 1:
#        pdf.append(len(np.where(b_rank < len(b_rank)*i/10)[0]))
#        cdf.append(len(np.where(b_rank < len(b_rank)*i/10)[0]))
#      else:
#      
#        tmp = len(np.where(b_rank < len(b_rank)*(i-1)/10)[0])
#        pdf.append(len(np.where(b_rank < len(b_rank)*i/10)[0])- tmp)    
#        cdf.append(len(np.where(b_rank < len(b_rank)*i/10)[0]))
#      
#  return np.array(pdf) / sum(pdf), np.array(cdf) / len(b_rank)

def bench_distribution(b_rank):
  pdf = []
  cdf = []
  
  for i in range(len(b_rank)):
    pdf.append(len(np.where(b_rank == i)[0]) / len(b_rank))
    cdf.append(len(np.where(b_rank <= i)[0]) / len(b_rank))
  
  return np.array(pdf), np.array(cdf)



def task_provider_plot(task, provider):
  fig, ax1 = plt.subplots()
  ax2 = ax1.twinx()
  
  values, X_axis = task.TaskValueFunction(0.01)
  probability, cdf, x_axis = provider.prob_distribution(task, 0.01)
  print('cdf: %f' %auc(x_axis, probability))
  
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
  
def Proposed_SocialWelfare(W_requesters, W_providers, time_unit, match):
  
  welfare = []
  
  dic_t_sub = {}
  
  for requester, provider in list(match.items()):
    
    probability, cdf, x_axis = W_providers[int(provider)].prob_distribution(W_requesters[int(requester)], time_unit)
    t_sub = W_providers[int(provider)].submit(x_axis, probability)
    
    dic_t_sub[int(provider)] = t_sub
    
    value = W_requesters[int(requester)].time_variant_value(t_sub, time_unit)
    welfare.append(value)

  return sum(welfare), dic_t_sub    

def Existing_SocialWelfare(W_requesters, W_providers, time_unit):
  
  welfare = []
  time = []
  
  for requester, provider in zip(W_requesters, W_providers):
   
    probability, cdf, x_axis = provider.prob_distribution(requester, time_unit)
    t_sub = provider.submit(x_axis, probability) 
    
    time.append(t_sub)
    value = requester.time_variant_value(t_sub, time_unit)
    welfare.append(value)

  return sum(welfare), np.array(time)

def time_variant_money(t_sub, W_requesters, W_providers, fee, payment, match, time_unit): #return time-variant fee n payment for SMA
  #t_sub: dictionary indicating each provider's submission time_unit
  #W_requester, W_providers가 capacity+1 만큼 select하는 것을 기억하자.
  
  changed_fee = []   
  changed_payment = []  
  
  for requester, provider in list(match.items()):
    
    value = W_requesters[int(requester)].time_variant_value(t_sub[int(provider)], time_unit)
    achievement_ratio = value / W_requesters[int(requester)].original_value_
    changed_payment.append(achievement_ratio*payment[int(provider)])
    changed_fee.append(achievement_ratio*fee[int(requester)])

  return np.array(changed_fee), np.array(changed_payment)
  
def bench_time_variant_money(t_sub, W_requesters, W_providers, fee, payment, time_unit): #return time-variant fee n payment without SMA
  
  changed_fee = []   
  changed_payment = []  
  
  for requester, provider, time, f, p in zip(W_requesters, W_providers, t_sub, fee, payment):
    value = requester.time_variant_value(time, time_unit)
    achievement_ratio = value / requester.original_value_
    changed_payment.append(achievement_ratio*p)
    changed_fee.append(achievement_ratio*f)

  return np.array(changed_fee), np.array(changed_payment)  
  
def budget_balance_check(fee, payment):
  return np.sum(fee) - np.sum(payment)   
  
def mere_social_welfare(requesters): #the existing works가 가정한 deadline내에 다 submit한다고 할 때, social welfare이다.
  social_welfare = []
  for requester in requesters:
    social_welfare.append(requester.original_value_)
  return sum(social_welfare)
    
def cost_calculator(providers):
    cost = []
    for provider in providers:
      cost.append(provider.bid_)
    
    return sum(cost)
  
def extrapolate(targets, x, y):

  for target in targets:
    fit = np.polyfit(x[1:], y[target].iloc[1:],2)
    y[target].iloc[0] = np.poly1d(fit)[0]
  
  return y
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
#  