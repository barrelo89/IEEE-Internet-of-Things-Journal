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

def plot_satisfaction_distribution(pdf1, mean_rank1, std1, pdf2, mean_rank2, std2, pdf3, mean_rank3, std3, rank1, bar_width):
  #이 plot을 통해서 preference가 더 잘 반영됬다는 것을 알 수 있다. 
  
  x_axis = np.array(range(len(pdf3)))*10
  
  
  plt.bar(x_axis,             pdf1, width = bar_width, color ='b', label = 'benchmark1')
  plt.bar(x_axis+bar_width,   pdf2, width = bar_width, color ='g', label = 'benchmark2')
  plt.bar(x_axis+2*bar_width, pdf3, width = bar_width, color = 'r', label = 'proposed')
  
  plt.vlines(x = (mean_rank3 / len(rank1))*100, ymin = 0, ymax = 0.5, color = 'r', linestyles = 'dashed', label = 'Average Rank of Proposed')
  plt.vlines(x = (mean_rank1 / len(rank1))*100, ymin = 0, ymax = 0.5, color = 'b', linestyles = 'dashdot', label = 'Average Rank of Benchmark1')
  plt.vlines(x = (mean_rank2 / len(rank1))*100, ymin = 0, ymax = 0.5, color = 'g', linestyles = 'dotted', label = 'Average Rank of Benchmark2')
  
  plt.xlabel('Rank Range of Matched Providers')
  plt.ylabel('Distribution of Mathced Providers')
  plt.xticks(x_axis)
  plt.ylim([0, 0.5])
  plt.legend(loc = 'best')
  plt.show() 

def Plot_Social_Welfare(task_n, provider_n, max_value, max_alpha, max_deadline, max_task_size, max_mu, max_provider_bid, max_provider_skill, task_columns, provider_columns):
  
  tasks = TaskCreator(task_n, max_value, max_alpha, max_deadline, max_task_size)
  providers = ProviderCreator(provider_n, max_mu, max_provider_bid, max_provider_skill)
  
  pro_exp_social = []
  exp_social = []
  pro_real_social = []
  real_social = []
  
  for i in range(1,11):
    capacity = 20*i
    
    auctioneer = Platform(capacity)

  #platform에서 winners를 선택한다. 
    W_requesters, req_threshold = auctioneer.WinningRequesterSelection(tasks)
    W_providers, pro_threshold = auctioneer.WinningProviderSelection(providers)

  #혹시 모르니깐 requesters와 providers의 숫자를 맞춘다.
    W_requesters, W_providers = auctioneer.Trimming(W_requesters, W_providers)

  #requesters와 providers의 preference ordering을 만든다.
    requester_preference_ordering = auctioneer.ordering_matrix(W_providers,provider_columns)
    provider_preference_ordering = auctioneer.ordering_matrix(W_requesters,task_columns)

    b_rank, b_mean_rank = benchmark_satisfaction_level(requester_preference_ordering)
    b_pdf, b_cdf = bench_distribution(b_rank)

    match = auctioneer.SMA(requester_preference_ordering, provider_preference_ordering)
    #In match, requester: keys, providers: values

    #expected social welfare comparison
    social_welfare = SocialWelfare(W_requesters, W_providers, 0.01, match)
    benchmark = SystemExpectedSocialwelfare(W_requesters, W_providers, 0.01)
    
    pro_exp_social.append(social_welfare)
    exp_social.append(benchmark)
    
    rank, mean_rank = auctioneer.satisfaction_level(match, requester_preference_ordering)
    pdf, cdf = auctioneer.satisfaction_distribution(rank)
    
    #real social welfare
    welfare, t_sub = Proposed_SocialWelfare(W_requesters, W_providers, 0.01, match)
    bench_welfare = Existing_SocialWelfare(W_requesters, W_providers, 0.01)
    
    pro_real_social.append(welfare)
    real_social.append(bench_welfare)
  
  x_axis = np.array(range(1,11))*20
  bar_width = 10
  fig, ax = plt.subplots(nrows = 1, ncols = 2)
  ax[0].bar(x_axis, pro_exp_social, width = bar_width, label = 'proposed exp.SW', color = 'b')
  ax[0].bar(x_axis+bar_width, exp_social, width = bar_width, label = 'benchmark exp.SW', color = 'r')
  ax[0].legend(loc = 'best')
  
  ax[1].bar(x_axis, pro_real_social, width = bar_width, label = 'proposed real.SW', color = 'b')
  ax[1].bar(x_axis+bar_width, real_social, width = bar_width, label = 'benchmark real.SW', color = 'r')
  ax[1].legend(loc = 'best')
  plt.show()  

def r_plot_preference_cdf(r_cumulative1, r_cumulative2, r_cumulative3, r_mean1, r_mean2, r_mean3):

  x_axis = np.arange(1, len(r_cumulative1)+1,1)

  plt.plot(x_axis, r_cumulative1, 'r', label = 'benchmark1')
  plt.plot(x_axis, r_cumulative2, 'b', label = 'benchmark2')
  plt.plot(x_axis, r_cumulative3, 'g', label = 'proposed')
  plt.vlines(x = r_mean1, ymin = 0, ymax = 1.1, color = 'r', linestyles = 'dashed', label = 'mean of benchmark1')
  plt.vlines(x = r_mean2, ymin = 0, ymax = 1.1, color = 'b', linestyles = 'dashed', label = 'mean of benchmark2')
  plt.vlines(x = r_mean3, ymin = 0, ymax = 1.1, color = 'g', linestyles = 'dashed', label = 'mean of proposed')
  plt.ylim((0,1.1))
  plt.xlabel('Rank of Matched Providers in Preference List', fontsize = 20)
  plt.ylabel('Cumulative Distribution', fontsize = 20)
  plt.xticks(fontsize = 20)
  plt.legend(loc = 'lower right')
  plt.tight_layout()
  plt.show()  

def p_plot_preference_cdf(r_cumulative1, r_cumulative2, r_cumulative3, r_mean1, r_mean2, r_mean3):

  x_axis = np.arange(1, len(r_cumulative1)+1,1)

  plt.plot(x_axis, r_cumulative1, 'r', label = 'benchmark1')
  plt.plot(x_axis, r_cumulative2, 'b', label = 'benchmark2')
  plt.plot(x_axis, r_cumulative3, 'g', label = 'proposed')
  plt.vlines(x = r_mean1, ymin = 0, ymax = 1.1, color = 'r', linestyles = 'dashed', label = 'mean of benchmark1')
  plt.vlines(x = r_mean2, ymin = 0, ymax = 1.1, color = 'b', linestyles = 'dashed', label = 'mean of benchmark2')
  plt.vlines(x = r_mean3, ymin = 0, ymax = 1.1, color = 'g', linestyles = 'dashed', label = 'mean of proposed')
  plt.ylim((0,1.1))
  plt.xlabel('Rank of Matched Requesters in Preference List', fontsize = 20)
  plt.ylabel('Cumulative Distribution', fontsize = 20)
  plt.xticks(fontsize = 20)
  plt.legend(loc = 'lower right')
  plt.tight_layout()
  plt.show()  
  
def preference_cdf(time_unit, task_n, provider_n, capacity, max_value, max_deadline, max_alpha, max_task_size, max_provider_bid, max_provider_skill, max_mu, task_columns, provider_columns, iteration, output): 

  r_distribution1 = []
  r_distribution2 = []
  r_distribution3 = []

  r_cumulative1 = []
  r_cumulative2 = []
  r_cumulative3 = []

  p_distribution1 = []
  p_distribution2 = []
  p_distribution3 = []

  p_cumulative1 = []
  p_cumulative2 = []
  p_cumulative3 = []

  r_mean1=[]
  r_mean2=[]
  r_mean3=[]

  p_mean1=[]
  p_mean2=[]
  p_mean3=[]

  for _ in range(iteration):
  
    tasks = TaskCreator(task_n, max_value, max_alpha, max_deadline, max_task_size)
    providers = ProviderCreator(provider_n, max_mu, max_provider_bid, max_provider_skill)

    auctioneer = Platform(capacity)
  
#without consideration to alpha and mu '''winner selection'''
    W_requesters, req_threshold = auctioneer.WinningRequesterSelection(tasks)
    W_providers, pro_threshold = auctioneer.WinningProviderSelection(providers)

#with consideration to alpha and mu
    New_W_requesters, New_req_threshold = auctioneer.New_WinningRequesterSelection(tasks)
    New_W_providers, New_pro_threshold = auctioneer.New_WinningProviderSelection(providers)

#trimming process
    W_requesters, W_providers = auctioneer.Trimming(W_requesters, W_providers)
    New_W_requesters, New_W_providers = auctioneer.Trimming(New_W_requesters, New_W_providers)



#1. preference ordering

#satisfaction level without consideration of preference, alpha and mu: 1
    requester_preference_ordering = auctioneer.ordering_matrix(W_providers,provider_columns)
    provider_preference_ordering = auctioneer.ordering_matrix(W_requesters,task_columns)

    r_rank, r_mean_rank = benchmark_satisfaction_level(requester_preference_ordering)
    p_rank, p_mean_rank = benchmark_satisfaction_level(provider_preference_ordering)
  
    r_pdf1, r_cdf1 = bench_distribution(r_rank)
    p_pdf1, p_cdf1 = bench_distribution(p_rank)
  
    r_distribution1.append(r_pdf1)
    r_cumulative1.append(r_cdf1)
  
    p_distribution1.append(p_pdf1)
    p_cumulative1.append(p_cdf1)
  
    r_mean1.append(r_mean_rank)
    p_mean1.append(p_mean_rank)
  
#satisfaction level without consideration of preference but with alpha, mu: 2
    New_requester_preference_ordering = auctioneer.ordering_matrix(New_W_providers,provider_columns)
    New_provider_preference_ordering = auctioneer.ordering_matrix(New_W_requesters,task_columns)

    New_r_rank, New_r_mean_rank = benchmark_satisfaction_level(New_requester_preference_ordering)
    New_p_rank, New_p_mean_rank = benchmark_satisfaction_level(New_provider_preference_ordering)
  
    r_pdf2, r_cdf2 = bench_distribution(New_r_rank)
    p_pdf2, p_cdf2 = bench_distribution(New_p_rank)

    r_distribution2.append(r_pdf2)
    p_distribution2.append(p_pdf2)
  
    r_cumulative2.append(r_cdf2)
    p_cumulative2.append(p_cdf2)
  
    r_mean2.append(New_r_mean_rank)
    p_mean2.append(New_p_mean_rank)
  
#Run Stable Marriage Algorithm: In match, requester: keys, providers: values
    match = auctioneer.SMA(New_requester_preference_ordering, New_provider_preference_ordering)



#satisfaction with consideration to preference, alpha, mu:3 
#requesters
    r_rank3, r_mean_rank3 = auctioneer.satisfaction_level(match, New_requester_preference_ordering)
  
#providers
    p_rank3, p_mean_rank3 = auctioneer.satisfaction_level(match, New_provider_preference_ordering)

    r_pdf3, r_cdf3 = auctioneer.satisfaction_distribution(r_rank3)
    p_pdf3, p_cdf3 = auctioneer.satisfaction_distribution(p_rank3)
  
    r_distribution3.append(r_pdf3)
    p_distribution3.append(p_pdf3)
  
    r_cumulative3.append(r_cdf3)
    p_cumulative3.append(p_cdf3)
  
    r_mean3.append(r_mean_rank3)
    p_mean3.append(p_mean_rank3)

  r_distribution1 = np.array(r_distribution1).mean(axis = 0)
  r_distribution2 = np.array(r_distribution2).mean(axis = 0)
  r_distribution3 = np.array(r_distribution3).mean(axis = 0)

  r_cumulative1 = np.array(r_cumulative1).mean(axis = 0)
  r_cumulative2 = np.array(r_cumulative2).mean(axis = 0)
  r_cumulative3 = np.array(r_cumulative3).mean(axis = 0)


  p_distribution1 = np.array(p_distribution1).mean(axis = 0)
  p_distribution2 = np.array(p_distribution2).mean(axis = 0)
  p_distribution3 = np.array(p_distribution3).mean(axis = 0)

  p_cumulative1 = np.array(p_cumulative1).mean(axis = 0)
  p_cumulative2 = np.array(p_cumulative2).mean(axis = 0)
  p_cumulative3 = np.array(p_cumulative3).mean(axis = 0)


  r_mean1= np.array(r_mean1).mean()
  r_mean2= np.array(r_mean2).mean()
  r_mean3= np.array(r_mean3).mean()

  p_mean1= np.array(p_mean1).mean()
  p_mean2= np.array(p_mean2).mean()
  p_mean3= np.array(p_mean3).mean()
  
  output.put((r_distribution1, r_distribution2, r_distribution3, r_cumulative1, r_cumulative2, r_cumulative3, p_distribution1, p_distribution2, p_distribution3, p_cumulative1, p_cumulative2, p_cumulative3,
  r_mean1, r_mean2, r_mean3, p_mean1, p_mean2, p_mean3))
    
#항상 3가지 case에 대해서 생각하자.
#1. without consideration to preference & alpha, mu
#2. with consideration to alpha, mu but without preference
#3. with consideration to preference & alpha, mu



if __name__ == "__main__":

#time_unit setting
  time_unit = 0.01
  
#sample을 몇 개를 생성할지 정하자 
  task_n = 1000
  provider_n = 2000
  capacity = 200

#task_info 설정
  max_value = 80
  max_deadline = 100
  max_alpha = 80 #나중에 max_alpha를 변화시키면서 한 번 보자
  max_task_size = 10

#provider bid & skill 정보를 설정
  max_provider_bid = 10 
  max_provider_skill = 10
  max_mu = 1 #이 값도 한 번 변화시키서 보자.

  task_columns = ['alpha', 'deadline', 'bid to size ratio']
  provider_columns = ['mu','bid','skill']

#number of iterations
  iteration = 10

  num_core = multiprocessing.cpu_count() - 2
  
  results = []
  outputs = []
  process = []
  
  for _ in range(num_core):
    outputs.append(multiprocessing.Queue())
  for i in range(num_core):
    process.append(multiprocessing.Process(target = preference_cdf, args = (time_unit, task_n, provider_n, capacity, max_value, max_deadline, max_alpha, max_task_size, max_provider_bid, max_provider_skill, max_mu, task_columns, provider_columns, iteration, outputs[i])))
  for pro in process:
    pro.start()
  for i in range(num_core):
    results.append(outputs[i].get())
    outputs[i].close()
  for pro in process:
    pro.terminate()
  
  results = np.array(results).mean(axis = 0)
  
  r_plot_preference_cdf(results[3], results[4], results[5], results[12], results[13], results[14])
  p_plot_preference_cdf(results[9], results[10], results[11], results[15], results[16], results[17])
 
  
#results[0] = r_distribution1
#results[1] = r_distribution2
#results[2] = r_distribution3
#results[3] = r_cumulative1
#results[4] = r_cumulative2
#results[5] = r_cumulative3
#results[6] = p_distribution1
#results[7] = p_distribution2
#results[8] = p_distribution3
#results[9] = p_cumulative1
#results[10] = p_cumulative2
#results[11] = p_cumulative3
#results[12] = r_mean1
#results[13] = r_mean2
#results[14] = r_mean3
#results[15] = p_mean1
#results[16] = p_mean2
#results[17] = p_mean3

#plot_satisfaction_distribution(r_distribution1, r_mean1, r_std1, r_distribution2, r_mean2, r_std2, r_distribution3, r_mean3, r_std3, r_rank3, bar_width=3)
#plot_satisfaction_distribution(p_distribution1, p_mean1, p_std1, p_distribution2, p_mean2, p_std2, p_distribution3, p_mean3, p_std3, p_rank3, bar_width=3)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
#Plot_Social_Welfare(task_n, provider_n, max_value, max_alpha, max_deadline, max_task_size, max_mu, max_provider_bid, max_provider_skill, task_columns, provider_columns)












































  
#--------------------------------------------------------------------------------------------
#tasks_info = auctioneer.requester_info_matrix(tasks)
#providers_info = auctioneer.provider_info_matrix(providers)

#task_columns = ['type', 'deadline', 'bid to size ratio', 'value']
#dic = {'sensitive': 0, 'quasi': 1, 'insensitive': 2} 
#type이 insensitive할 수 록 좋다.
#deadline이 클 수 록 좋다.
#bid to size ratio 클 수 록 좋다
#value가 클 수 록 좋다. descending이 좋다.

#provider_columns = ['type','bid','skill']
#columns=['type','bid','skill']
#dic = {'sensitive': 0, 'quasi': 1, 'insensitive': 2}
#sensitive할 수 록 좋다. 작을 수록 ascending
#bid가 작을수록 좋다. ascending
#skill이 클 수 록 좋다. descending이다.
 
#이걸로 실제로는 언제 submit할지에 대해서 modeling을 해보았다.
#-> 그렇다면 submit time을 distribution에 근거해서 sampling을 했을 때, social welfare가 어떻게 될 지 한 번 보자.
   
#해당 t_sub에서 value를 찾기 위해서는 Tasks.TaskValueFunction(time_unit)에서 index를 알아야한다.
#t_sub를 가지고 알 수 있는 것이 아니라. 그럼 index는 t_sub / time_unit으로 구할 수 있겠지. 
 
#SystemExpectedSocialwelfare 얘는 winner selection과 provider selection에서 
#greedy 하게 select하고 그 후에 고려하는 것이 없지만

#proposed work는 requesters' preference를 고려해서 한다
#preference를 고려해서 더 level of satisfaction이 높겠지 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 






