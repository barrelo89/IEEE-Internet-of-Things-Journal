import numpy as np
import pandas as pd
import random
import itertools

class Platform():
  def __init__(self, capacity):
    self.capacity = capacity#platform은 limited capacity를 가지고 있다. 
    
  def requester_info_matrix(self, tasks): #tasks의 정보를 dataframe으로 return
    
    preference_matrix=[]
    columns = ['alpha', 'deadline', 'bid to size ratio']
    
    for task in tasks:
      preference_matrix.append([task.alpha_, task.deadline_, task.bid_/task.task_size_])
  
    preference = pd.DataFrame(preference_matrix, columns = columns)
    
    return preference

  def provider_info_matrix(self, providers): #providers의 정보를 dataframe으로 return
    preference_matrix = []
    columns=['mu','bid','skill']
  
    for provider in providers:
      preference_matrix.append([provider.mean_, provider.bid_, provider.skill_])
  
    preference = pd.DataFrame(preference_matrix, columns = columns)
    
    return preference    
      
  def WinningRequesterSelection(self, task_creatures): #selection criteria에 대해서 아직 incomplete
        
    #columns = columns = ['alpha', 'deadline', 'bid to size ratio']
    #alpha는 작은 것이 좋다
    #deadling은 긴 것이 좋고
    #bid to size ratio도 큰 것이 좋다.
    preference_matrix = self.requester_info_matrix(task_creatures)
    selection_metrics = preference_matrix['bid to size ratio']
    
    #selection_metrics에서 선발 기준을 택한다.
    #argsort는 기본이 ascending이다. 하지만 우리가 원하는 것은 descending으로 되는 걸 원한다.
    
    selection_indices = np.argsort(selection_metrics)[::-1]
    
    if len(task_creatures) > self.capacity:
      return task_creatures[selection_indices[:self.capacity]], task_creatures[selection_indices[self.capacity]]
    
    elif len(task_creatures) == self.capacity:
      return task_creatures[ selection_indices[:self.capacity-1] ], task_creatures[selection_indices[self.capacity-1]]
    
    else:
      return task_creatures[ selection_indices[:len(task_creatures)-1] ], task_creatures[selection_indices[len(task_creatures)-1]]
    
  def New_WinningRequesterSelection(self, task_creatures): #selection criteria에 대해서 아직 incomplete
        
    #columns = columns = ['alpha', 'deadline', 'bid to size ratio']
    #alpha는 작은 것이 좋다
    #deadling은 긴 것이 좋고
    #bid to size ratio도 큰 것이 좋다.
    preference_matrix = self.requester_info_matrix(task_creatures)
    
    global max_alpha
    global max_ratio1
    
    #maximum alpha 값을 구하고
    max_alpha = max(preference_matrix['alpha'])
    max_ratio1 = max(preference_matrix['bid to size ratio'])
    
    selection_metrics = (preference_matrix['bid to size ratio'] / max_ratio1) / (preference_matrix['alpha'] / max_alpha) 
    
    #selection_metrics에서 선발 기준을 택한다.
    #argsort는 기본이 ascending이다. 하지만 우리가 원하는 것은 descending으로 되는 걸 원한다.
    
    selection_indices = np.argsort(selection_metrics)[::-1]
     #argsort: ascending order -> [::-1]를 써서 descending으로 바꿈
     
    if len(task_creatures) > self.capacity:
      return task_creatures[selection_indices[:self.capacity]], task_creatures[selection_indices[self.capacity]]
    
    elif len(task_creatures) == self.capacity:
      return task_creatures[ selection_indices[:self.capacity-1] ], task_creatures[selection_indices[self.capacity-1]]
    
    else:
      return task_creatures[ selection_indices[:len(task_creatures)-1] ], task_creatures[selection_indices[len(task_creatures)-1]]
    
  def WinningProviderSelection(self, provider_creatures): #selection criterian에 대해서 incomplete
    
    #columns=['mu','bid','skill']
    #mu는 작은 것이 좋다.
    #bid는 작을수록 좋다, skill은 클 수 록 좋다.
    preference_matrix = self.provider_info_matrix(provider_creatures)
    selection_metrics = preference_matrix['bid'] / preference_matrix['skill']
    
    selection_indices = np.argsort(selection_metrics)
    
    if len(provider_creatures) > self.capacity:
      return provider_creatures[selection_indices[:self.capacity]], provider_creatures[selection_indices[self.capacity]]
    
    elif len(provider_creatures) == self.capacity:
      return provider_creatures[ selection_indices[:self.capacity-1] ], provider_creatures[selection_indices[self.capacity-1]]
    
    else:
      return provider_creatures[ selection_indices[:len(provider_creatures)-1] ], provider_creatures[selection_indices[len(provider_creatures)-1]]
   
  def New_WinningProviderSelection(self, provider_creatures): #selection criterian에 대해서 incomplete
    
    #columns=['mu','bid','skill']
    #mu는 작은 것이 좋다.
    #bid는 작을수록 좋다, skill은 클 수 록 좋다.
    preference_matrix = self.provider_info_matrix(provider_creatures)
    
    global max_mu
    global max_ratio
    
    max_mu = max(preference_matrix['mu'])
    max_ratio = max(preference_matrix['bid'] / preference_matrix['skill'])
        
    selection_metrics = (preference_matrix['mu'] / max_mu)*((preference_matrix['bid'] / preference_matrix['skill']) / max_ratio)
    
    selection_indices = np.argsort(selection_metrics)
    
    if len(provider_creatures) > self.capacity:
      return provider_creatures[selection_indices[:self.capacity]], provider_creatures[selection_indices[self.capacity]]
    
    elif len(provider_creatures) == self.capacity:
      return provider_creatures[ selection_indices[:self.capacity-1] ], provider_creatures[selection_indices[self.capacity-1]]
    
    else:
      return provider_creatures[ selection_indices[:len(provider_creatures)-1] ], provider_creatures[selection_indices[len(provider_creatures)-1]]
    
  def Trimming(self, w_requesters, w_providers):
    
    if len(w_requesters) < len(w_providers):
      w_requesters, w_providers = w_requesters, w_providers[:len(w_requesters)]
      
    elif len(w_requesters) > len(w_providers):
      w_requesters, w_providers = w_requesters[:len(w_providers)], w_providers
      
    return w_requesters, w_providers  
  
  def preference_permutation(self, columns):
    n_cols = len(columns)
    preference_combo = []
    for i in range(1, n_cols+1):
      result = list(itertools.permutations(columns, i))
      preference_combo = itertools.chain(preference_combo, result)
    
    return list(preference_combo)

  def preference_ordering(self, preference, subject_info):
  
    my_preference = preference[random.randint(0, len(preference)-1)]
    #my_preference는 all possible permutation중에서 하나를 선택한다.
    
    if 'deadline' in list(preference[-1]):
    #여기는 task에 대한 ordering을 한다.
      criteria = {'alpha':1, 'deadline':0, 'bid to size ratio':0}
    #ascending:1, descending:0
      return subject_info.sort_values(by=list(my_preference), ascending= [criteria[my] for my in my_preference]), subject_info.sort_values(by=list(my_preference), ascending= list(np.zeros(len(my_preference)))).index
    elif 'skill' in list(preference[-1]):
      criteria = {'mu':1, 'bid':1, 'skill':0}
    #ascending:1, descending:0
      return subject_info.sort_values(by=list(my_preference), ascending= [criteria[my] for my in my_preference]), subject_info.sort_values(by=list(my_preference), ascending= list(np.zeros(len(my_preference)))).index
    
  def ordering_matrix(self, creatures, columns):
  #all possible preference permutation
    preference = self.preference_permutation(columns)
  
  #extract creature info
    if 'deadline' in columns:#task라는 말
      info = self.requester_info_matrix(creatures)
    elif 'skill' in columns: #provider라는 말
      info = self.provider_info_matrix(creatures)

    matrix = [self.preference_ordering(preference, info)[1] for _ in range(len(creatures))]
  #one row: one preference ordering
    return pd.DataFrame(matrix)      
  
  def SMA(self, requester_preference_ordering, provider_preference_ordering):
    #requester_preference_ordering
    # each row indicates each requester's preference ordering on providers
    
    #provider_preference_ordering
    # each row indicates each provider's preference ordering on requesters
    
    requester_index = 0
    match = {} #key: requester index, value: provider index
  
    while len(match) < len(requester_preference_ordering.columns):
    
      #if len(match) == len(requester_preference_ordering.columns):
      #  return match 
      #print('requester index: %f' %requester_index)
    
      if requester_index not in match.keys():# if requester is not matched, find the first provider in its preference ordering
      
        provider_index = requester_preference_ordering.iloc[requester_index].dropna().iloc[0] #iloc인 것 주의!dataframe은 list와 다르게 []안에 들어갈 것이 무조건 row name, column name 이여야 한다. 순서가 아니라
        #print('provider index: %f' %provider_index)
      
      else: # if the requester is already matched move to the next requester
        requester_index +=1
        if requester_index > len(requester_preference_ordering.columns)-1:
          requester_index = 0
        continue

   
      if provider_index not in match.values():# if the provider is not matched
        match[requester_index] = provider_index # match it to the requester
  
      else: # if the provider is already matched to another requester
      
        competitor = list(match.items())[list(np.where(np.equal( list(match.values()), provider_index)))[0][0]][0] 
        #print('competitor: %f' %competitor)
        if  np.where(provider_preference_ordering.loc[provider_index] == requester_index) <  np.where(provider_preference_ordering.loc[provider_index] == competitor) :
          #print('win')
          del match[competitor]
          requester_preference_ordering.loc[competitor, list(np.where(np.equal(provider_index, requester_preference_ordering.loc[competitor])))[0][0]] = None
          match[requester_index] = provider_index
         
        else: 
          #print('loss')
          requester_preference_ordering.loc[requester_index, np.where(np.equal(provider_index, requester_preference_ordering.loc[requester_index]))[0][0]] = None
        
      requester_index += 1
    
      if requester_index > len(requester_preference_ordering.columns)-1:
        requester_index = 0
  
      #print(len(match))
      
    return match  
     
  def satisfaction_level(self, match, requester_preference_ordering):
  #총 선택된 provider의 순위 합이 적을 수 록 좋은 것
    rank = []
    for requester, provider in list(match.items()):

      rank.append(np.where(np.equal(requester_preference_ordering.loc[requester], provider))[0][0])
  #return the ranking of matched provider in each requester's preference ordering matrix
    return np.array(rank), np.mean(rank)

#  def satisfaction_distribution(self, rank):
#    pdf = []
#    cdf = []  
#    for i in range(1,11):
#    
#      if i == 1:
#        pdf.append(len(np.where(rank < len(rank)*i/10)[0]))
#        cdf.append(len(np.where(rank < len(rank)*i/10)[0]))
#      else:
#      
#        tmp = len(np.where(rank < len(rank)*(i-1)/10)[0])
#        pdf.append(len(np.where(rank < len(rank)*i/10)[0])- tmp)    
#        cdf.append(len(np.where(rank < len(rank)*i/10)[0]))
#      
#    return np.array(pdf) / sum(pdf), np.array(cdf) / len(rank)   

  def satisfaction_distribution(self, rank):
    pdf = []
    cdf = []
    
    for i in range(len(rank)):
      pdf.append(len(np.where(rank == i)[0]) / len(rank))
      cdf.append(len(np.where(rank <= i)[0]) / len(rank))
      
    return np.array(pdf), np.array(cdf)  

  def WPS_payment(self, winners, threshold):
    
    threshold_value = threshold.bid_ / threshold.skill_
    
    payment = []
    
    for i in range(0, len(winners)):
      payment.append(winners[i].skill_ * threshold_value)
    
    return np.array(payment)
    
  def New_WPS_payment(self, winners, threshold):
  
    threshold_value = threshold.mean_*(threshold.bid_ / threshold.skill_)
    #print('provider')
    #print('threshold: %f' %threshold_value)
    #print('mu: %f' %threshold.mean_)
    
    payment = []
    
    for i in range(0, len(winners)):
      payment.append(winners[i].skill_ * threshold_value / winners[i].mean_)
    
    return np.array(payment)
  
  def WRS_payment(self, winners, threshold):
    
    threshold_value = threshold.bid_ / threshold.task_size_
    
    payment = []
    
    for i in range(0, len(winners)):
      payment.append(winners[i].task_size_*threshold_value)
    
    return np.array(payment)  

  def New_WRS_payment(self, winners, threshold):
    
    threshold_value = (threshold.bid_/threshold.task_size_)* (1/threshold.alpha_)
    #print('requester')
    #print('threshold: %f' %threshold_value)
    #print('alpha: %f' %threshold.alpha_)
    
    payment = []
    
    for i in range(0, len(winners)):
      payment.append((winners[i].task_size_*winners[i].alpha_)*threshold_value)
    
    return np.array(payment)       
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
