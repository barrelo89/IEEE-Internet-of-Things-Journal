import random
import numpy as np
import math
#-----------------------------------------------------Task-------------------------------------------------------------------------------------------------------------------------------------------------
class Tasks(): #Task의 class를 만든다. Task가 가져야 할 features를 정의하자 먼저.(task_types, 1, 15, 10, expiry)
  def __init__(self, max_value, max_alpha, max_deadline, max_task_size):
    
    self.original_value_ = np.around(random.uniform(0.1, max_value), decimals = 2)
    self.alpha_ = 1 + np.around(random.uniform(0.1, max_alpha), decimals = 2)
    self.deadline_ = np.around(random.uniform(1,max_deadline),decimals = 2)
    self.expiry_ = self.deadline_+ np.around(random.uniform(0, 0.5*self.deadline_) ,decimals=2)
    self.task_size_ = random.randint(1, max_task_size)
    self.bid_ = self.original_value_
    
  def TaskValueFunction(self, time_unit):
    values = []
    X_axis = np.arange(0,self.expiry_+time_unit, time_unit)
    
    for iter in X_axis[:int(self.deadline_/time_unit)+1] :
      values.append(self.original_value_)
    for it in X_axis[int(self.deadline_/time_unit)+1:]:
      if -self.alpha_*(it-self.deadline_)**2 + self.original_value_ >= 0:
        values.append(-self.alpha_*(it-self.deadline_)**2 + self.original_value_)  
      else : values.append(0)
        
    return values, X_axis
    
  def time_variant_value(self, t_sub, time_unit):
    
    values, X_axis = self.TaskValueFunction(time_unit)
    
    if t_sub in X_axis:
    
      return values[int(t_sub / time_unit)]
    
    else: return 0
  
    


    


