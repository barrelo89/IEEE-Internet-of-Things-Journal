import numpy as np
import pandas as pd
import random
import itertools

class Platform():
    def __init__(self, capacity):
        self.capacity = capacity #a platform has a limited capacity to handle requested tasks

    def requester_info_matrix(self, tasks): #return requesters' bid information as a dataframe

        preference_matrix=[]
        columns = ['alpha', 'deadline', 'bid to size ratio']

        for task in tasks:
            preference_matrix.append([task.alpha_, task.deadline_, task.bid_/task.task_size_])

        preference = pd.DataFrame(preference_matrix, columns = columns)

        return preference

    def provider_info_matrix(self, providers): #return providers' bid information as a dataframe
        preference_matrix = []
        columns=['mu','bid']

        for provider in providers:
            preference_matrix.append([provider.mean_, provider.bid_])

        preference = pd.DataFrame(preference_matrix, columns = columns)

        return preference

    def WinningRequesterSelection(self, task_creatures):

        preference_matrix = self.requester_info_matrix(task_creatures)
        selection_metrics = preference_matrix['bid to size ratio']

        #In numpy.argsort, its default sorting is in ascending manner. What we want is in descending order

        selection_indices = np.argsort(selection_metrics)[::-1]

        if len(task_creatures) > self.capacity:
            return task_creatures[selection_indices][:self.capacity], task_creatures[selection_indices][self.capacity]

        elif len(task_creatures) == self.capacity:
            return task_creatures[selection_indices][:self.capacity-1], task_creatures[selection_indices][self.capacity-1]

        else:
            return task_creatures[selection_indices][:len(task_creatures)-1], task_creatures[selection_indices][len(task_creatures)-1]

    def WinningProviderSelection(self, provider_creatures):

        #columns=['mu','bid']
        #the smaller mu, the better
        #the smaller bid, the better
        preference_matrix = self.provider_info_matrix(provider_creatures)
        #selection_metrics = preference_matrix['bid'] / preference_matrix['skill']

        selection_metrics = preference_matrix['bid'] #selection criterion: bid (ask value)

        selection_indices = np.argsort(selection_metrics)

        if len(provider_creatures) > self.capacity:
            return provider_creatures[selection_indices][:self.capacity], provider_creatures[selection_indices][self.capacity]

        elif len(provider_creatures) == self.capacity:
            return provider_creatures[selection_indices][:self.capacity-1], provider_creatures[selection_indices][self.capacity-1]

        else:
            return provider_creatures[selection_indices][:len(provider_creatures)-1], provider_creatures[selection_indices][len(provider_creatures)-1]

    def New_WinningRequesterSelection(self, task_creatures, constant):

        preference_matrix = self.requester_info_matrix(task_creatures)

        max_alpha = (max(preference_matrix['alpha'])**constant)
        max_ratio = max(preference_matrix['bid to size ratio'])

        #constant가 추가되었다.
        selection_metrics = (preference_matrix['bid to size ratio'] / max_ratio) / (preference_matrix['alpha']**constant/max_alpha)

        #In numpy.argsort, its default sorting is in ascending manner. What we want is in descending order

        selection_indices = np.argsort(selection_metrics)[::-1]
        #argsort: ascending order -> adding [::-1], we change it into descending order

        if len(task_creatures) > self.capacity:
            return task_creatures[selection_indices][:self.capacity], task_creatures[selection_indices][self.capacity]

        elif len(task_creatures) == self.capacity:
            return task_creatures[selection_indices][:self.capacity-1], task_creatures[selection_indices][self.capacity-1]

        else:
            return task_creatures[selection_indices][:len(task_creatures)-1], task_creatures[selection_indices][len(task_creatures)-1]

    def New_WinningProviderSelection(self, provider_creatures, constant):

        #columns=['mu','bid']
        #the smaller mu, the better
        #the smaller bid, the better
        preference_matrix = self.provider_info_matrix(provider_creatures)

        max_mu = (max(preference_matrix['mu'])**constant)
        #max_ratio = max(preference_matrix['bid'] / preference_matrix['skill'])
        max_ratio = max(preference_matrix['bid'])

        #selection_metrics = (preference_matrix['mu'] / max_mu)*((preference_matrix['bid'] / preference_matrix['skill']) / max_ratio)
        selection_metrics = ((preference_matrix['mu'])**constant / max_mu)*(preference_matrix['bid'] / max_ratio)

        selection_indices = np.argsort(selection_metrics)

        if len(provider_creatures) > self.capacity:
            return provider_creatures[selection_indices[:self.capacity]], provider_creatures[selection_indices[self.capacity]]

        elif len(provider_creatures) == self.capacity:
            return provider_creatures[ selection_indices[:self.capacity-1] ], provider_creatures[selection_indices[self.capacity-1]]

        else:
            return provider_creatures[ selection_indices[:len(provider_creatures)-1] ], provider_creatures[selection_indices[len(provider_creatures)-1]]

    def Trimming(self, w_requesters, w_providers, req_threshold, pro_threshold):

        if len(w_requesters) < len(w_providers):
            pro_threshold = w_providers[len(w_requesters)]

            return w_requesters, w_providers[:len(w_requesters)], req_threshold, pro_threshold

        elif len(w_requesters) > len(w_providers):
            req_threshold = w_requesters[len(w_providers)]

            return w_requesters[:len(w_providers)], w_providers, req_threshold, pro_threshold

        else:
            return w_requesters, w_providers, req_threshold, pro_threshold

    def WPS_payment(self, winners, threshold):

        #threshold_value = threshold.bid_ / threshold.skill_
        threshold_value = threshold.bid_

        payment = []

        for i in range(0, len(winners)):
            #payment.append(winners[i].skill_ * threshold_value)
            payment.append(threshold_value)

        return np.array(payment)

    def New_WPS_payment(self, winners, threshold, constant):

        #threshold_value = threshold.mean_*(threshold.bid_ / threshold.skill_)
        threshold_value = ((threshold.mean_)**constant)*threshold.bid_

        payment = []

        for i in range(0, len(winners)):
            payment.append(threshold_value / (winners[i].mean_)**constant)

        return np.array(payment)

    def WRS_payment(self, winners, threshold):

        threshold_value = threshold.bid_ / threshold.task_size_

        payment = []

        for i in range(0, len(winners)):
            payment.append(winners[i].task_size_*threshold_value)

        return np.array(payment)

    def New_WRS_payment(self, winners, threshold, constant):

        threshold_value = (threshold.bid_/threshold.task_size_)* (1/(threshold.alpha_**constant))

        payment = []

        for i in range(0, len(winners)):
            payment.append((winners[i].task_size_*(winners[i].alpha_**constant))*threshold_value)

        return np.array(payment)
