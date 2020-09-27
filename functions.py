from scipy.stats import truncnorm
from sklearn.metrics import auc
from task import Tasks
from provider import Provider
from platforms import Platform
from munkres import Munkres
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import sys
import time

def TaskSizer(W_requesters):

    task_size = []

    for requester in W_requesters:
        task_size.append(requester.task_size_)

    return np.array(task_size)

def BudgetBalanceCheck(fee, payment):
    return np.sum(fee) - np.sum(payment)

def CostCalculator(providers, task_sizes):

    cost = []

    for provider, task_size in zip(providers, task_sizes):
        cost.append(provider.bid_*task_size)

    return sum(cost)

def TaskCreator(n,max_value, max_alpha, max_deadline, max_task_size):

    creatures=[]

    for _ in np.arange(n):
        creatures.append(Tasks(max_value, max_alpha, max_deadline, max_task_size))#types, type_range, max_value, max_deadline, expiry)

    return np.array(creatures)

def ProviderCreator(n, max_mu, max_provider_bid, time_unit):

    creatures =[]

    for _ in range(n):
        creatures.append(Provider(max_mu, max_provider_bid, time_unit))
        #creatures.append(Provider(max_mu, max_provider_bid, max_provider_skill))

    return np.array(creatures)

def MereSW(requesters): #mere social welfare: assume all the requested task would be completed in deadline

    social_welfare = []

    for requester in requesters:

        social_welfare.append(requester.original_value_)

    return sum(social_welfare)

def ExpectedSW(task_creatures, provider_creatures): #expected social welfare

    expected_social_welfare = []

    for task, provider in zip(task_creatures, provider_creatures):
        expected_social_welfare.append(auc(task.TaskValueFunction(provider.time_unit)[1], task.TaskValueFunction(provider.time_unit)[0]*provider.prob_distribution(task)[0]))

    return np.sum(expected_social_welfare)

def PostSW(W_requesters, W_providers): #after submission, realized values

    welfare = []
    submission_time = []

    for requester, provider in zip(W_requesters, W_providers):

        t_sub = provider.submit(requester)
        submission_time.append(t_sub)

        value = requester.time_variant_value(t_sub, provider.time_unit)
        welfare.append(value)

    return sum(welfare), np.array(submission_time)

def TimeVariantMoney(t_sub, W_requesters, W_providers, fee, payment): #return time-variant fee & payment without SMA

    changed_fee = []
    changed_payment = []

    for requester, provider, time, f, p in zip(W_requesters, W_providers, t_sub, fee, payment):
        value = requester.time_variant_value(time, provider.time_unit)
        achievement_ratio = value / requester.original_value_
        changed_payment.append(achievement_ratio*p)
        changed_fee.append(achievement_ratio*f)

    return np.array(changed_fee), np.array(changed_payment)

def CreateProfitMatrix(tasks, providers): #row: requesters, column: providers

    profit_matrix = []

    #start_time = time.process_time()

    for t_idx, task in enumerate(tasks):

        profit_matrix_row = []

        for p_idx, provider in enumerate(providers):
            expected_profit = auc(task.TaskValueFunction(provider.time_unit)[1],task.TaskValueFunction(provider.time_unit)[0]*provider.prob_distribution(task)[0]) - provider.bid_*task.task_size_
            profit_matrix_row.append(expected_profit)

        profit_matrix.append(profit_matrix_row)

    #end_time = time.process_time()
    #print('matrix:', end_time - start_time)

    return np.array(profit_matrix)

def HungarianSelection(profit_matrix):

    min_element = abs(profit_matrix.min())
    max_element = profit_matrix.max()

    input_profit_matrix = max_element - (profit_matrix + min_element)
    #input_profit_matrix = sys.maxsize - (profit_matrix + min_element)
    #input_profit_matrix = sys.maxsize - profit_matrix

    #start_time = time.process_time()
    hungarian = Munkres()
    selected_pairs = hungarian.compute(input_profit_matrix)

    #end_time = time.process_time()
    #print('selection:', end_time - start_time)

    selected_requesters_idx = []
    selected_providers_idx = []

    expected_social_welfare = []

    for pair in selected_pairs:

        selected_requesters_idx.append(pair[0])
        selected_providers_idx.append(pair[1])
        expected_social_welfare.append(profit_matrix[pair[0], pair[1]])

    selected_requesters_idx = np.array(selected_requesters_idx)
    selected_providers_idx = np.array(selected_providers_idx)
    expected_social_welfare = np.array(expected_social_welfare)

    sw_sorted_idx = np.argsort(expected_social_welfare)[::-1] #in an descending order

    sorted_expected_social_welfare = expected_social_welfare[sw_sorted_idx]
    sorted_selected_requesters_idx = selected_requesters_idx[sw_sorted_idx]
    sorted_selected_providers_idx = selected_providers_idx[sw_sorted_idx]

    return sorted_expected_social_welfare, sorted_selected_requesters_idx, sorted_selected_providers_idx
#----------------------------------------------visualization functions--------------------------------------------------
def DataExtraction(data_sequence):

    result = []
    #[mere_SW, expected_SW, realized_SW, pre_budget, post_budget, payment, fee, changed_payment, changed_fee, cost, running_time]
    for data in data_sequence:
        result.append(data)
        #data.shape = 10x11 (# of capacity values x # of result values)

    return np.array(result)

def DataTrimming(data_matrix):

    output = []

    if len(data_matrix.shape) == 3: #3D data matrtix

        num_layer, num_row, num_col = data_matrix.shape #num_layer: # of iteration, num_row: kinds of capacity variation, num_col: kinds of result values

        for row_idx in range(num_row):

            row_result = []

            for layer_idx in range(num_layer):

                if data_matrix[layer_idx, row_idx, 0] != 0:
                    row_result.append(data_matrix[layer_idx, row_idx, :])

            if len(row_result) == 0:
                row_result = [0 for _ in range(num_col)]
            else:
                row_result = np.array(row_result).mean(axis = 0)

            row_result = [item for item in row_result]

            output.append(row_result)

        return np.array(output)

    if len(data_matrix.shape) == 4: #4D data matrix

        num_layer, num_provider, num_row, num_col = data_matrix.shape #num_layer: # of iteration, num_provider: kinds of provider #, num_row: kinds of capacity variation, num_col: kinds of result values

        result = []

        for provider_idx in range(num_provider):

            provider_result = []

            for row_idx in range(num_row):

                row_result = []

                for layer_idx in range(num_layer):

                    if data_matrix[layer_idx, provider_idx, row_idx, 0] != 0:

                        row_result.append(data_matrix[layer_idx, provider_idx, row_idx, :])

                if len(row_result) == 0:
                    row_result = [0 for _ in range(num_col)]
                else:
                    row_result = np.array(row_result).mean(axis = 0)

                row_result = [item for item in row_result]
                provider_result.append(row_result)
            result.append(provider_result)

        return np.array(result)































'''
def DataExtraction(data_sequence):

    result = []

    for data in data_sequence:
        result.append(pd.DataFrame(data, columns = ['mere_SW', 'expected_SW', 'realized_SW', 'pre_budget', 'post_budget', 'payment', 'fee', 'changed_payment', 'changed_fee', 'cost', 'running_time']))

    return result

def DataTrimming(data_list):

    result = []
    mean_matrix = []

    for data in data_list:

        mean_layer_matrix = []

        for payment, fee, changed_payment, changed_fee, SW in zip(data['payment'], data['fee'], data['changed_payment'], data['changed_fee'], data['mere_SW']):

            mean_row_matrix = []

            if SW > 0:

                payment = payment.sum()
                fee = fee.sum()
                changed_payment = changed_payment.sum()
                changed_fee = changed_fee.sum()

            for item in [payment, fee, changed_payment, changed_fee]:
                mean_row_matrix.append(item)

            mean_layer_matrix.append(mean_row_matrix)

        result.append(data.values[:, [0, 1, 2, 3, 4, 9, 10]])
        mean_matrix.append(mean_layer_matrix)

    result = np.array(result)
    mean_matrix = np.array(mean_matrix)

    result = np.concatenate([result, mean_matrix], axis = 2)


    num_layer, num_row, num_col = result.shape

    output = []

    for row_idx in range(num_row):

        row_result = []

        for layer_idx in range(num_layer):

            if result[layer_idx, row_idx, 0] != 0:
                row_result.append(result[layer_idx, row_idx, :])

        if len(row_result) == 0:
            row_result = [0 for _ in range(num_col)]
        else:
            row_result = np.array(row_result).mean(axis = 0)

        row_result = [item for item in row_result]

        output.append(row_result)

    return np.array(output)
'''
#
