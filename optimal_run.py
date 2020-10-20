from task import Tasks
from provider import Provider
from platforms import Platform
from functions import*
from munkres import Munkres
import sys
import multiprocessing
import numpy as np
import time
import pickle

def SW1(tasks, providers, capacity): #simple greedy approach

    #create an auctioneer
    auctioneer = Platform(capacity)

    start_time = time.process_time()
    #winner selection process
    W_requesters, req_threshold = auctioneer.WinningRequesterSelection(tasks)
    W_providers, pro_threshold = auctioneer.WinningProviderSelection(providers)

    #trimming process: make the # of selected requesters and providers equal
    W_requesters, W_providers, req_threshold, pro_threshold = auctioneer.Trimming(W_requesters, W_providers, req_threshold, pro_threshold)

    #cost calculation
    task_size = TaskSizer(W_requesters)
    cost = CostCalculator(W_providers, task_size)

    #calculate the payment to providers which guarantees truthfulness of providers
    payment = auctioneer.WPS_payment(W_providers, pro_threshold) #unit payment
    payment = payment*task_size #effective payment

    #calculate the fee for requesters which guarantees their truthfulness
    fee = auctioneer.WRS_payment(W_requesters, req_threshold)

    #mere social welfare: simple summation of task values without considering task depreciation after deadline
    mere_SW = MereSW(W_requesters) - cost

    #expected social welfare: summation of task values considering task depreciation after deadline
    expected_SW = ExpectedSW(W_requesters, W_providers) - cost

    #actually realized social welfare
    realized_SW, t_sub = PostSW(W_requesters, W_providers)
    realized_SW = realized_SW - cost

    #budget balance check before submission
    pre_budget = BudgetBalanceCheck(fee, payment)

    #budget balance after submission
    changed_fee, changed_payment = TimeVariantMoney(t_sub, W_requesters, W_providers, fee, payment)

    post_budget = BudgetBalanceCheck(changed_fee, changed_payment) #without consideration to alpha, mu and preference

    end_time = time.process_time()

    running_time = end_time - start_time

    #budget balance check
    #if pre_budget < 0: #budget balance not met. Note that we can only check the budget balance before providers' actual submission
    #    return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, running_time] #return all zeros
    #else: #budget balance met
    #    return [mere_SW, expected_SW, realized_SW, pre_budget, post_budget, payment.sum(), fee.sum(), changed_payment.sum(), changed_fee.sum(), cost, running_time]
    return [mere_SW, expected_SW, realized_SW, pre_budget, post_budget, payment.sum(), fee.sum(), changed_payment.sum(), changed_fee.sum(), cost, running_time]

def SW2(tasks, providers, capacity, power4requester, power4provider):

    #create an auctioneer
    auctioneer = Platform(capacity)

    start_time = time.process_time()
    #winner selection process with consideration to alpha and mu
    W_requesters, req_threshold = auctioneer.New_WinningRequesterSelection(tasks, power4requester)
    W_providers, pro_threshold = auctioneer.New_WinningProviderSelection(providers, power4provider)

    #trimming process
    W_requesters, W_providers, req_threshold, pro_threshold = auctioneer.Trimming(W_requesters, W_providers, req_threshold, pro_threshold)

    #cost calculation
    task_size = TaskSizer(W_requesters)
    cost = CostCalculator(W_providers, task_size)

    #payment calculation
    payment = auctioneer.New_WPS_payment(W_providers, pro_threshold, power4provider)
    payment = payment*task_size

    #fee calculation
    fee = auctioneer.New_WRS_payment(W_requesters, req_threshold, power4requester)

    #mere social welfare
    mere_SW = MereSW(W_requesters) - cost

    #expected social welfare
    expected_SW = ExpectedSW(W_requesters, W_providers) - cost

    #actually realized social welfare
    realized_SW, t_sub = PostSW(W_requesters, W_providers)
    realized_SW = realized_SW - cost

    #budget balance check before submission
    pre_budget = BudgetBalanceCheck(fee, payment)

    #budget balance after submission
    changed_fee, changed_payment = TimeVariantMoney(t_sub, W_requesters, W_providers, fee, payment)

    post_budget = BudgetBalanceCheck(changed_fee, changed_payment)

    end_time = time.process_time()

    running_time = end_time - start_time

    #budget balance check
    #if pre_budget < 0: #budget balance not met
    #    return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, running_time] #return all zeros
    #else: #budget balance met
    #    return [mere_SW, expected_SW, realized_SW, pre_budget, post_budget, payment.sum(), fee.sum(), changed_payment.sum(), changed_fee.sum(), cost, running_time]#proposed heuristic approach
    return [mere_SW, expected_SW, realized_SW, pre_budget, post_budget, payment.sum(), fee.sum(), changed_payment.sum(), changed_fee.sum(), cost, running_time]#proposed heuristic approach

def SW3(tasks, providers, capacity):

    start_time = time.process_time()
    #create profit matrix which would be input into hungarian algorithm
    '''THIS PROCESS CAN TAKE SIGNIFICANT AMOUNT OF TIME'''
    profit_matrix = CreateProfitMatrix(tasks, providers)

    #select the optimal requester-provider pairs which maximize the expected social welfare
    sorted_SW, requesters_idx, providers_idx = HungarianSelection(profit_matrix)

    #re-arrange requesters and providers
    sorted_requesters, sorted_providers = tasks[requesters_idx], providers[providers_idx]

    #create a platform
    auctioneer = Platform(capacity)

    #slice the winners according to platform capacity
    if len(requesters_idx) > capacity:
        req_threshold, pro_threshold = sorted_requesters[capacity], sorted_providers[capacity]
        W_requesters, W_providers = sorted_requesters[:capacity], sorted_providers[:capacity]

    elif len(requesters_idx) == capacity:
        req_threshold, pro_threshold = sorted_requesters[capacity-1], sorted_providers[capacity-1]
        W_requesters, W_providers = sorted_requesters[:capacity-1], sorted_providers[:capacity-1]

    else:
        req_threshold, pro_threshold = sorted_requesters[-1], sorted_providers[-1]
        W_requesters, W_providers = sorted_requesters[:-1], sorted_providers[:-1]


    #return the list of task size which will be used in the fee calculation
    task_size = TaskSizer(W_requesters)

    #return the cost incurred to providers
    cost = CostCalculator(W_providers, task_size)

    #calculate the payment to providers which guarantees truthfulness of providers
    payment = auctioneer.WPS_payment(W_providers, pro_threshold) #unit payment
    payment = payment*task_size #effective payment

    #calculate the fee for requesters which guarantees their truthfulness
    fee = auctioneer.WRS_payment(W_requesters, req_threshold)

    #mere social welfare: simple summation of task values without considering task depreciation after deadline
    mere_SW = MereSW(W_requesters) - cost

    #expected social welfare
    expected_SW = np.sum(sorted_SW[:len(W_requesters)])

    #actually realized social welfare
    realized_SW, t_sub = PostSW(W_requesters, W_providers)
    realized_SW = realized_SW - cost
    #budget balance check before submission
    pre_budget = BudgetBalanceCheck(fee, payment)

    #budget balance after submission
    changed_fee, changed_payment = TimeVariantMoney(t_sub, W_requesters, W_providers, fee, payment)

    post_budget = BudgetBalanceCheck(changed_fee, changed_payment) #without consideration to alpha, mu and preference

    end_time = time.process_time()
    running_time = end_time - start_time

    #budget balance check
    if pre_budget < 0: #budget balance not met. Note that we can only check the budget balance before providers' actual submission
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, running_time] #return all zeros
    else: #budget balance met
        return [mere_SW, expected_SW, realized_SW, pre_budget, post_budget, payment.sum(), fee.sum(), changed_payment.sum(), changed_fee.sum(), cost, running_time]

def SW(capacity, requester_num_unit, requester_range, power4requester, power4provider, output):

    greedy_cube = []
    proposed_cube = []
    optimal_cube = []

    for idx in range(1, requester_range + 1):

        total_requester_num = requester_num_unit*idx
        total_provider_num = total_requester_num*2

        tasks = TaskCreator(total_requester_num, max_value, max_alpha, max_deadline, max_task_size)
        providers = ProviderCreator(total_provider_num, max_mu, max_provider_bid, time_unit)


        #newly added!!!!!
        capacity = total_requester_num
        #newly added!!!!!

        #mere_SW, expected_SW, realized_SW, pre_budget, post_budget, payment, fee, changed_payment, changed_fee, cost, running_time
        greedy_result = SW1(tasks, providers, capacity)
        print('Greedy Complete!')
        proposed_result = SW2(tasks, providers, capacity, power4requester, power4provider)
        print('Proposed Complete!')
        optimal_result = SW3(tasks, providers, capacity)
        print('Optimal Complete!')

        greedy_cube.append(greedy_result)
        proposed_cube.append(proposed_result)
        optimal_cube.append(optimal_result)

    output.put([np.array(greedy_cube), np.array(proposed_cube), np.array(optimal_cube)])

#task information
max_value = 100
max_deadline = 100
max_task_size = 10
max_alpha = 100

#provider information
max_provider_bid = 10
max_mu = 1.5
time_unit = 0.5

#number of cores to use
num_core = multiprocessing.cpu_count() - 4

#power values for providers and requesters
power4requester, power4provider = 0.5, 0.5

#iteration
num_iter = 1

#platform capacity
capacity = 100
#for now, we ignore the capacity

#requester & provider number
requester_num_unit = 100
requester_range = 5

result = []

for _ in range(num_iter):

    outputs = []
    process = []

    for _ in range(num_core):
        outputs.append(multiprocessing.Queue())

    for output in outputs:
        process.append(multiprocessing.Process(target = SW, args = (capacity, requester_num_unit, requester_range, power4requester, power4provider, output)))

    for pro in process:
        pro.start()

    for output in outputs:
        result.append(output.get())

    for output in outputs:
        output.close()

    for pro in process:
        pro.terminate()

data_save_path = 'data/optimal'
pickle.dump(result, open(os.path.join(data_save_path, "optimal_latest.p"), 'wb'))



















































#end
