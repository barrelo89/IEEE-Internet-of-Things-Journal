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

def SW2(tasks, providers, capacity, power_unit, power_range):

    result = []

    #create an auctioneer
    auctioneer = Platform(capacity)

    for req_idx in range(1, power_range + 1):

        power4requester = req_idx*power_unit

        req_result = []

        for pro_idx in range(1, power_range + 1):

            power4provider = pro_idx*power_unit

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
            if pre_budget < 0: #budget balance not met
                req_result.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, running_time]) #return all zeros
            else: #budget balance met
                req_result.append([mere_SW, expected_SW, realized_SW, pre_budget, post_budget, payment.sum(), fee.sum(), changed_payment.sum(), changed_fee.sum(), cost, running_time])#proposed heuristic approach

        result.append(req_result)

    print(np.array(result).shape)
    return result

def SW(capacity, total_requester_num, total_provider_num, power_unit, power_range, output):

    tasks = TaskCreator(total_requester_num, max_value, max_alpha, max_deadline, max_task_size)
    providers = ProviderCreator(total_provider_num, max_mu, max_provider_bid, time_unit)

    #mere_SW, expected_SW, realized_SW, pre_budget, post_budget, payment, fee, changed_payment, changed_fee, cost, running_time
    proposed_result = SW2(tasks, providers, capacity, power_unit, power_range)
    print('Proposed Complete!')

    output.put(proposed_result)
    #proposed_cube.append(proposed_result)
    #output.put(np.array(proposed_cube))

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

#iteration
num_iter = 2

#requester & provider number
total_requester_num = 1000
total_provider_num = total_requester_num*2

#platform capacity
capacity = 500

#power values for providers and requesters
power_unit = 0.2#0.1
power_range = 10


result = []

for _ in range(num_iter):

    outputs = []
    process = []

    for _ in range(num_core):
        outputs.append(multiprocessing.Queue())

    for output in outputs:

        process.append(multiprocessing.Process(target = SW, args = (capacity, total_requester_num, total_provider_num, power_unit, power_range, output)))

    for pro in process:
        pro.start()

    for output in outputs:
        result.append(output.get())

    for output in outputs:
        output.close()

    for pro in process:
        pro.terminate()

pickle.dump(result, open("new_parameter_result.p", 'wb'))



















































#end
