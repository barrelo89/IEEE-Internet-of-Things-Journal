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

def SW1(tasks, providers, capacity_unit, capacity_range): #simple greedy approach

    result = []

    for idx in range(1, capacity_range + 1):

        #create an auctioneer
        capacity = capacity_unit*idx
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
        if pre_budget < 0: #budget balance not met. Note that we can only check the budget balance before providers' actual submission
            result.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, running_time]) #return all zeros
        else: #budget balance met
            result.append([mere_SW, expected_SW, realized_SW, pre_budget, post_budget, payment.sum(), fee.sum(), changed_payment.sum(), changed_fee.sum(), cost, running_time])

    return np.array(result)

def SW2(tasks, providers, capacity_unit, capacity_range, power4requester, power4provider):

    result = []

    for idx in range(1, capacity_range + 1):

        #create an auctioneer
        capacity = capacity_unit*idx
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
        if pre_budget < 0: #budget balance not met
            result.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, running_time]) #return all zeros
        else: #budget balance met
            result.append([mere_SW, expected_SW, realized_SW, pre_budget, post_budget, payment.sum(), fee.sum(), changed_payment.sum(), changed_fee.sum(), cost, running_time])#proposed heuristic approach

    return np.array(result)

def SW(tasks, providers, capacity_unit, capacity_range, power4requester, power4provider, output):

    #mere_SW, expected_SW, realized_SW, pre_budget, post_budget, payment, fee, changed_payment, changed_fee, cost, running_time
    greedy_result = SW1(tasks, providers, capacity_unit, capacity_range,)
    print('Greedy Complete!')
    proposed_result = SW2(tasks, providers, capacity_unit, capacity_range, power4requester, power4provider)
    print('Proposed Complete!')

    output.put([greedy_result, proposed_result])

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

#requester & provider number
total_requester_num = 1000
total_provider_num = total_requester_num*2

#platform capacity
capacity_unit = 100
capacity_range = 10

result = []

for _ in range(num_iter):

    outputs = []
    process = []

    for _ in range(num_core):
        outputs.append(multiprocessing.Queue())

    for output in outputs:
        #create requesters and providers
        tasks = TaskCreator(total_requester_num, max_value, max_alpha, max_deadline, max_task_size)
        providers = ProviderCreator(total_provider_num, max_mu, max_provider_bid, time_unit)

        process.append(multiprocessing.Process(target = SW, args = (tasks, providers, capacity_unit, capacity_range, power4requester, power4provider, output)))

    for pro in process:
        pro.start()

    for output in outputs:
        result.append(output.get())

    for output in outputs:
        output.close()

    for pro in process:
        pro.terminate()

data_save_path = 'data/capacity_result'
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)
pickle.dump(result, open(os.path.join(data_save_path, "new_capacity_result.p"), 'wb'))



















































#end
