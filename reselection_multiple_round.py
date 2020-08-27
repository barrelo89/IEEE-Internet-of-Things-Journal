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
from sklearn.utils import shuffle

def DataFormation(data):

    data = pd.DataFrame(data, columns = ['SW_1', 'SW_2'])

    SW_1, SW_2 = data['SW_1'], data['SW_2']

    SW_1 = DataExtraction(SW_1)
    SW_2 = DataExtraction(SW_2)

    return SW_1, SW_2

def DistributionCalculator(data, num_requester, num_provider):

    SW_1, SW_2 = DataFormation(data)

    SW_1 = DataTrimming(SW_1)
    SW_2 = DataTrimming(SW_2)


    #avg requester utility
    av_r1 = (SW_1[:, 2] + SW_1[:, -2] - SW_1[:, -3]) / num_requester[0]
    av_r2 = (SW_2[:, 2] + SW_2[:, -2] - SW_2[:, -3]) / num_requester[1]

    #avg provider utility
    av_p1 = (SW_1[:, -4] - SW_1[:, -2]) / num_provider[0]
    av_p2 = (SW_2[:, -4] - SW_2[:, -2]) / num_provider[1]

    av_r1 = np.mean(av_r1[av_r1.nonzero()])
    av_r2 = np.mean(av_r2[av_r2.nonzero()])

    av_p1 = np.mean(av_p1[av_p1.nonzero()])
    av_p2 = np.mean(av_p2[av_p2.nonzero()])

    print(av_r1, av_r2, av_p1, av_p2)

    p_distribution = np.array([np.sqrt(av_p1), np.sqrt(av_p2)])
    r_distribution = np.array([np.sqrt(av_r1), np.sqrt(av_r2)])

    requester_distribution = r_distribution / sum(r_distribution)
    provider_distribution = p_distribution / sum(p_distribution)

    return requester_distribution, provider_distribution

def num_participants(total_requester_num, total_provider_num, requester_distribution, provider_distribution):

    req_sampling = []
    pro_sampling = []

    for _ in range(total_requester_num):
        req_sampling.append(np.random.choice(['platform1', 'platform2'], p = requester_distribution))

    for _ in range(total_provider_num):
        pro_sampling.append(np.random.choice(['platform1', 'platform2'], p = provider_distribution))

    req_sampling = np.array(req_sampling)
    pro_sampling = np.array(pro_sampling)

    num_requester = [sum(req_sampling == 'platform1'), sum(req_sampling == 'platform2')]
    num_provider = [sum(pro_sampling == 'platform1'), sum(pro_sampling == 'platform2')]

    num_requester = np.array(num_requester)
    num_provider = np.array(num_provider)

    return num_requester, num_provider

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

def SW(tasks, providers, capacity_unit, capacity_range, power4requester, power4provider, num_requester, num_provider, output):

    tasks = shuffle(tasks)
    providers = shuffle(providers)

    #mere_SW, expected_SW, realized_SW, pre_budget, post_budget, payment, fee, changed_payment, changed_fee, cost, running_time
    greedy_result = SW1(tasks[:num_requester[0]], providers[:num_provider[0]], capacity_unit, capacity_range)
    #print('Greedy Complete!')
    proposed_result = SW2(tasks[num_requester[0]:], providers[num_provider[0]:], capacity_unit, capacity_range, power4requester, power4provider)
    #print('Proposed Complete!')

    output.put([greedy_result, proposed_result])

#task information
max_value = 100
max_deadline = 100
max_task_size = 10
max_alpha = 100

#provider information
max_provider_bid = 10
max_mu = 1.5
time_unit = 0.1

#number of cores to use
num_core = multiprocessing.cpu_count() - 2

#power values for providers and requesters
power4requester, power4provider = 0.5, 0.5

#iteration
num_iter = 5
#number of auction round
num_round = 100

#num of requesters and providers in the previous simulation
num_requester = 1000
num_provider = 2000

#num of requesters and providers in this simulation
total_requester_num = 2000
total_provider_num = 4000

#platform capacity
capacity_unit = 100
capacity_range = 10

#calculate the reselection probability
data_path = 'data/reselection/new_capacity_result.p'
result = pickle.load(open(data_path, 'rb'))

pro_probability_list = []
req_probability_list = []

pro_probability_list.append([0.5, 0.5])
req_probability_list.append([0.5, 0.5])

requester_distribution, provider_distribution = DistributionCalculator(result, [num_requester, num_requester], [num_provider, num_provider])
print("Round 0")
print("Provider Distribution:", provider_distribution)
print("Requester Distribution:", requester_distribution)
pro_probability_list.append(provider_distribution)
req_probability_list.append(requester_distribution)

for round in range(1, num_round + 1):

    print("Round {}: Running".format(round))

    result = []
    num_requester_array = []
    num_provider_array = []

    for _ in range(num_iter):

        outputs = []
        process = []

        num_requester, num_provider = num_participants(total_requester_num, total_provider_num, requester_distribution, provider_distribution)

        num_requester_array.append(num_requester)
        num_provider_array.append(num_provider)

        for _ in range(num_core):
            outputs.append(multiprocessing.Queue())

        for output in outputs:
            #create requesters and providers
            tasks = TaskCreator(total_requester_num, max_value, max_alpha, max_deadline, max_task_size)
            providers = ProviderCreator(total_provider_num, max_mu, max_provider_bid, time_unit)

            process.append(multiprocessing.Process(target = SW, args = (tasks, providers, capacity_unit, capacity_range, power4requester, power4provider, num_requester, num_provider, output)))

        for pro in process:
            pro.start()

        for output in outputs:
            result.append(output.get())

        for output in outputs:
            output.close()

        for pro in process:
            pro.terminate()

    num_requester = np.array(num_requester_array).mean(axis = 0)
    num_provider = np.array(num_provider_array).mean(axis = 0)

    requester_distribution, provider_distribution = DistributionCalculator(result, num_requester, num_provider)
    pro_probability_list.append(provider_distribution)
    req_probability_list.append(requester_distribution)
    print("Provider Distribution:", provider_distribution)
    print("Requester Distribution:", requester_distribution)

pro_probability_list = np.array(pro_probability_list)
req_probability_list = np.array(req_probability_list)

np.save('data/reselection/distribution_provider_3.npy', pro_probability_list)
np.save('data/reselection/distribution_requester_3.npy', req_probability_list)






















































#end
