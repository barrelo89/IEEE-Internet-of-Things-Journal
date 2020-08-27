import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import*

#including optimal SW
def DataFormation(data):

    data = pd.DataFrame(data, columns = ['SW_1', 'SW_2', 'SW_3'])

    SW_1, SW_2, SW_3 = data['SW_1'], data['SW_2'], data['SW_3']

    SW_1 = DataExtraction(SW_1)
    SW_2 = DataExtraction(SW_2)
    SW_3 = DataExtraction(SW_3)

    return SW_1, SW_2, SW_3

requester_num_unit = 100
requester_range = 5

#data for all the parameter but running time: 'optimal_latest.p' for running time
data_path = 'data/optimal/optimal_result.p'#'sw1.p'
data = pickle.load(open(data_path, 'rb'))


#[mere_SW, expected_SW, realized_SW, pre_budget, post_budget, payment, fee, changed_payment, changed_fee, cost, running_time]
SW_1, SW_2, SW_3 = DataFormation(data)

#[mere_SW, expected_SW, realized_SW, pre_budget, post_budget, cost, running_time, payment, fee, changed_payment, changed_fee]
SW_1 = DataTrimming(SW_1)
SW_2 = DataTrimming(SW_2)
SW_3 = DataTrimming(SW_3)

#SW_1[0, 0] = 3.37294425e+03
#SW_1[0, 1] = 1.73334185e+03
#SW_1[0, 2] = 1.71322589e+03

#SW_2[0, 0] = 3.18677835e+03
#SW_2[0, 1] = 2.03498085e+03
#SW_2[0, 2] = 2.06463619e+03

'''
#mere SW
plt.plot(range(requester_num_unit, requester_num_unit*requester_range + 1, requester_num_unit), SW_1[:, 0], label = 'SW1: Mere')
plt.plot(range(requester_num_unit, requester_num_unit*requester_range + 1, requester_num_unit), SW_2[:, 0], label = 'SW2: Mere')
plt.plot(range(requester_num_unit, requester_num_unit*requester_range + 1, requester_num_unit), SW_3[:, 0], label = 'SW3: Mere')
#plt.ylim(0, 9000)
plt.xlabel('Number of Requesters', fontsize = 20)
plt.ylabel(r'Social Welfare (x$10^3$)', fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xticks(range(requester_num_unit, requester_num_unit*requester_range + 1, requester_num_unit), fontsize = 20)
plt.yticks(np.arange(0, 9100, 3000), np.arange(0, 91, 30), fontsize = 20)
plt.legend(loc = 4, fontsize = 15)
plt.tight_layout()
plt.show()
'''
#expected SW
#plt.plot(range(requester_num_unit, requester_num_unit*requester_range + 1, requester_num_unit), SW_1[:, 1], label = 'SW1: Expected')
plt.plot(range(requester_num_unit, requester_num_unit*requester_range + 1, requester_num_unit), SW_2[:, 1], '--o', markersize = 13, label = 'Greedy: Expected')
plt.plot(range(requester_num_unit, requester_num_unit*requester_range + 1, requester_num_unit), SW_3[:, 1], '--^', markersize = 13, label = 'Hungarian: Expected')
#plt.ylim(0, 9000)
plt.xlabel('Number of Requesters', fontsize = 20)
plt.ylabel(r'Social Welfare (x$10^3$)', fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xticks(range(requester_num_unit, requester_num_unit*requester_range + 1, requester_num_unit), fontsize = 20)
plt.yticks(np.arange(0, 9100, 3000), np.arange(0, 91, 30), fontsize = 20)
#plt.legend(loc = 4, fontsize = 15)
plt.tight_layout()
plt.show()

'''
#realized SW
plt.plot(range(requester_num_unit, requester_num_unit*requester_range + 1, requester_num_unit),SW_1[:, 2], label = 'SW1: Realized')
plt.plot(range(requester_num_unit, requester_num_unit*requester_range + 1, requester_num_unit),SW_2[:, 2], label = 'SW2: Realized')
plt.plot(range(requester_num_unit, requester_num_unit*requester_range + 1, requester_num_unit),SW_3[:, 2], label = 'SW3: Realized')
#plt.ylim(0, 9000)
plt.xlabel('Number of Requesters', fontsize = 15)
plt.ylabel(r'Social Welfare (x$10^3$)', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xticks(range(requester_num_unit, requester_num_unit*requester_range + 1, requester_num_unit), fontsize = 20)
plt.yticks(np.arange(0, 9100, 3000), np.arange(0, 91, 30), fontsize = 20)
plt.legend(loc = 4, fontsize = 15)
plt.tight_layout()
plt.show()
'''

#running time
#plt.plot(range(requester_num_unit, requester_num_unit*requester_range + 1, requester_num_unit), (SW_1[:, -1] / 3600), label = 'SW1: Running Time')
print(SW_2[:, -1])
print(SW_3[:, -1])
plt.plot(range(requester_num_unit, requester_num_unit*requester_range + 1, requester_num_unit), np.log10(SW_2[:, -1]), '--o', markersize = 13, label = 'Greedy')
plt.plot(range(requester_num_unit, requester_num_unit*requester_range + 1, requester_num_unit), np.log10(SW_3[:, -1]), '--^', markersize = 13, label = 'Hungarian')
plt.hlines(y = 0, xmin = 100, xmax = 500, colors = 'k', linestyles = 'dashed')
plt.xlabel('Number of Requesters', fontsize = 20)
plt.ylabel(r'$log_{10}$(Running Time) (s)', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xticks(range(requester_num_unit, requester_num_unit*requester_range + 1, requester_num_unit), fontsize = 20)
plt.yticks(range(-1, 5), fontsize = 20)
plt.legend(loc = 'best', fontsize = 20)
plt.tight_layout()
plt.show()






































#end
