import pickle
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from functions import*
import matplotlib as mpl

def DataFormation(data):

    SW_1 = DataExtraction(data)

    return SW_1

#platform capacity
capacity = 500

#provider number unit & range
requester_num = 1000
provider_num = 2000

#zoomed-in setting
#data_path = 'data/parameter/parameter_result.p'

#macro range
data_path = 'data/parameter/new_parameter_result.p'
data = pickle.load(open(data_path, 'rb'))

#[mere_SW, expected_SW, realized_SW, pre_budget, post_budget, payment, fee, changed_payment, changed_fee, cost, running_time]
SW = DataFormation(data)
SW = DataTrimming(SW)

#zoomed-in
#power_unit = 0.1
power_unit = 0.2
power_range = 10
power4requester = np.arange(power_unit, power_range*power_unit + 0.05, power_unit)
power4provider = np.arange(power_unit, power_range*power_unit + 0.05, power_unit)

power4requester, power4provider = np.meshgrid(power4requester, power4provider)

#macro
x_tick = [0.5, 1.0, 1.5, 2.0]
y_tick = [0.5, 1.0, 1.5, 2.0]

#zoomed-in parameter
#x_tick = [0.2, 0.6, 1.0]
#y_tick = [0.2, 0.6, 1.0]

'''
#mere SW
#[mere_SW, expected_SW, realized_SW, pre_budget, post_budget, payment, fee, changed_payment, changed_fee, cost, running_time]
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.plot_surface(power4requester, power4provider, SW[:, :, 0], alpha = 0.5, cmap = cm.jet)
ax.set_xlabel(r'$\beta$ of $\alpha$', fontsize = 20, labelpad = 20)
ax.set_ylabel(r'$\beta$ of $\lambda$', fontsize = 20, labelpad = 20)
ax.set_zlabel(r'Social welfare (x$10^3$)', fontsize = 20, labelpad = 20)

z_tick = [27000, 28000, 29000, 30000]

ax.set_xticks(x_tick)
ax.set_xticklabels(x_tick, fontsize=20)

ax.set_yticks(y_tick)
ax.set_yticklabels(y_tick, fontsize=20)

ax.set_zticks(z_tick)
ax.set_zticklabels([int(z/1000) for z in z_tick], fontsize=20)

ax.view_init(azim = -135)
#plt.tight_layout()
plt.show()
'''
#expected SW
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.plot_surface(power4requester, power4provider, SW[:, :, 1], alpha = 0.5, cmap = cm.jet)
ax.set_xlabel(r'$\beta$ of $\alpha$', fontsize = 20, labelpad = 20)
ax.set_ylabel(r'$\beta$ of $\lambda$', fontsize = 20, labelpad = 20)
ax.set_zlabel(r'Social welfare (x$10^3$)', fontsize = 20, labelpad = 10)

#macro
z_tick = [5000, 10000, 15000, 20000]
#zoomed-in setting
#z_tick = [20000, 21000, 22000, 23000]

ax.set_xticks(x_tick)
ax.set_xticklabels(x_tick, fontsize=20)

ax.set_yticks(y_tick)
ax.set_yticklabels(y_tick, fontsize=20)

ax.set_zticks(z_tick)
ax.set_zticklabels([int(z/1000) for z in z_tick], fontsize=20)

ax.view_init(azim = -135)
#plt.tight_layout()
plt.show()

'''
#realized SW
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.plot_surface(power4requester, power4provider, SW[:, :, 2], alpha = 0.5, cmap = cm.jet)
ax.set_xlabel(r'$\beta$ of $\alpha$', fontsize = 20, labelpad = 20)
ax.set_ylabel(r'$\beta$ of $\lambda$', fontsize = 20, labelpad = 20)
ax.set_zlabel(r'Social welfare (x$10^3$)', fontsize = 20, labelpad = 20)

z_tick = [20000, 21000, 22000, 23000]

ax.set_xticks(x_tick)
ax.set_xticklabels(x_tick, fontsize=20)

ax.set_yticks(y_tick)
ax.set_yticklabels(y_tick, fontsize=20)

ax.set_zticks(z_tick)
ax.set_zticklabels([int(z/1000) for z in z_tick], fontsize=20)

ax.view_init(azim = -135)
plt.tight_layout()
plt.show()
'''

#Platform Utility: before submission
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.plot_surface(power4requester, power4provider, SW[:, :, 3], alpha = 0.5, cmap = cm.jet)
ax.set_xlabel(r'$\beta$ of $\alpha$', fontsize = 20, labelpad = 20)
ax.set_ylabel(r'$\beta$ of $\lambda$', fontsize = 20, labelpad = 20)
ax.set_zlabel(r'Platform Utility (x$10^3$)', fontsize = 20, labelpad = 10)

#zoomed-in setting
z_tick = [4000, 8000, 12000]

ax.set_xticks(x_tick)
ax.set_xticklabels(x_tick, fontsize=20)

ax.set_yticks(y_tick)
ax.set_yticklabels(y_tick, fontsize=20)
ax.set_zticklabels([int(z/1000) for z in z_tick], fontsize=20)

ax.set_zticks(z_tick)

ax.view_init(azim = 45)
#plt.tight_layout()
plt.show()

'''
#Platform Utility: after submission
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.plot_surface(power4requester, power4provider, SW[:, :, 4], alpha = 0.5, cmap = cm.jet)
ax.set_xlabel(r'$\beta$ of $\alpha$', fontsize = 20, labelpad = 20)
ax.set_ylabel(r'$\beta$ of $\lambda$', fontsize = 20, labelpad = 20)
ax.set_zlabel(r'Platform Utility (x$10^3$)', fontsize = 20, labelpad = 20)

z_tick = [2000, 4000, 6000, 8000]

ax.set_xticks(x_tick)
ax.set_xticklabels(x_tick, fontsize=20)

ax.set_yticks(y_tick)
ax.set_yticklabels(y_tick, fontsize=20)

ax.set_zticks(z_tick)
ax.set_zticklabels([int(z/1000) for z in z_tick], fontsize=20)
ax.view_init(azim = 45)
#plt.tight_layout()
plt.show()
'''

#AVG Requester Utility
fig = plt.figure()
ax = fig.gca(projection = '3d')
#print((SW_2[:, :, 2] + SW_2[:, :, -2] - SW_2[:, :, -3]))
ax.plot_surface(power4requester, power4provider, (SW[:, :, 2] + SW[:, :, -2] - SW[:, :, -3]) / requester_num, alpha = 0.5, cmap = cm.jet)
ax.set_xlabel(r'$\beta$ of $\alpha$', fontsize = 20, labelpad = 20)
ax.set_ylabel(r'$\beta$ of $\lambda$', fontsize = 20, labelpad = 20)
ax.set_zlabel(r'Requester Utility', fontsize = 20, labelpad = 10)

#macro
z_tick = [4, 8, 12, 16]

#zoomed-in setting
#z_tick = [12, 14, 16]

ax.set_xticks(x_tick)
ax.set_xticklabels(x_tick, fontsize=20)

ax.set_yticks(y_tick)
ax.set_yticklabels(y_tick, fontsize=20)

ax.set_zticks(z_tick)
ax.set_zticklabels(z_tick, fontsize=20)

ax.view_init(azim = -135)
#plt.tight_layout()
plt.show()

#AVG Provider Utility
fig = plt.figure()
ax = fig.gca(projection = '3d')
#print((SW_2[:, :, 2] + SW_2[:, :, -2] - SW_2[:, :, -3]))
ax.plot_surface(power4requester, power4provider, (SW[:, :, -4] - SW[:, :, -2]) / provider_num, alpha = 0.5, cmap = cm.jet)
ax.set_xlabel(r'$\beta$ of $\alpha$', fontsize = 20, labelpad = 20)
ax.set_ylabel(r'$\beta$ of $\lambda$', fontsize = 20, labelpad = 20)
ax.set_zlabel(r'Provider Utility', fontsize = 20, labelpad = 10)

#macro
z_tick = [1, 2, 3, 4]
#zoomed-in setting
#z_tick = [0.4, 1.2, 2.0]

ax.set_xticks(x_tick)
ax.set_xticklabels(x_tick, fontsize=20)

ax.set_yticks(y_tick)
ax.set_yticklabels(y_tick, fontsize=20)

ax.set_zticks(z_tick)
ax.set_zticklabels(z_tick, fontsize=20)

ax.view_init(azim = -135)
#plt.tight_layout()
plt.show()




































#end
