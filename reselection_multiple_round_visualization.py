import numpy as np
import matplotlib.pyplot as plt

provider = np.load('data/reselection/distribution_provider_3.npy', allow_pickle = True)
requester = np.load('data/reselection/distribution_requester_3.npy', allow_pickle = True)
print(provider.shape)

plt.plot(provider[:, 0], 'r--o', markersize = 6, label = 'B.M')
plt.plot(provider[:, 1], 'g--^', markersize = 6, label = 'ESWM')
plt.hlines(0.5, xmin = 0, xmax = 100, linestyles = 'dashed')
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.ylim([0, 1])
plt.xlabel('Round Index', fontsize = 25)
plt.ylabel('Participation Probability', fontsize = 25)
#plt.legend(loc = 'upper right', fontsize = 21)
plt.tight_layout()
plt.show()

plt.plot(requester[:, 0], 'r--o', markersize = 6, label = 'B.M')
plt.plot(requester[:, 1], 'g--^', markersize = 6, label = 'ESWM')
plt.hlines(0.5, xmin = 0, xmax = 100, linestyles = 'dashed')
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.ylim([0, 1])
plt.xlabel('Round Index', fontsize = 25)
plt.ylabel('Participation Probability', fontsize = 25)
plt.legend(fontsize = 21)
plt.tight_layout()
plt.show()


























#end
