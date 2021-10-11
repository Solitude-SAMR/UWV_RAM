import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

plt.style.use('seaborn-talk')

np.random.seed(9)

op_all = np.load('op_cell.npy')
pfd_all = np.load('pfd_cell.npy')

n = len(pfd_all)
op_all = op_all[:n]

pfd_true = sum(pfd_all * op_all/sum(op_all))

print('model pfd ground truth: ', pfd_true)
print('model acu ground truth: ',np.mean(pfd_all))


acu_list = []
pfd_list = []
pfd_var_list = []
acu_var_list = []
pfd_bound_list = []
acu_bound_list = []
t = range(100,n,100)

for k in t:
    idx = np.random.choice(n, k, replace=False)
    op = op_all[idx]
    pfd = pfd_all[idx]

    acu = np.mean(pfd)
    v1 = (sum(op))**2
    v2 = sum(op**2)
    n_eff = v1/v2
    pfd_model = sum(pfd * op)/sum(op)

    pfd_var = sum(op * (pfd - pfd_model)**2)/sum(op)/(n_eff-1)
    acu_var = sum((pfd-acu)**2)/((k-1)*k)

    pfd_bound = 1.96* sqrt(pfd_var)
    acu_bound = 1.96* sqrt(acu_var)


    acu_list.append(acu)
    pfd_list.append(pfd_model)
    pfd_var_list.append(pfd_var)
    acu_var_list.append(acu_var)
    pfd_bound_list.append(pfd_bound)
    acu_bound_list.append(acu_bound)

pfd_list = np.array(pfd_list)
pfd_bound_list = np.array(pfd_bound_list)
acu_list = np.array(acu_list)
acu_bound_list = np.array(acu_bound_list)

fig, ax1 = plt.subplots(figsize=(8, 6),constrained_layout=True)
plt.title(r'CIFAR10 ACU', fontsize=20)

# color = 'tab:red'
# ax1.set_xlabel('k')
# ax1.set_ylabel('Est. $\it{pmi}$', color=color)
# lns1 = ax1.plot(t, pfd_list, color=color, label = 'pmi')
# lns2 = ax1.fill_between(t, pfd_list, (pfd_list + pfd_bound_list), color='r', alpha=.1, label = 'confidence interval')
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('Variance', color=color)  # we already handled the x-label with ax1
# lns3 = ax2.plot(t, pfd_var_list, color=color, label = 'ACU')
# ax2.tick_params(axis='y', labelcolor=color)



# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# fig.savefig('weighted_mean.png', dpi = 300, bbox_inches = 'tight')


color = 'tab:red'
ax1.set_xlabel('k')
ax1.set_ylabel('Est. ACU', color=color)
ax1.plot(t, acu_list, color=color)
ax1.fill_between(t, acu_list, (acu_list+acu_bound_list), color='r', alpha=.1)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Variance', color=color)  # we already handled the x-label with ax1
ax2.plot(t, acu_var_list, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.savefig('ACU.png',dpi = 300, bbox_inches = 'tight')




# print('model ACU: ', acu)
# print('model pfd mean: ', pfd_model)
# print('model pfd variance: ', pfd_var)
# print('model 97.5 confidence upper bound: ',confidence_bound)
