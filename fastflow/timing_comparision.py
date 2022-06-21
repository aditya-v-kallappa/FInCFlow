import numpy as np
import matplotlib.pyplot as plt
import json

fastflow_params = ['12.79M', '39.32M']
macow_params = ['12.35M', '38.59M']
emerging_params = ['12.86M', '37.87M']
fastflow_params = ['12.86M', '39.47M']

fastflow_timings = [
    [0.05727847329999989, 0.06612396699999952, 0.08215905969999984, 0.11489676929999995, 0.17220211640000027,
    0.29212771209999955,  0.5724227276999997],
    [0.1746840266999996, 0.20273688460000017, 0.2525629038000005, 0.35165285309999916, 0.5279688547000007, 
    0.8938244006000033, 1.7693690774000033]
]
macow_timings = [
    [0.47171502113342284, 0.6066135406494141, 0.7501152276992797, 1.034312629699707, 1.3358250379562377, 
    1.955954670906067, 2.6227743148803713],
    [1.6704790115356445, 2.1545565843582155, 2.660679578781128, 3.6806631326675414, 4.758614945411682, 
    6.932291150093079, 9.377770805358887]
]
emerging_timings = [
    [0.3799474028000004, 0.7334945429999997, 1.4667077527999997, 2.929962100299997, 5.921108602500001, 
    11.984675121300004, 24.07182260809999],
    [1.1724421088000003, 2.2223494578000005, 4.4249174682999985, 8.816962767699996, 17.8099431735, 
    35.8138566002, 72.2996635002]
]
cinc_timings = [
    [0.12350185459999992, 0.5274108111, 1.0856701186, 1.7050097441000005, 2.9588237096999976, 
    5.642688028500002, 11.034693004400003],
    [0.35932010780001067, 1.6497812744999976, 3.3542549741000074, 5.313616497599997, 9.204100106400006, 
    17.637160505600015, 34.14620475559998]
]

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
ff_timings = [np.load(f'numpy_files/fastflow_{b}.npy') for b in batch_sizes]
print(ff_timings)
# x_axis = range(1, 8)
# # fig = plt.figure()
# # gs = fig.add_gridspec(2, 1, hspace=3, wspace=3)
# # ax = gs.subplots(2)
# plt.subplots_adjust(left=20, bottom=20, right=30, top=30, wspace=10, hspace=10)
# fig, ax = plt.subplots(2, figsize=(5, 5))
# fig.tight_layout()
# # ax[0].set_xticks(['(8, 8)', '(8, 16)', '(16, 16)', '(16, 32)', '(32, 32)', '(32, 64)', '(64, 64)'])
# ax[0].set_title('#Params ~ 12.5M')
# ax[0].set_xticks(x_axis)
# ax[0].set_yticks([1, 10, 20, 50])
# ax[0].plot(x_axis, fastflow_timings[0], color='r', label='FastFlow')
# ax[0].plot(x_axis, macow_timings[0], color='b', label='MACOW')
# ax[0].plot(x_axis, cinc_timings[0], color='cyan', label='CInC Flow')
# ax[0].plot(x_axis, emerging_timings[0], color='g', label='Emerging')
# ax[0].legend()
# ax[0].set_xlabel('Image Size')
# ax[0].set_ylabel('Time in sec')

# ax[1].set_title('#Params ~ 38.5M')
# ax[1].set_xticks(x_axis)
# ax[1].set_yticks([1, 10, 20, 50, 75])
# ax[1].plot(x_axis, fastflow_timings[1], color='r', label='FastFlow')
# ax[1].plot(x_axis, macow_timings[1], color='b', label='MACOW')
# ax[1].plot(x_axis, cinc_timings[1], color='cyan', label='CInC Flow')
# ax[1].plot(x_axis, emerging_timings[1], color='g', label='Emerging')
# ax[1].legend()
# ax[1].set_xlabel('Image Size')
# ax[1].set_ylabel('Time in sec')




# plt.savefig('timings.png')
# plt.plot()