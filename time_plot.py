import matplotlib.pyplot as plt
import numpy as np

# f0_num      = [2, 20, 50, 100, 150, 200, 250, 275]
# solo_t_p100 = [0.00839, 0.05095, 0.12815, 0.2635, 0.4004, 0.5348, 0.6862, 0.7616]
# solo_t_rtx  = [0.00819, 0.03674, 0.08326, 0.1641, 0.2554, 0.3424, 0.4328, 0.4876]
# multi_pipe  = [0.00911, 0.03756, 0.08129, 0.1637, 0.2521, 0.3402, 0.4332, 0.4841]
# multi_queue = [0.00826, 0.03832, 0.08249, 0.1648, 0.2520, 0.3417, 0.4358, 0.4864]
# split_var   = [0.00906, 0.05142, 0.12882, 0.2618, 0.4011, 0.5284, 0.6867, 0.7582]

# plt.figure(dpi=320)
# plt.plot(f0_num, solo_t_p100, marker='o', label='Tesla P100')
# # plt.plot(f0_num, solo_t_rtx, marker='s', label='RTX 2080')
# # plt.plot(f0_num, multi_pipe, marker='D', label='Multi GPUs(Pipe)')
# # plt.plot(f0_num, multi_queue, marker='^', label='Multi GPUs(Queue)')
# plt.plot(f0_num, split_var, marker='P', label='Multi GPUs(args split)')
# plt.xlabel('Virtual Resonance Number')
# plt.ylabel('Gradient Calculating Time')
# plt.xticks(f0_num)
# plt.legend()
# plt.grid()
# plt.savefig('cal_time_1.png')

# plt.figure(dpi=320)
# # plt.plot(f0_num, solo_t_p100, marker='o', label='Tesla P100')
# plt.plot(f0_num, solo_t_rtx, marker='s', label='RTX 2080')
# plt.plot(f0_num, multi_pipe, marker='D', label='Multi GPUs(Pipe)')
# plt.plot(f0_num, multi_queue, marker='^', label='Multi GPUs(Queue)')
# # plt.plot(f0_num, split_var, marker='P', label='Multi GPUs(args split)')
# plt.xlabel('Virtual Resonance Number')
# plt.ylabel('Gradient Calculating Time')
# plt.xticks(f0_num)
# plt.legend()
# plt.grid()
# plt.savefig('cal_time_2.png')


f0_num      = [2, 4, 6]
solo_t_p100 = [0.0658, 0.1908, 0.3691]
solo_t_rtx  = [0.0530, 0.1512, 0.3039]
multi       = [0.0637, 0.1857, 0.3726]
split_var   = [0.0727, 0.1973, 0.3829]

plt.figure(figsize=(12,6),dpi=320)
# plt.grid()
bar_x = np.arange(3)*2+1
plt.bar(bar_x,      solo_t_p100,    width=0.35, edgecolor='#8470FF',label='Tesla P100')
plt.bar(bar_x+0.4,  solo_t_rtx,     width=0.35, edgecolor='#8470FF',label='RTX 2080')
plt.bar(bar_x+0.8,  multi,     width=0.35, edgecolor='#8470FF',label='Multi GPUs(Pipe)')
# plt.bar(bar_x+1.2,  multi_queue,    width=0.35, edgecolor='#8470FF',label='Multi GPUs(Queue)')
plt.bar(bar_x+1.2,  split_var,      width=0.35, edgecolor='#8470FF',label='Multi GPUs(args split)')
plt.xlabel('Virtual Resonance Number')
plt.ylabel('Hessian Matrix Calculating Time')
plt.xticks(bar_x+0.6,['2', '4', '6'])
plt.legend()
plt.savefig('cal_time_bar_hessian.png')