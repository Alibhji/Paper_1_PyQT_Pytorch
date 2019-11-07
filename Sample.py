# # import matplotlib.pyplot as plt
# # import numpy as np
# #
# #
# # fig=plt.figure(figsize=(15,5.5))
# # ax1=plt.subplot2grid((1,3),(0,0))
# # ax2=plt.subplot2grid((1,3),(0,1))
# # ax3=plt.subplot2grid((1,3),(0,2))
# #
# #
# # image=np.random.random_integers(1,10,size=(100,100))
# # cax=ax1.imshow(image,interpolation="none",aspect='equal')
# # cbar=fig.colorbar(cax,ax=ax1,orientation=u'horizontal')
# #
# # x=np.linspace(0,1)
# # ax2.plot(x,x**2)
# # ax3.plot(x,x**3)
# #
# # plt.show()
#
#
# import random
# import matplotlib.pyplot as plt
# from matplotlib import style
#
# style.use('fivethirtyeight')
#
# fig = plt.figure()
#
# def create_plots():
#     xs = []
#     ys = []
#
#     for i in range(10):
#         x = i
#         y = random.randrange(10)
#
#         xs.append(x)
#         ys.append(y)
#     return xs, ys
#
#
# ax1 = fig.add_subplot(221)
# ax2 = fig.add_subplot(222)
# ax3 = fig.add_subplot(212)


import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(2, 9))

gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_title('Varying Density')
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2:, :])

plt.tight_layout()
plt.show()