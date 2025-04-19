import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

a = 1
b = 5
def f(x):
   return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2

def jac(x):
   dx1 = -2 * a + 4 * b * x[0] ** 3 - 4 * b * x[0] * x[1] + 2 * x[0]
   dx2 = 2 * b * (x[1] - x[0] ** 2)
   return np.array([dx1, dx2])

def hess(x):
   dx1dx1 = 12 * b * x[0] ** 2 - 4 * b * x[1] + 2
   dx1dx2 = dx2dx1 = -4 * b * x[0]
   dx2dx2 = 2 * b
   return np.array([[dx1dx1, dx1dx2], [dx2dx1, dx2dx2]])


def penalty(x):
    #return -100 * gelu(-1 * ((1 - x[0]) ** 2 + (1 - x[1]) ** 2 - .1 ** 2))
    #return -100 * np.maximum(-1 * ((1 - x[0]) ** 2 + (1 - x[1]) ** 2 - .1 ** 2), 0)
    return 100 * np.minimum((1 - x[0]) ** 2 + (1 - x[1]) ** 2 - .1 ** 2, 0)

def penaltyJ(x):
   pen = penalty(x)
   if pen:
      jx = 2 * (1 - x[0])
      jy = 2 * (1 - x[1])
   else:
      jx = 0
      jy = 0
   return 100 * np.array([jx, jy])

def penaltyH(x):
   pen = penalty(x)
   if pen:
      jx = -2
      jy = -2
   else:
      jx = 0
      jy = 0
   return 100 * np.array([[jx, 0], [0, jy]])

def objective(x):
   return f(x) + penalty(x)

def obj_J(x, conmult=1):
   return jac(x) + conmult * penaltyJ(x) 

def obj_H(x):
   return hess(x) + penaltyH(x) 

def jac_sgd(x, C=1): return obj_J(x, C) * np.random.uniform(0.5, 1.5, 2)

def H_sgd(x): return obj_H(x) * np.random.uniform(0.5, 1.5, 2)

x = np.array([0.1, 0.1])
x_sgd = x.copy()
j_sgd = jac_sgd(x_sgd) 


nx = 111
x_grid = np.linspace(-2, 2, nx)
xx, yy = np.meshgrid(x_grid, x_grid)
x_plot = np.array([xx, yy]).reshape(2, -1).T
#evals = np.array([f(x) for x in x_plot]).reshape(nx, nx)
evals = f([xx, yy]) + 1e-12
#plt.contour(xx, yy, evals20, 9)
arrowscale = 2
L = 0.1
learning_rate = L / 5
l0 = learning_rate
N = 500
x_hist = np.zeros((N, 2))
x_hist_sgd = np.zeros((N, 2))
f_hist_sgd = np.zeros((N, 2))
c_hist_sgd = np.zeros((N, 1))
m = np.zeros(x.size)
v = np.zeros(x.size)
sv = np.zeros(x.size)
mass = 0.8
beta1=0.1
beta2=0.2
eps=1e-8
#delta = 0
delta = 1e-4
#delta = 1e-4
for ii in range(N):
   x_hist[ii] = x
   x_hist_sgd[ii] = x_sgd
   f_hist_sgd[ii] = f(x_sgd)
   c_hist_sgd[ii] = -1 * penalty(x_sgd)
   fig, ax = plt.subplots(2)
   ax[0].contour(xx, yy, np.log(evals), 8)
   ax[0].scatter(*x_sgd, c='pink', zorder=3)
   ax[0].scatter(x_hist_sgd[:ii + 1, 0], x_hist_sgd[:ii + 1, 1], c='pink', alpha=0.1, zorder=2)
   #plt.scatter(*x, c='orange', zorder=2)
   #plt.scatter(x_hist[:ii + 1, 0], x_hist[:ii + 1, 1], c='orange', alpha=0.1, zorder=2)
   #plt.scatter([1], [1], c='red', s=100, zorder=2, marker='x')
   ax[0].add_patch(plt.Circle((1, 1), 0.1, edgecolor='r', facecolor='none', zorder=3, ls='--'))
   ax[0].set_title('obj: %.1e, con: %.1e, learning rate %.1e' % (f(x_sgd), penalty(x_sgd), learning_rate))
   #plt.arrow(x[0], x[1], arrowscale * j[0], arrowscale * j[1], zorder=2)
   #ax[0].arrow(x_sgd[0], x_sgd[1], arrowscale * j_sgd[0], arrowscale * j_sgd[1], zorder=2)
   #ax[0].arrow(x_sgd[0], x_sgd[1], arrowscale * v[0], arrowscale * v[1], zorder=2, ec='pink')
   #ax[0].arrow(x_sgd[0], x_sgd[1], arrowscale * m[0], arrowscale * m[1], zorder=2, ec='lightblue')
   ax[1].plot(range(ii+1), f_hist_sgd[:ii+1])
   if c_hist_sgd.max() > 0:
      ax2 = ax[1].twinx()
      ax2.plot(range(ii+1), c_hist_sgd[:ii+1], c='r', ls='--')
      ax2.set_yscale('log')
      ax2.set_ylim(1e-4, c_hist_sgd.max() + 1)
   ax[1].set_yscale('log')
   ax[1].axhline(0.00187446, c='w', ls='--')
   plt.savefig('adam_base_%i.png' % ii)
   plt.clf()
   plt.close('all')
   m = (1 - beta1) * j_sgd + beta1 * m  # first  moment estimate.
   v = (1 - beta2) * (j_sgd**2) + beta2 * v  # second moment estimate.
   mhat = m / (1 - beta1**(ii + 1))  # bias correction.
   vhat = v / (1 - beta2**(ii + 1))
   x_sgd = x_sgd - learning_rate * mhat / (np.sqrt(vhat) + eps)
   #j_sgd = obj_J(x_sgd)
   #learning_rate *= 1 / (1 + delta * ii)
   j_sgd = jac_sgd(x_sgd)
   #j_sgd = jac_sgd(x_sgd, l0 / learning_rate)
   #H = penaltyH(x_sgd)
   #if np.sum(H ** 2) > 0:
   #   j_sgd = ( np.linalg.inv(H)).dot(jac_sgd(x_sgd))
   #else:
   #   j_sgd = jac(x_sgd)
   #x -= learning_rate * v
