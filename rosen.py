import numpy as np
import matplotlib.pyplot as plt
from py_wake.utils.gradients import fd, cs, autograd
plt.style.use('dark_background')

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


a = 1
b = 5
def f(x):
   return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2

def penalty(x, realmax=False):
    if realmax: return 100 * np.maximum((.1 ** 2 - (1 - x[0]) ** 2 - (1 - x[1]) ** 2), 0)
    else: return gelu(100 * (.1 ** 2 - (1 - x[0]) ** 2 - (1 - x[1]) ** 2))

def objective(x):
   return f(x) + penalty(x)

obj_J = cs(objective)

def jac_sgd(x): return obj_J(x) * np.random.uniform(0.5, 1.5, 2)

x = np.array([.19, .19])
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
N = 200
x_hist = np.zeros((N, 2))
x_hist_sgd = np.zeros((N, 2))
m = np.zeros(x.size)
v = np.zeros(x.size)
sv = np.zeros(x.size)
mass = 0.8
beta1=0.9
beta2=0.999
eps=1e-8
for ii in range(N):
   x_hist[ii] = x
   x_hist_sgd[ii] = x_sgd
   fig, ax = plt.subplots()
   ax.contour(xx, yy, np.log(evals), 8)
   ax.scatter(*x_sgd, c='pink', zorder=3)
   ax.scatter(x_hist_sgd[:ii + 1, 0], x_hist_sgd[:ii + 1, 1], c='pink', alpha=0.1, zorder=2)
   #plt.scatter(*x, c='orange', zorder=2)
   #plt.scatter(x_hist[:ii + 1, 0], x_hist[:ii + 1, 1], c='orange', alpha=0.1, zorder=2)
   #plt.scatter([1], [1], c='red', s=100, zorder=2, marker='x')
   ax.add_patch(plt.Circle((1, 1), 0.1, color='r', zorder=3))
   ax.set_title('obj: %.1e, con: %.1e' % (f(x_sgd), penalty(x_sgd, realmax=True)))
   #plt.arrow(x[0], x[1], arrowscale * j[0], arrowscale * j[1], zorder=2)
   ax.arrow(x_sgd[0], x_sgd[1], arrowscale * j_sgd[0], arrowscale * j_sgd[1], zorder=2)
   ax.arrow(x_sgd[0], x_sgd[1], arrowscale * v[0], arrowscale * v[1], zorder=2, ec='pink')
   ax.arrow(x_sgd[0], x_sgd[1], arrowscale * m[0], arrowscale * m[1], zorder=2, ec='lightblue')
   plt.savefig('adam_ex_%i' % ii)
   plt.clf()
   m = (1 - beta1) * j_sgd + beta1 * m  # first  moment estimate.
   v = (1 - beta2) * (j_sgd**2) + beta2 * v  # second moment estimate.
   mhat = m / (1 - beta1**(ii + 1))  # bias correction.
   vhat = v / (1 - beta2**(ii + 1))
   x_sgd = x_sgd - learning_rate * mhat / (np.sqrt(vhat) + eps)
   #j_sgd = obj_J(x_sgd)
   j_sgd = jac_sgd(x_sgd)
   #H = penaltyH(x_sgd)
   #if np.sum(H ** 2) > 0:
   #   j_sgd = ( np.linalg.inv(H)).dot(jac_sgd(x_sgd))
   #else:
   #   j_sgd = jac(x_sgd)
   #x -= learning_rate * v
