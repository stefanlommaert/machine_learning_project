import numpy as np
import matplotlib.pyplot as plt

def dF(r, theta):
    return 0.5*(r - r**3), 1

X, Y = np.meshgrid(np.linspace(0.0, 1.0, 50), np.linspace(0.0, 1.0, 50))
u, v = np.zeros_like(X), np.zeros_like(X)
NI, NJ = X.shape
for i in range(NI):
    for j in range(NJ):
        x, y = X[i, j], Y[i, j]
        print((x,y))
        r, theta = (x**2 + y**2)**0.5, np.arctan2(y, x)
        fp = dF(r, theta)
        u[i,j] = (r + fp[0]) * np.cos(theta + fp[1]) - x
        v[i,j] = (r + fp[0]) * np.sin(theta + fp[1]) - y

plt.streamplot(X, Y, u, v)
plt.axis('square')
plt.axis([0, 1, 0, 1])
plt.show()
