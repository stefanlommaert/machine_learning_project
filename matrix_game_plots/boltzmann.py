#from matplotlib import pyplot as plt
#imports zijn wel nodig om de projections in de registers van pyplot te steken (dus niet aankomen ;) )
#meer voorbeelden :=> open_spiel/open_spiel/python/egt/visualization_test.py 
import pyspiel
import numpy as np
from open_spiel.python.egt import dynamics
from open_spiel.python.egt.utils import game_payoffs_array
from open_spiel.python.egt import visualization
from matplotlib.figure import Figure
from matplotlib.quiver import Quiver
import matplotlib.pyplot as plt


# plot_name = 'Dispersion game'
# row_player = [[-1,1],[1,-1]]
# vector_player = [[-1,1],[1,-1]]

# plot_name = 'matching pennies'
# row_player = [[1,-1],[-1,1]]
# vector_player = [[-1,1],[1,-1]]

plot_name = 'Battle of the sexes'
row_player = [[3,0],[0,2]]
vector_player = [[2,0],[0,3]]

# plot_name = 'Subsidy game'
# row_player = [[10,0],[11,12]]
# vector_player = [[10,11],[0,12]]

a11 = row_player[0][0]
a12 = row_player[0][1]
a21 = row_player[1][0]
a22 = row_player[1][1]
b11 = vector_player[0][0]
b12 = vector_player[0][1]
b21 = vector_player[1][0]
b22 = vector_player[1][1]

a = row_player
b = vector_player

Tx = 0.5
Ty = 0.5
K = 1

x = [0.9, 0.1]
y = [0.8, 0.2]
x = [0.1, 0.9]
y = [0.91, 0.09]
# x = [0.1, 0.9]
# y = [0.9, 0.1]

x_learned_policy = []
y_learned_policy = []

for _ in range(10000):
        r1x = a11*y[0] + a12*y[1]
        r2x = a21*y[0] + a22*y[1]
        r1y = b11*x[0] + b12*x[1]
        r2y = b21*x[0] + b22*x[1]


        Ay = [a11*y[0] + a12*y[1], a21*y[0] + a22*y[1]]
        Bx = [b11*x[0] + b12*x[1], b21*x[0] + b22*x[1]]


        u = [0,0]
        v = [0,0]

        for i in range(len(u)):
                J = 0
                Sk1 = 0
                Sk2 = 0
                Sk3 = 0
                for j in range(2):
                        for k in range(2):
                                if a[i][k] <= a[i][j]:
                                        Sk1 += y[k]
                                if a[i][k] < a[i][j]:
                                        Sk2 += y[k]
                                if a[i][k] == a[i][j]:
                                        Sk3 += y[k]
                        J += (a[i][j]*y[j]*(Sk1**K - Sk2**K))/Sk3
                u[i] = J

        for i in range(len(v)):
                J = 0
                Sk1 = 0
                Sk2 = 0
                Sk3 = 0
                for j in range(2):
                        for k in range(2):
                                if b[i][k] <= b[i][j]:
                                        Sk1 += x[k]
                                if b[i][k] < b[i][j]:
                                        Sk2 += x[k]
                                if b[i][k] == b[i][j]:
                                        Sk3 += x[k]
                        J += (b[i][j]*x[j]*(Sk1**K - Sk2**K))/Sk3
                v[i] = J

        delta_x = x[0]*(u[0] - (x[0]*u[0] +  x[1]*u[1]) +   Tx*(x[1]*np.log(x[1]/x[0])))
        delta_y = y[0]*(v[0] - (y[0]*v[0] +  y[1]*v[1]) +   Ty*(y[1]*np.log(y[1]/y[0])))

        # delta_x = x[0]*(Ay[0] - (x[0]*Ay[0] +  x[1]*Ay[1]) +   Tx*(x[1]*np.log(x[1]/x[0])))
        # delta_y = y[0]*(Bx[0] - (y[0]*Bx[0] +  y[1]*Bx[1]) +   Ty*(y[1]*np.log(y[1]/y[0])))
        
        alpha = 0.01
        x = [x[0]+alpha*delta_x, x[1]-alpha*delta_x]
        y = [y[0]+alpha*delta_y, y[1]-alpha*delta_y]

        x_learned_policy.append(x[0])
        y_learned_policy.append(y[0])


x = [0.8, 0.2]
y = [0.9, 0.1]
x = [0.1, 0.9]
y = [0.89, 0.11]
# x = [0.9, 0.1]
# y = [0.1, 0.9]

x2_learned_policy = []
y2_learned_policy = []

for _ in range(10000):
        r1x = a11*y[0] + a12*y[1]
        r2x = a21*y[0] + a22*y[1]
        r1y = b11*x[0] + b12*x[1]
        r2y = b21*x[0] + b22*x[1]

        u = [0,0]
        v = [0,0]

        for i in range(len(u)):
                J = 0
                Sk1 = 0
                Sk2 = 0
                Sk3 = 0
                for j in range(2):
                        for k in range(2):
                                if a[i][k] <= a[i][j]:
                                        Sk1 += y[k]
                                if a[i][k] < a[i][j]:
                                        Sk2 += y[k]
                                if a[i][k] == a[i][j]:
                                        Sk3 += y[k]
                        J += (a[i][j]*y[j]*(Sk1**K - Sk2**K))/Sk3
                u[i] = J

        for i in range(len(v)):
                J = 0
                Sk1 = 0
                Sk2 = 0
                Sk3 = 0
                for j in range(2):
                        for k in range(2):
                                if b[i][k] <= b[i][j]:
                                        Sk1 += x[k]
                                if b[i][k] < b[i][j]:
                                        Sk2 += x[k]
                                if b[i][k] == b[i][j]:
                                        Sk3 += x[k]
                        J += (b[i][j]*x[j]*(Sk1**K - Sk2**K))/Sk3
                v[i] = J

        delta_x = x[0]*(u[0] - (x[0]*u[0] +  x[1]*u[1]) +   Tx*(x[1]*np.log(x[1]/x[0])))
        delta_y = y[0]*(v[0] - (y[0]*v[0] +  y[1]*v[1]) +   Ty*(y[1]*np.log(y[1]/y[0])))

        alpha = 0.01
        x = [x[0]+alpha*delta_x, x[1]-alpha*delta_x]
        y = [y[0]+alpha*delta_y, y[1]-alpha*delta_y]

        x2_learned_policy.append(x[0])
        y2_learned_policy.append(y[0])


x = [1-0.111, 1-0.899]
y = [1-0.1, 1-0.9]
x = [0.05, 0.95]
y = [0.7, 0.3]
# x = [0.9, 0.1]
# y = [0.9, 0.1]

x3_learned_policy = []
y3_learned_policy = []

for _ in range(10000):
        r1x = a11*y[0] + a12*y[1]
        r2x = a21*y[0] + a22*y[1]
        r1y = b11*x[0] + b12*x[1]
        r2y = b21*x[0] + b22*x[1]

        u = [0,0]
        v = [0,0]

        for i in range(len(u)):
                J = 0
                Sk1 = 0
                Sk2 = 0
                Sk3 = 0
                for j in range(2):
                        for k in range(2):
                                if a[i][k] <= a[i][j]:
                                        Sk1 += y[k]
                                if a[i][k] < a[i][j]:
                                        Sk2 += y[k]
                                if a[i][k] == a[i][j]:
                                        Sk3 += y[k]
                        J += (a[i][j]*y[j]*(Sk1**K - Sk2**K))/Sk3
                u[i] = J

        for i in range(len(v)):
                J = 0
                Sk1 = 0
                Sk2 = 0
                Sk3 = 0
                for j in range(2):
                        for k in range(2):
                                if b[i][k] <= b[i][j]:
                                        Sk1 += x[k]
                                if b[i][k] < b[i][j]:
                                        Sk2 += x[k]
                                if b[i][k] == b[i][j]:
                                        Sk3 += x[k]
                        J += (b[i][j]*x[j]*(Sk1**K - Sk2**K))/Sk3
                v[i] = J

        delta_x = x[0]*(u[0] - (x[0]*u[0] +  x[1]*u[1]) +   Tx*(x[1]*np.log(x[1]/x[0])))
        delta_y = y[0]*(v[0] - (y[0]*v[0] +  y[1]*v[1]) +   Ty*(y[1]*np.log(y[1]/y[0])))

        alpha = 0.01
        x = [x[0]+alpha*delta_x, x[1]-alpha*delta_x]
        y = [y[0]+alpha*delta_y, y[1]-alpha*delta_y]

        x3_learned_policy.append(x[0])
        y3_learned_policy.append(y[0])


x = [1-0.1, 1-0.9]
y = [1-0.111, 1-0.899]
x = [0.3, 0.7]
y = [0.95, 0.05]
# x = [0.9, 0.1]
# y = [0.9, 0.1]

x4_learned_policy = []
y4_learned_policy = []

for _ in range(10000):
        r1x = a11*y[0] + a12*y[1]
        r2x = a21*y[0] + a22*y[1]
        r1y = b11*x[0] + b12*x[1]
        r2y = b21*x[0] + b22*x[1]

        u = [0,0]
        v = [0,0]

        for i in range(len(u)):
                J = 0
                Sk1 = 0
                Sk2 = 0
                Sk3 = 0
                for j in range(2):
                        for k in range(2):
                                if a[i][k] <= a[i][j]:
                                        Sk1 += y[k]
                                if a[i][k] < a[i][j]:
                                        Sk2 += y[k]
                                if a[i][k] == a[i][j]:
                                        Sk3 += y[k]
                        J += (a[i][j]*y[j]*(Sk1**K - Sk2**K))/Sk3
                u[i] = J

        for i in range(len(v)):
                J = 0
                Sk1 = 0
                Sk2 = 0
                Sk3 = 0
                for j in range(2):
                        for k in range(2):
                                if b[i][k] <= b[i][j]:
                                        Sk1 += x[k]
                                if b[i][k] < b[i][j]:
                                        Sk2 += x[k]
                                if b[i][k] == b[i][j]:
                                        Sk3 += x[k]
                        J += (b[i][j]*x[j]*(Sk1**K - Sk2**K))/Sk3
                v[i] = J

        delta_x = x[0]*(u[0] - (x[0]*u[0] +  x[1]*u[1]) +   Tx*(x[1]*np.log(x[1]/x[0])))
        delta_y = y[0]*(v[0] - (y[0]*v[0] +  y[1]*v[1]) +   Ty*(y[1]*np.log(y[1]/y[0])))

        alpha = 0.01
        x = [x[0]+alpha*delta_x, x[1]-alpha*delta_x]
        y = [y[0]+alpha*delta_y, y[1]-alpha*delta_y]

        x4_learned_policy.append(x[0])
        y4_learned_policy.append(y[0])



# x = [1-0.111, 1-0.899]
# y = [1-0.1, 1-0.9]
x = [0.9, 0.1]
y = [0.05, 0.95]
# x = [0.9, 0.1]
# y = [0.9, 0.1]

x5_learned_policy = []
y5_learned_policy = []

for _ in range(10000):
        r1x = a11*y[0] + a12*y[1]
        r2x = a21*y[0] + a22*y[1]
        r1y = b11*x[0] + b12*x[1]
        r2y = b21*x[0] + b22*x[1]

        u = [0,0]
        v = [0,0]

        for i in range(len(u)):
                J = 0
                Sk1 = 0
                Sk2 = 0
                Sk3 = 0
                for j in range(2):
                        for k in range(2):
                                if a[i][k] <= a[i][j]:
                                        Sk1 += y[k]
                                if a[i][k] < a[i][j]:
                                        Sk2 += y[k]
                                if a[i][k] == a[i][j]:
                                        Sk3 += y[k]
                        J += (a[i][j]*y[j]*(Sk1**K - Sk2**K))/Sk3
                u[i] = J

        for i in range(len(v)):
                J = 0
                Sk1 = 0
                Sk2 = 0
                Sk3 = 0
                for j in range(2):
                        for k in range(2):
                                if b[i][k] <= b[i][j]:
                                        Sk1 += x[k]
                                if b[i][k] < b[i][j]:
                                        Sk2 += x[k]
                                if b[i][k] == b[i][j]:
                                        Sk3 += x[k]
                        J += (b[i][j]*x[j]*(Sk1**K - Sk2**K))/Sk3
                v[i] = J

        delta_x = x[0]*(u[0] - (x[0]*u[0] +  x[1]*u[1]) +   Tx*(x[1]*np.log(x[1]/x[0])))
        delta_y = y[0]*(v[0] - (y[0]*v[0] +  y[1]*v[1]) +   Ty*(y[1]*np.log(y[1]/y[0])))

        alpha = 0.01
        x = [x[0]+alpha*delta_x, x[1]-alpha*delta_x]
        y = [y[0]+alpha*delta_y, y[1]-alpha*delta_y]

        x5_learned_policy.append(x[0])
        y5_learned_policy.append(y[0])


# x = [1-0.1, 1-0.9]
# y = [1-0.111, 1-0.899]
x = [0.95, 0.05]
y = [0.1, 0.9]
# x = [0.9, 0.1]
# y = [0.9, 0.1]

x6_learned_policy = []
y6_learned_policy = []

for _ in range(10000):
        r1x = a11*y[0] + a12*y[1]
        r2x = a21*y[0] + a22*y[1]
        r1y = b11*x[0] + b12*x[1]
        r2y = b21*x[0] + b22*x[1]

        u = [0,0]
        v = [0,0]

        for i in range(len(u)):
                J = 0
                Sk1 = 0
                Sk2 = 0
                Sk3 = 0
                for j in range(2):
                        for k in range(2):
                                if a[i][k] <= a[i][j]:
                                        Sk1 += y[k]
                                if a[i][k] < a[i][j]:
                                        Sk2 += y[k]
                                if a[i][k] == a[i][j]:
                                        Sk3 += y[k]
                        J += (a[i][j]*y[j]*(Sk1**K - Sk2**K))/Sk3
                u[i] = J

        for i in range(len(v)):
                J = 0
                Sk1 = 0
                Sk2 = 0
                Sk3 = 0
                for j in range(2):
                        for k in range(2):
                                if b[i][k] <= b[i][j]:
                                        Sk1 += x[k]
                                if b[i][k] < b[i][j]:
                                        Sk2 += x[k]
                                if b[i][k] == b[i][j]:
                                        Sk3 += x[k]
                        J += (b[i][j]*x[j]*(Sk1**K - Sk2**K))/Sk3
                v[i] = J

        delta_x = x[0]*(u[0] - (x[0]*u[0] +  x[1]*u[1]) +   Tx*(x[1]*np.log(x[1]/x[0])))
        delta_y = y[0]*(v[0] - (y[0]*v[0] +  y[1]*v[1]) +   Ty*(y[1]*np.log(y[1]/y[0])))

        alpha = 0.01
        x = [x[0]+alpha*delta_x, x[1]-alpha*delta_x]
        y = [y[0]+alpha*delta_y, y[1]-alpha*delta_y]

        x6_learned_policy.append(x[0])
        y6_learned_policy.append(y[0])



game = pyspiel.create_matrix_game(row_player, vector_player)
payoff_tensor = game_payoffs_array(game)
dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection="2x2") # schaal , keuze plot 
res = ax.quiver(dyn)

plt.title(plot_name+ ", k=" + str(K) + ", Tx="+str(Tx)+", Ty="+str(Ty))
plt.xlabel('Player 1, probability of action 1')
plt.ylabel('Player 2, probability of action 1')

plt.plot(x_learned_policy, y_learned_policy, color="red")
plt.plot(x2_learned_policy, y2_learned_policy, color="red")
plt.plot(x3_learned_policy, y3_learned_policy, color="red")
plt.plot(x4_learned_policy, y4_learned_policy, color="red")
plt.plot(x5_learned_policy, y5_learned_policy, color="red")
plt.plot(x6_learned_policy, y6_learned_policy, color="red")




plt.show()
