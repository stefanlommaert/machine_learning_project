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

# plot_name = 'testing game'
# row_player = [[3,0],[5,1]]
# vector_player = [[3,5],[0,1]]


# plot_name = 'Dispersion game'
# row_player = [[-1,1],[1,-1]]
# vector_player = [[-1,1],[1,-1]]

plot_name = 'matching pennies'
row_player = [[1,-1],[-1,1]]
vector_player = [[-1,1],[1,-1]]

# plot_name = 'Battle of the sexes'
# row_player = [[3,0],[0,2]]
# vector_player = [[2,0],[0,3]]

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

x = [0.9, 0.1]
y = [0.9, 0.1]

x_learned_policy = []
y_learned_policy = []

for _ in range(1000):
        r1x = a11*y[0] + a12*y[1]
        r2x = a21*y[0] + a22*y[1]
        r1y = b11*x[0] + b12*x[1]
        r2y = b21*x[0] + b22*x[1]

        Tx = 1
        Ty = 1
        
        Ay = [a11*y[0] + a12*y[1], a21*y[0] + a22*y[1]]
        Bx = [b11*x[0] + b12*x[1], b21*x[0] + b22*x[1]]

        delta_x = x[0]*(Ay[0] - (x[0]*Ay[0] +  x[1]*Ay[1]) +   Tx*(x[1]*np.log(x[1]/x[0])))
        delta_y = y[0]*(Bx[0] - (y[0]*Bx[0] +  y[1]*Bx[1]) +   Ty*(y[1]*np.log(y[1]/y[0])))
        
        alpha = 0.01
        x = [x[0]+alpha*delta_x, x[1]-alpha*delta_x]
        y = [y[0]+alpha*delta_y, y[1]-alpha*delta_y]


        # print(x[0],y[0])
        # print(delta_x, delta_x_2)
        x_learned_policy.append(x[0])
        y_learned_policy.append(y[0])


# plt.title("Line graph")
# plt.plot(x_learned_policy, y_learned_policy, color="red")

# plt.show()









game = pyspiel.create_matrix_game(row_player, vector_player)

payoff_tensor = game_payoffs_array(game)
dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection="2x2") # schaal , keuze plot 
res = ax.quiver(dyn)
# ax.streamplot(dyn)

plt.title("Gradient plot: "+plot_name)
plt.xlabel('Player 1, probability of action 1')
plt.ylabel('Player 2, probability of action 1')
plt.plot(x_learned_policy, y_learned_policy, color="red")
plt.show()
