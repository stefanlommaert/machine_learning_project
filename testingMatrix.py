# import random
import pyspiel
import numpy as np
from open_spiel.python.egt import dynamics
from open_spiel.python.egt.utils import game_payoffs_array
import matplotlib.pyplot as plt





row_player = [[1,-1],[-1,1]]
vector_player = [[1,-1],[-1,1]]
game = pyspiel.create_matrix_game(row_player, vector_player)
state = game.new_initial_state()
payoff_matrix = game_payoffs_array(game)

# BELANGRIJK! SinglePopulation kan enkel gebruikt worden bij symmetrische matrices, 
# indien niet symmetrisch moet MultiPopulation gebruikt worden
dyn = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)
chance_distribution = np.array([0.49, 0.51])
learning_rate = 0.1
for i in range(50):
    chance_distribution += learning_rate*dyn(chance_distribution)
    # print(chance_distribution)


X, Y = np.meshgrid(np.linspace(0.0, 1.0, 50), np.linspace(0.0, 1.0, 50))
u, v = np.zeros_like(X), np.zeros_like(X)
NI, NJ = X.shape
for i in range(NI):
    for j in range(NJ):
        x, y = X[i, j], Y[i, j]
        chance_distribution = np.array([x, 1-x])
        print((x,y),dyn(chance_distribution))
        # print((x,y))
        u[i,j] = dyn(chance_distribution)[0]
        v[i,j] = dyn(chance_distribution)[1]
plt.streamplot(X, Y, u, v)
plt.axis('square')
plt.axis([0, 1, 0, 1])
plt.show()




# game = pyspiel.load_matrix_game("matrix_rps")
# payoff_matrix = game_payoffs_array(game)
# dyn = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)
# x = np.array([0.2, 0.2, 0.6])
# print(dyn(x))
