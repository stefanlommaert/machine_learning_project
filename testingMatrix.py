# import random
import pyspiel
import numpy as np
from open_spiel.python.egt import dynamics
from open_spiel.python.egt.utils import game_payoffs_array
import matplotlib.pyplot as plt


row_player = [[-1,1],[1,-1]]
vector_player = [[-1,1],[1,-1]]
r11 = row_player[0][0]
r12 = row_player[0][1]
r21 = row_player[1][0]
r22 = row_player[1][1]
c11, c12, c21, c22 = r11, r12, r21, r22

game = pyspiel.create_matrix_game(row_player, vector_player)
state = game.new_initial_state()

X, Y = np.meshgrid(np.linspace(0.0, 1.0, 50), np.linspace(0.0, 1.0, 50))
u, v = np.zeros_like(X), np.zeros_like(X)
NI, NJ = X.shape
payoff_matrix = game_payoffs_array(game)
for i in range(NI):
    for j in range(NJ):
        # hier probeer ik de klote API van openspiel (spoiler: het trekt op niks!)
        # dynX = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)
        # dynY = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)

        # xChance = np.array([X[i, j], 1-X[i, j]])
        # yChance = np.array([Y[i, j], 1-Y[i, j]])
        # u[i,j] = dynX(yChance)[0]
        # v[i,j] = dynY(xChance)[0]

        # deze methode gebruikt de functies van paper 3 (Nash Convergence of Gradient Dynamics in Iterated General-Sum Games)
        alpha, beta = X[i, j], Y[i, j]
        u[i,j] = beta*((r11+r22)-(r21+r12))-(r22-r12)
        v[i,j] = alpha*((c11+c22)-(c21+c12))-(c22-c12)
        #projection of vector on border when point lies on border
        if (alpha==0 or alpha==1):
            u[i,j]=0
        if (beta==0 or beta==1):
            v[i,j]=0
plt.streamplot(X, Y, u, v)
plt.axis('square')
plt.axis([0, 1, 0, 1])
plt.show()



# BELANGRIJK! SinglePopulation kan enkel gebruikt worden bij symmetrische matrices, 
# indien niet symmetrisch moet MultiPopulation gebruikt worden
# game = pyspiel.load_matrix_game("matrix_rps")
# payoff_matrix = game_payoffs_array(game)
# dyn = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)
# x = np.array([0.2, 0.2, 0.6])
# print(dyn(x))
