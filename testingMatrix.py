# import random
import pyspiel
import numpy as np
from open_spiel.python.egt import dynamics
from open_spiel.python.egt.utils import game_payoffs_array




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
    print(chance_distribution)


# game = pyspiel.load_matrix_game("matrix_rps")
# payoff_matrix = game_payoffs_array(game)
# dyn = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)
# x = np.array([0.2, 0.2, 0.6])
# print(dyn(x))
