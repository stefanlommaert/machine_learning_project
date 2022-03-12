# import random
#ik weet niet goed wat we meer moeten doen dan gewoon de nash waarde vinden en vergelijken met de eigenlijke nash (door nashpy)
import pyspiel
import numpy as np
from open_spiel.python.egt import dynamics
from open_spiel.python.egt.utils import game_payoffs_array
import matplotlib.pyplot as plt
import nashpy as nash
import numpy as np

game =  pyspiel.create_matrix_game([[-1, 1], [1, -1]], [[1, -1], [-1, 1]])


state = game.new_initial_state()

#game= pyspiel.load_matrix_game("matrix_rps")

payoff_matrix= game_payoffs_array(game)
print(str(payoff_matrix))
dyn=dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)

x=np.array([0.2,0.2,0.6])

alpha= 0.01

for n in range(1,100):
    x+= alpha*dyn(x)
    
print("estemated nash ="+str(x))

    
#CALCULATE THE REAL NASH VALUES WITH NASHPY   
A = np.array([[1, -1], [-1, 1]])
#omgekeerde want zero sum helaas zijn beiden libs niet echt volgens zelfde standaard

B =  np.array([[-1, 1], [1, -1]])

rps = nash.Game(A,B)
eqs = rps.support_enumeration()

print("calculated NASH = "+ str(list(eqs)))
