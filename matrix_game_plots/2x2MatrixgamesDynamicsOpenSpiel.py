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


plot_name = 'Dispersion game'
row_player = [[-1,1],[1,-1]]
vector_player = [[-1,1],[1,-1]]

# plot_name = 'matching pennies'
# row_player = [[1,-1],[-1,1]]
# vector_player = [[-1,1],[1,-1]]

# plot_name = 'Battle of the sexes'
# row_player = [[3,0],[0,2]]
# vector_player = [[2,0],[0,3]]

# plot_name = 'Subsidy game'
# row_player = [[10,0],[11,12]]
# vector_player = [[10,11],[0,12]]

r11 = row_player[0][0]
r12 = row_player[0][1]
r21 = row_player[1][0]
r22 = row_player[1][1]
c11 = vector_player[0][0]
c12 = vector_player[0][1]
c21 = vector_player[1][0]
c22 = vector_player[1][1]

game = pyspiel.create_matrix_game(row_player, vector_player)

payoff_tensor = game_payoffs_array(game)
dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.boltzmannq)

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection="2x2") # schaal , keuze plot 
res = ax.quiver(dyn)
#ax.streamplot(dyn)

plt.title("Battle of the Sexes")
plt.xlabel('Woman, probability of action 1')
plt.ylabel('Man, probability of action 1')
plt.show()
