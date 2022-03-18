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

game =  pyspiel.create_matrix_game([[-1, 1], [1, -1]], [[1, -1], [-1, 1]]) #matchin pennies 
payoff_tensor = game_payoffs_array(game)
dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection="2x2") # schaal , keuze plot 
#res = ax.quiver(dyn)
ax.streamplot(dyn)

plt.title("Matching pennies")
plt.xlabel('Player 1, probability of action heads')
plt.ylabel('Player 2, probability of action heads')
plt.show()
