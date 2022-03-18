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

game =  pyspiel.load_game("matrix_rps")
payoff_matrix= payoff_matrix = game_payoffs_array(game)
dyn = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection="3x3") #eerste nummer bepaald de schaal tegenover elkaar tweede welk soort plot je wilt (van openspiel)
res = ax.quiver(dyn)
ax.streamplot(dyn)
plt.show()

