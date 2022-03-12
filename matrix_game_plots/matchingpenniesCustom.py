# import random
import pyspiel
import numpy as np
from open_spiel.python.egt import dynamics
from open_spiel.python.egt.utils import game_payoffs_array
import matplotlib.pyplot as plt

#game = pyspiel.load_game(FLAGS.game)
game =  pyspiel.create_matrix_game([[-1, 1], [1, -1]], [[1, -1], [-1, 1]]) #matchin pennies 
plot_name="matching pennies"
#print(FLAGS.game, game)
print("loaded game")

# convert game to matrix form if it isn't already a matrix game
if not isinstance(game, pyspiel.MatrixGame):
 game = pyspiel.extensive_to_matrix_game(game)
 num_rows, num_cols = game.num_rows(), game.num_cols()
 print("converted to matrix form with shape (%d, %d)" % (num_rows, num_cols))

# use iterated dominance to reduce the space unless the solver is LP (fast)

# game is now finalized
num_rows, num_cols = game.num_rows(), game.num_cols()
row_actions = [game.row_action_name(row) for row in range(num_rows)]
col_actions = [game.col_action_name(col) for col in range(num_cols)]
row_payoffs, col_payoffs = game_payoffs_array(game)
payoff_tensor= game_payoffs_array(game) 
dyn = dynamics.MultiPopulationDynamics(payoff_tensor , dynamics.replicator) #kan ook boltzman Q learning doen

X, Y = np.meshgrid(np.linspace(0.0, 1.0, 30), np.linspace(0.0, 1.0, 30))
print(X)
print(Y)
u, v = np.zeros_like(X), np.zeros_like(X)
NI, NJ = X.shape
payoff_matrix = game_payoffs_array(game)
for i in range(NI):
    for j in range(NJ):
     

        xChance = np.array([X[i, j], 1-X[i, j]])
        yChance = np.array([Y[i, j], 1-Y[i, j]])
        totalChance =np.concatenate((xChance,yChance))
        print(totalChance)
        changeTotalChance= dyn(totalChance)
        
        u[i,j] = changeTotalChance[0]
        v[i,j] = changeTotalChance[2]
        

  


plt.streamplot(X, Y, u, v, density=0.7)
plt.quiver(X, Y, u, v)
plt.axis('square')
plt.title(plot_name)
plt.xlabel('Player 1, probability of action 1')
plt.ylabel('Player 2, probability of action 1')
plt.axis([0, 1, 0, 1])
plt.show()
