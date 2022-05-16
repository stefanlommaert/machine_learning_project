import re
from matplotlib import pyplot as plt
import numpy as np
import random
import pyspiel
from open_spiel.python.egt.utils import game_payoffs_array
from open_spiel.python.egt import visualization
from matplotlib.figure import Figure
from matplotlib.quiver import Quiver

epsilon =  0.1
options = [0,1] #0 is head 1 is tail 

# plot_name = 'Dispersion game'
# row_player = [[-1,1],[1,-1]]
# vector_player = [[-1,1],[1,-1]]

# plot_name = 'Battle of the sexes'
# row_player = [[3,0],[0,2]]
# vector_player = [[2,0],[0,3]]

# plot_name = 'Subsidy game'
# row_player = [[10,0],[11,12]]
# vector_player = [[10,11],[0,12]]

plot_name = 'Matching pennies'
row_player=[[-1, 1], [1, -1]]
vector_player = [[1, -1], [-1, 1]]


r11 = row_player[0][0]
r12 = row_player[0][1]
r21 = row_player[1][0]
r22 = row_player[1][1]
c11 = vector_player[0][0]
c12 = vector_player[0][1]
c21 = vector_player[1][0]
c22 = vector_player[1][1]

game = pyspiel.create_matrix_game(row_player, vector_player)
# game = pyspiel.load_matrix_game("matrix_pd")
payoff_tensor= game_payoffs_array(game) 

rewardsP1  =[0,0]
countsP1   =[0,0] 
averagerewardsP1 = [0,0] 

rewardsP2  = [0,0]
countsP2   = [0,0] 
averagerewardsP2= [0,0]


def P1explore():
    return random.choice(options)

def P1exploit():
    return averagerewardsP1.index(max(averagerewardsP1))

def P1_select_action():
    if (random.random()<epsilon):
        return P1explore()
    else:
        return P1exploit()

def P2explore():
    return random.choice(options)

def P2exploit():
    return averagerewardsP2.index(max(averagerewardsP2))

def P2_select_action():
    if (random.random()<epsilon):
        return P2explore()
    else:
        return P2exploit()
    

def update(p1action,p2action):
    payoffP1 = payoff_tensor[0,p1action,p2action]
    payoffP2 = payoff_tensor[1,p1action,p2action]
    countsP1[p1action] += 1
    rewardsP1[p1action]+= payoffP1
    averagerewardsP1[p1action] = float(rewardsP1[p1action])/float(countsP1[p1action])
    
    
    countsP2[p2action]+=1
    rewardsP2[p2action]+=payoffP2
    averagerewardsP2[p2action] = float(rewardsP2[p2action])/float(countsP2[p2action])
    
P1_averages=[]
P2_averages=[]
measurement_stepsize=10
def play_game(i):
    for x in range(i):
        p1action = P1_select_action()
        p2action = P2_select_action()
        update(p1action, p2action)
      
        if ((x%measurement_stepsize)==0 and (x>measurement_stepsize)):
            P1_averages.append(float(countsP1[0])/float(sum(countsP1)))
            
            P2_averages.append(float(countsP2[0])/float(sum(countsP2)))
            
            
    
    print("Player 1 rewards: ",rewardsP1)
    print("Player 1 counts:  ",countsP1) 
    print("Player 1 average rewards: ",averagerewardsP1)

    print("Player 2 rewards: ",rewardsP2)
    print("Player 2 counts:  ",countsP2) 
    print("Player 2 average rewards: ",averagerewardsP2)
       
        
episodes = 1000000
play_game(episodes)
#plt.axis('square')
plt.title("Epsilon Greedy: "+plot_name+ ", episodes = "+str(episodes))
plt.xlabel('Episode')
plt.ylabel('Minimal distance to any Nash (%)')
#plt.axis([0, 1, 0, 1])

actual_nash= [[np.array([0.5,0.5])]]
# create some x data and some integers for the y axis
x = np.array(P1_averages)
y = np.array(P2_averages)

totalvalues= []
nash_equlibriums = np.array([0.5])
for values in x:
     totalvalues.append( abs(np.min(nash_equlibriums-values)))


# plot the data
#plt.plot(x,y)

# number_label=measurement_stepsize om de hoeveelste plot level aan te duiden maar is nogal scuffed
print(totalvalues)
print(len(totalvalues))
plt.plot(range(len(totalvalues)),np.array(totalvalues))
# label  = "start"

# plt.annotate(label, # this is the texst
#              (x[0],y[0]), # these are the coordinates to position the label
#              textcoords="offset points", # how to position the texst
#              xytext=(0,1), # distance from texst to points (xs,ys)
#              ha='center',
#              fontsize=10) # horizontal alignment can be left, right or center
# label  = "end"

# plt.annotate(label, # this is the texst
#              (x[-1],y[-1]), # these are the coordinates to position the label
#              textcoords="offset points", # how to position the texst
#              xytext=(0,1), # distance from texst to points (xs,ys)
#              ha='center',
#              fontsize=10) # horizontal alignment can be left, right or center
plt.show()


 