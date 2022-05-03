from matplotlib import pyplot as plt
import numpy as np
import random
import pyspiel
from open_spiel.python.egt.utils import game_payoffs_array
from open_spiel.python.egt import visualization
from matplotlib.figure import Figure
from matplotlib.quiver import Quiver
import matplotlib
import ternary
epsilon =  0.1
options = [0,1,2] #0 is rock 1 is paper 2 is scissor 

row_player = [[0, -0.25, 0.5], [0.25, 0, -0.05], [-0.5, 0.05, 0]]
vector_player = [[0, 0.25, -0.5], [-0.25, 0, 0.05], [0.5, -0.05, 0]]


game = pyspiel.create_matrix_game(row_player, vector_player)

payoff_tensor = game_payoffs_array(game)

rewardsP1  =[0,0,0]
countsP1   =[0,0,0] 
averagerewardsP1 = [0,0,0]

rewardsP2  = [0,0,0]
countsP2   = [0,0,0] 
averagerewardsP2= [0,0,0]



def P1explore():
    return random.choice(options)


def P1_select_action():
    if (random.random()<epsilon):
        return P1explore()
    else:
        return P1exploit()

def P2explore():
    return random.choice(options)


def P2_select_action():
    if (random.random()<epsilon):
        return P2explore()
    else:
        return P2exploit()
    
def P1exploit():
    if sum(countsP2)==0 : 
        rockchanceP2  = 1/3
        paperchanceP2  = 1/3
        scissorchanceP2  = 1/3
    
    else:
        rockchanceP2  = float(countsP2[0])/float(sum(countsP2))
        paperchanceP2  = float(countsP2[1])/float(sum(countsP2))
        scissorchanceP2  = float(countsP2[2])/float(sum(countsP2))
                                
    
    
  
    utility_action1 = row_player[0][0]*rockchanceP2 + row_player[0][1]*paperchanceP2 + row_player[0][2]*scissorchanceP2 
     
    utility_action2 = row_player[1][0]*rockchanceP2 + row_player[1][1]*paperchanceP2 + row_player[1][2]*scissorchanceP2 
    utility_action3 = row_player[2][0]*rockchanceP2 + row_player[2][1]*paperchanceP2 + row_player[2][2]*scissorchanceP2
    all_actions= [utility_action1, utility_action2, utility_action3] 
     
    return (all_actions.index(max(all_actions)))


def P2exploit():
    if sum(countsP1)==0 : 
        rockchanceP1  = 1/3
        paperchanceP1  = 1/3
        scissorchanceP1  = 1/3
    
    else:
        rockchanceP1  = float(countsP1[0])/float(sum(countsP1))
        paperchanceP1  = float(countsP1[1])/float(sum(countsP1))
        scissorchanceP1  = float(countsP1[2])/float(sum(countsP1))
                                
    
    # print(rockchanceP1,paperchanceP1,scissorchanceP1)
  
    utility_action1 = vector_player[0][0]*rockchanceP1 + vector_player[1][0]*paperchanceP1 + vector_player[2][0]*scissorchanceP1 
     
    utility_action2 = vector_player[0][1]*rockchanceP1 + vector_player[1][1]*paperchanceP1 + vector_player[2][1]*scissorchanceP1 
    utility_action3 = vector_player[0][2]*rockchanceP1 + vector_player[1][2]*paperchanceP1 + vector_player[2][2]*scissorchanceP1
    all_actions= [utility_action1, utility_action2, utility_action3] #TODO: hier klopt iets niet nu is het 
    # print("all_actions: ",all_actions)   
    return (all_actions.index(max(all_actions)))
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
            P1_averages.append(np.array(countsP1)/float(sum(countsP1)))
            
            P2_averages.append(np.array(countsP2)/float(sum(countsP2)))
            
            
    
    print("Player 1 rewards: ",rewardsP1)
    print("Player 1 counts:  ",countsP1) 
    print("Player 1 average rewards: ",averagerewardsP1)
    print("player 1 chance action1: ", np.array(countsP1)/float(sum(countsP1)))
    
    print("Player 2 rewards: ",rewardsP2)
    print("Player 2 counts:  ",countsP2) 
    print("Player 2 average rewards: ",averagerewardsP2)
    print("player 2 chance action1: ", np.array(countsP2)/float(sum(countsP2)))
       
        
episodes = 1000000
play_game(episodes)

fig, tax = ternary.figure(scale=100)
fig.set_size_inches(10, 9)

# Plot points.
points =  P2_averages
points = [(100*x,100*y,100*z) for (x, y, z) in points]
tax.plot_colored_trajectory(points, cmap="hsv", linewidth=2.0)

# Axis labels. (See below for corner labels.)
fontsize = 14
offset = 0.08
tax.left_axis_label("Scissor %", fontsize=fontsize, offset=offset)
tax.right_axis_label("Paper %", fontsize=fontsize, offset=offset)
tax.bottom_axis_label("Rock %", fontsize=fontsize, offset=-offset)
tax.set_title("epsilon Q_learning RPS player 2, "+"episodes = "+str(episodes), fontsize=20)

# Decoration.
tax.boundary(linewidth=1)
tax.gridlines(multiple=10, color="gray")
tax.ticks(axis='lbr', linewidth=1, multiple=20)
tax.get_axes().axis('off')

label= "start"
tax.annotate(label, # this is the texst
             (points[0][0],points[0][1],points[0][2]), # these are the coordinates to position the label
             textcoords="offset points", # how to position the texst
             xytext=(0,1),
             ha='center',
             fontsize=10) # horizontal alignment can be left, right or center
label  = "end"

tax.annotate(label, # this is the texst
             (points[-1][0],points[-1][1],points[-1][2]), # these are the coordinates to position the label
             textcoords="offset points", # how to position the texst
             ha='center',
              xytext=(0,1),
             fontsize=10) # horizontal alignment can be left, right or center

fig_P1, tax_P1 = ternary.figure(scale=100)
fig_P1.set_size_inches(10, 9)

# Plot points.
points =  P1_averages
points = [(100*x,100*y,100*z) for (x, y, z) in points]
tax_P1.plot_colored_trajectory(points, cmap="hsv", linewidth=2.0)

# Axis labels. (See below for corner labels.)
fontsize = 14
offset = 0.08
tax_P1.left_axis_label("Scissor %", fontsize=fontsize, offset=offset)
tax_P1.right_axis_label("Paper %", fontsize=fontsize, offset=offset)
tax_P1.bottom_axis_label("Rock %", fontsize=fontsize, offset=-offset)
tax_P1.set_title("epsilon Q_learning RPS player 1, "+"episodes = "+str(episodes), fontsize=20)

# Decoration.
tax_P1.boundary(linewidth=1)
tax_P1.gridlines(multiple=10, color="gray")
tax_P1.ticks(axis='lbr', linewidth=1, multiple=20)
tax_P1.get_axes().axis('off')

label= "start"
tax_P1.annotate(label, # this is the texst
             (points[0][0],points[0][1],points[0][2]), # these are the coordinates to position the label
             textcoords="offset points", # how to position the texst
             xytext=(0,1),
             ha='center',
             fontsize=10) # horizontal alignment can be left, right or center
label  = "end"

tax_P1.annotate(label, # this is the texst
             (points[-1][0],points[-1][1],points[-1][2]), # these are the coordinates to position the label
             textcoords="offset points", # how to position the texst
             ha='center',
              xytext=(0,1),
             fontsize=10) # horizontal alignment can be left, right or center
tax.show()
tax_P1.show()