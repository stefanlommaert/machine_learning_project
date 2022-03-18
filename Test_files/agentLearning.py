from types import new_class
from matplotlib import pyplot as plt

import numpy as np
#TODO: DIT WEG DENK IK (of gewoon verplaatsen)

row_player = [[1,-1],[-1,1]]
vector_player = [[-1,1],[1,-1]]
payoff_array = [row_player, vector_player]
beginning_pos = [0.4, 0.4]

def q_learning(chance_distribution_array, payoff_array):
    alpha, beta = chance_distribution_array[0], chance_distribution_array[1]
    possible_actions_player_1 = [0]
    if (alpha+0.05 <=1):
        possible_actions_player_1.append(0.05)
    if (alpha-0.05 >=0):
        possible_actions_player_1.append(-0.05)

    possible_actions_player_2 = [0]
    if (beta+0.05 <=1):
        possible_actions_player_2.append(0.05)
    if (beta-0.05 >=0):
        possible_actions_player_2.append(-0.05)

    actions_player_1 = dict()
    for action1 in possible_actions_player_1:
        utility_player_1 = calculate_utility([alpha+action1, beta], payoff_array[0])
        actions_player_1[(utility_player_1)] = action1
    best_action1 = (max(actions_player_1.keys()))
    utility_increase_player_1 = best_action1-calculate_utility([alpha, beta], payoff_array[0])
    action_player_1 = round(actions_player_1[best_action1],3)

    actions_player_2 = dict()
    for action2 in possible_actions_player_2:
        utility_player_2 = calculate_utility([alpha, beta+action2], payoff_array[1])
        actions_player_2[(utility_player_2)] = action2
    best_action2 = (max(actions_player_2.keys()))
    utility_increase_player_2 = best_action2-calculate_utility([alpha, beta], payoff_array[1])
    action_player_2 = round(actions_player_2[best_action2],3)
    return [alpha+action_player_1*(utility_increase_player_1/(utility_increase_player_1+utility_increase_player_2)), beta + action_player_2*(utility_increase_player_2/(utility_increase_player_1+utility_increase_player_2))]
    
def calculate_utility(chance_distribution_array, payoff_array):
    alpha, beta = chance_distribution_array[0], chance_distribution_array[1]
    print(alpha)
    print(beta)
    
    r11 = payoff_array[0][0]
    r12 = payoff_array[0][1]
    r21 = payoff_array[1][0]
    r22 = payoff_array[1][1]
    utility_player = r11*(alpha*beta) + r22*((1- alpha)*(1- beta)) + r12*(alpha*(1-beta))+ r21*((1-alpha)*beta)
    return round(utility_player, 5)

new_chance_distribution = beginning_pos
x = np.array([])
y = np.array([])


for i in range(100):
    try:
        new_chance_distribution = q_learning(new_chance_distribution, payoff_array)
        chance_1, chance_2 = new_chance_distribution[0], new_chance_distribution[1]

        x = np.append(x, new_chance_distribution[0])
        y = np.append(y, new_chance_distribution[1])
    except:
        print('error')

    
plt.plot(x, y, 'o')
plt.axis('square')
plt.xlabel('Player 1, probability of action 1')
plt.ylabel('Player 2, probability of action 1')
plt.axis([0, 1, 0, 1])
plt.show()
