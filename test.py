import pyspiel
import random

fcpa_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=150 100,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1 1,"
        "stack=20000 20000,bettingAbstraction=fcpa)")
game = pyspiel.load_game(fcpa_game_string)
state = game.new_initial_state()
# for i in range(15):
#         print("\r\n", state)
#         print(random.choice(state.legal_actions()))
#         action = input("Action: ")
#         state.apply_action(int(action))  
print("\r\n", state.chance_outcomes())
# state.apply_action(5)
# print("\r\n", state)
# state.apply_action(8)
# print("\r\n", state)
# state.apply_action(2)
# print("\r\n", state)
# state.apply_action(3)
# print("\r\n", state)
# print(state.legal_actions())
# state.apply_action(1)
# print("\r\n", state)
# print(state.legal_actions())
# state.apply_action(1)
# print("\r\n", state)
# print(state.legal_actions())

# state.apply_action(11)
# print("\r\n", state)
# print(state.legal_actions())
# state.apply_action(25)
# print("\r\n", state)
# print(state.legal_actions())
# state.apply_action(51)
# print("\r\n", state)
# print(state.legal_actions())
