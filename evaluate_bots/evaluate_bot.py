"""Play bots against each other."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import fcpa_random_agent as ra
import FCPA_agent_WORKING as fa
import numpy as np



import pyspiel


def evaluate_bots(state, bots, rng):
  """Plays bots against each other, returns terminal utility for each bot."""
  for bot in bots:
    bot.restart_at(state)
  while not state.is_terminal():
    if state.is_chance_node():
      outcomes, probs = zip(*state.chance_outcomes())
      action = rng.choice(outcomes, p=probs)
      for bot in bots:
        bot.inform_action(state, pyspiel.PlayerId.CHANCE, action)
      state.apply_action(action)
    elif state.is_simultaneous_node():
      joint_actions = [
          bot.step(state)
          if state.legal_actions(player_id) else pyspiel.INVALID_ACTION
          for player_id, bot in enumerate(bots)
      ]
      state.apply_actions(joint_actions)
    else:
      current_player = state.current_player()
      action = bots[current_player].step(state)
      for i, bot in enumerate(bots):
        if i != current_player:
          bot.inform_action(state, current_player, action)
      state.apply_action(action)
  return state.returns()

def main():
  random_bot = ra.get_agent_for_tournament(0)
  fcpa_bot = fa.get_agent_for_tournament(1)
  bots = [random_bot, fcpa_bot]
  fcpa_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=100 50,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,bettingAbstraction=fcpa)")
  game = pyspiel.load_game(fcpa_game_string)
  first_player_total=0
  second_player_total=0
  for i in range(200):
      returns = evaluate_bots(game.new_initial_state(), bots, np.random)
      first_player_total+=returns[0]
      second_player_total+=returns[1]
      print("Iteration: ",i," Returns: ",returns)
  print("First player RETURNS 200 games: ",first_player_total/200)
  print("Second player RETURNS 200 games : ",second_player_total/200)    

if __name__ == "__main__":
  main()
