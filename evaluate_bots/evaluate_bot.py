"""Play bots against each other."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
import fcpa_random_agent as ra
import FCPA_agent_WORKING as fa
import numpy as np



import pyspiel
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 12763391, "The seed to use for the RNG.")

# Supported types of players: "random", "human", "check_call", "fold"
flags.DEFINE_string("player0", "human", "check_call")
flags.DEFINE_string("player1", "check_call", "check_call")


def LoadAgent(agent_type, game, player_id, rng):
  if agent_type == "random":
    return uniform_random.UniformRandomBot(player_id, rng)
  elif agent_type == "human":
    return human.HumanBot()
  elif agent_type == "check_call":
    policy = pyspiel.PreferredActionPolicy([1, 0])
    return pyspiel.make_policy_bot(game, player_id, FLAGS.seed, policy)
  elif agent_type == "fold":
    policy = pyspiel.PreferredActionPolicy([0, 1])
    return pyspiel.make_policy_bot(game, player_id, FLAGS.seed, policy)
  else:
    raise RuntimeError("Unrecognized agent type: {}".format(agent_type))


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
  rng = np.random.RandomState(FLAGS.seed)
  random_bot = ra.get_agent_for_tournament(1)
  fcpa_bot = fa.get_agent_for_tournament(0)
 
  fcpa_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=100 50,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,bettingAbstraction=fcpa)")
  game = pyspiel.load_game(fcpa_game_string)
  bots = [fcpa_bot, LoadAgent("check_call",game,1,rng)]
  first_player_total=0
  second_player_total=0
  scoreplayer0= 0
  scoreplayer1= 0
  
  amount_of_games=200
  for i in range(amount_of_games):
      returns = evaluate_bots(game.new_initial_state(), bots, np.random)
      first_player_total+=returns[0]
      second_player_total+=returns[1]
      if returns[0]>=returns[1]:
            scoreplayer0+=1
      else:
            scoreplayer1+=1
                  
      print("Iteration: ",i," Returns: ",returns)
  print("First player RETURNS 200 games: ",first_player_total)
  print("Second player RETURNS 200 games : ",second_player_total)    
  print("gamescore of bot 0 : ",scoreplayer0,"/",amount_of_games)
  print("gamescore of bot 1 : ",scoreplayer1,"/",amount_of_games)
  
if __name__ == "__main__":
  main()
