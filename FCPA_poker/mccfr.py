
# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example use of the CFR algorithm on Kuhn Poker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app

from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr
from open_spiel.python.algorithms import rcfr
import pyspiel
# def sequence_weights_to_tabular_profile(root, policy):
#     """Returns the `dict` of `list`s of action-prob pairs-form of `policy_fn`."""
#     tabular_policy = {}
#     players = range(root.num_players())
#     for state in rcfr.all_states(root):
#         for player in players:
#             legal_actions = state.legal_actions(player)
#             if len(legal_actions) < 1:
#                 continue
#             info_state = state.information_state_string(player)
#             if info_state in tabular_policy:
#                 continue
            
#             my_policy = policy.action_probabilities(state)
#             tabular_policy[info_state] = list(zip(legal_actions, my_policy))
#     return tabular_policy

def main(_):
    fcpa_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=1,blind=150 100,"
        "firstPlayer=2 1 1 1,numSuits=1,numRanks=4,numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,bettingAbstraction=fcpa)")
    game = pyspiel.load_game(fcpa_game_string)
    es_solver = external_sampling_mccfr.ExternalSamplingSolver(
    game, external_sampling_mccfr.AverageType.SIMPLE)
    for _ in range(10):
        es_solver.iteration()
    conv = exploitability.nash_conv(game, es_solver.average_policy())
    print("KUHN, conv = {}".format(conv))
    
    print("Iteration {} exploitability {}".format(10, conv))

    average_policy = es_solver.average_policy()
     
    average_policy_values = expected_game_score.policy_value(
    game.new_initial_state(), [average_policy] * 2)
   
    # for key in sequence_weights_to_tabular_profile(game.new_initial_state(),es_solver.average_policy()):
    #     print("key :",key, "average_policy_values :",sequence_weights_to_tabular_profile(game.new_initial_state(),es_solver.average_policy())[key])   
    print("Computed player 0 value: {}".format(average_policy_values[0]))
    

if __name__ == "__main__":
  app.run(main)
