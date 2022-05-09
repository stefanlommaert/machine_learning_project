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
import random

from absl import app
import numpy as np

from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr
from open_spiel.python.algorithms import mccfr
import pyspiel

import json

def main(_):
    game = pyspiel.load_game("leduc_poker")
    es_solver = external_sampling_mccfr.ExternalSamplingSolver(
    game, external_sampling_mccfr.AverageType.SIMPLE)
    for _ in range(10000):
        es_solver.iteration()
    conv = exploitability.nash_conv(game, es_solver.average_policy())
    #print("type: ", es_solver.average_policy()._infostates)
    #print("type: ", type(es_solver.average_policy))
    
    #Save the policy (or atleast the most important part )
    d2=np.load("leduc_poker/leduc_agent_infostats.npy",allow_pickle=True) #allow the magic pickle
    #load the policy back in 
    print(d2[()])
    print(type(d2[()])) #magic vegaga symbol
    
    print("KUHN, conv = {}".format(conv))
    
    print("Iteration {} exploitability {}".format(10, conv))

    average_policy = mccfr.AveragePolicy(game, list(range(game.num_players())),
                         d2[()])
    
    #construct the average policy with the saved file 
    average_policy_values = expected_game_score.policy_value(
    game.new_initial_state(), [average_policy] * 2)
    print("loading in average policy : ")
    # states =rcfr.all_states(game.new_initial_state())
    # for state in states:
    #     print("state :",state)
    #     print("average_policy_values :",es_solver.average_policy().action_probabilities(state))
    #     for key in es_solver.average_policy().action_probabilities(state):
    #         print("key :",key)
    #         value = es_solver.average_policy().action_probabilities(state)[key]
    #         print("value :",value)
    #         randfloat = random.random()
    #         if randfloat < value:
    #             print("RANDOM CHOSEN ACTION: ",key)
    #             break
     
    print("Computed player 0 value: {}".format(average_policy_values[0]))
    print("Expected player 0 value: {}".format(-1 / 18))
    

if __name__ == "__main__":
  app.run(main)
