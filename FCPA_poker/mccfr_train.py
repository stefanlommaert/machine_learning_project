
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
import time

from absl import app
import numpy as np

from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr
from open_spiel.python.algorithms import rcfr
import pyspiel

def find_between( s, first, last ):
      try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start-1:end+1]
      except ValueError:
        return ""
    
def main(_):
    fcpa_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=2,blind=150 100,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,bettingAbstraction=fcpa)")
    game = pyspiel.load_game(fcpa_game_string)
    print("GOT HERE")
    es_solver = external_sampling_mccfr.ExternalSamplingSolver(
    game, external_sampling_mccfr.AverageType.SIMPLE)
    
    #load back in previous trained model (coninue training on model)
    es_solver._infostates= np.load("FCPA_poker/FCPA_tournament/fcpa_agent/infostates/full_fpca_agent_infostats.npy",allow_pickle=True)[()] 
    start = time.time()
    print("STARTING ITERATION: ")
    
    for i in range(100):
        if i % 1000 == 0:
            print("ITERATION: ",i)
        es_solver.iteration()
        
    end = time.time()
    print("ENDING ITERATION: ")
    
    # different_action_abstractions= []
    # different_card_abstractions=[]
    # print("Trying to save infostates")
    # #np.save('FCPA_poker/FCPA_tournament/fcpa_agent/infostates/full_fpca_agent_infostats.npy', es_solver.average_policy()._infostates) 
    # print("saved infostates")    
    # for key in es_solver.average_policy()._infostates:
    #     cards= ""
    #     #op één of andere reden is er geen copy voor strings en moet je dit doen #nicePython
    #     actions= (key + '.')[:-1]
        
    #     for i in range(2):
    #         rounds = find_between(actions,"[","]") 
    #         actions=actions.replace(rounds,"")
    #         cards=cards+rounds
    #     if not ( cards in different_card_abstractions):
    #         different_card_abstractions.append(cards)
    #     if not (actions in different_action_abstractions):
    #         different_action_abstractions.append(actions)    
   
    # print("ALL CARD COMBINATIONS: =",different_card_abstractions)
    # print("AMOUNT of CARD COMBINATIONS: =",len(different_card_abstractions))
    
    # print("ALL ACTION COMBINATIONS: =",different_action_abstractions)    
    # print("AMOUNT OF ACTION COMBINATIONS: =",len(different_action_abstractions))
    # print("AMOUNT OF STATES IN POLICY: ",len(es_solver.average_policy()._infostates))
    print("TOOK: ",end-start)
   

    
    

if __name__ == "__main__":
  app.run(main)
