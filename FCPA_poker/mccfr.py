
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


def main(_):
    fcpa_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=150 100,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,bettingAbstraction=fcpa)")
    game = pyspiel.load_game(fcpa_game_string)
    print("GOT HERE")
    es_solver = external_sampling_mccfr.ExternalSamplingSolver(
    game, external_sampling_mccfr.AverageType.SIMPLE)
    start = time.time()
    print("STARTING ITERATION: ")
    
    for i in range(10000):
        print('episode :',i )
        es_solver.iteration()
    end = time.time()
    print("ENDING ITERATION: ")
    
   
    
    print("TOOK: ",end-start)
   
    
    print("Trying to save infostates")
    np.save('FCPA_poker/FCPA_tournament/fcpa_agent/infostates/full_fpca_agent_infostats.npy', es_solver.average_policy()._infostates)   

    
    

if __name__ == "__main__":
  app.run(main)
