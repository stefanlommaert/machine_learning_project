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

"""Python Deep CFR example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from open_spiel.python import policy
from open_spiel.python.algorithms import deep_cfr_tf2
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 1, "Number of iterations")
flags.DEFINE_integer("num_traversals", 3, "Number of traversals/games")
flags.DEFINE_string("game_name", "leduc_poker", "Name of the game")


def main(unused_argv):
  logging.info("Loading %s", FLAGS.game_name)
  fcpa_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=150 100,"
        "firstPlayer=2 1 1 1,numSuits=1,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,bettingAbstraction=fcpa)")
  game = pyspiel.load_game(fcpa_game_string)
  deep_cfr_solver = deep_cfr_tf2.DeepCFRSolver(
      game,
      policy_network_layers=(64, 64, 64, 64),
      advantage_network_layers=(64, 64, 64, 64),
      num_iterations=FLAGS.num_iterations,
      num_traversals=FLAGS.num_traversals,
      learning_rate=1e-3,
      batch_size_advantage=2048,
      batch_size_strategy=2048,
      memory_capacity=1e6,
      policy_network_train_steps=5,
      advantage_network_train_steps=5,
      reinitialize_advantage_networks=True,
      infer_device="cpu",
      train_device="cpu")
  _, advantage_losses, policy_loss = deep_cfr_solver.solve()
  deep_cfr_solver.save_policy_network("tesorflow")
  
  deep_cfr_solver.train_policy_network_from_file("tesorflow")
  for player, losses in advantage_losses.items():
    print("ER IS GEWOON NIETS AAN HET GEBEUREN TF ")
    logging.info("Advantage for player %d: %s", player,
                 losses[:2] + ["..."] + losses[-2:])
    logging.info("Advantage Buffer Size for player %s: '%s'", player,
                 len(deep_cfr_solver.advantage_buffers[player]))
  logging.info("Strategy Buffer Size: '%s'",
               len(deep_cfr_solver.strategy_buffer))
  logging.info("Final policy loss: '%s'", policy_loss)



if __name__ == "__main__":
  app.run(main)
