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

"""NFSP agents trained on Kuhn Poker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
from matplotlib import pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import nfsp
from open_spiel.python.algorithms import random_agent
FLAGS = flags.FLAGS

flags.DEFINE_integer("num_train_episodes", int(1e5),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 1000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_list("hidden_layers_sizes", [
    128,
], "Number of hidden units in the avg-net and Q-net.")
#buffer first needs to full so the first few values can't be computed 
flags.DEFINE_integer("replay_buffer_capacity", int(2e2),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e2),
                     "Size of the reservoir buffer.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")
### Settings for Evaluation ###
# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_every = 100
save_plot_every = 1000
evaluate_num = 10000

# The paths for saving the logs and learning curves
root_path = './experiments/leduc_holdem_nfsp_result/'
log_path = root_path + 'log.txt'
csv_path = root_path + 'performance.csv'
figure_path = root_path + 'figures/'



class NFSPPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies, mode):
    game = env.game
    player_ids = [0, 1]
    super(NFSPPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._mode = mode
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    with self._policies[cur_player].temp_mode_as(self._mode):
      p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict


def main(unused_argv):
  game = "leduc_poker"
  num_players = 2

  env_configs = {"players": num_players}
  env = rl_environment.Environment(game, **env_configs)
  
  eval_env =  rl_environment.Environment(game, **env_configs)
  
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
  kwargs = {
      "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
      "epsilon_decay_duration": FLAGS.num_train_episodes,
      "epsilon_start": 0.06,
      "epsilon_end": 0.001,
  }
  Step_values=[]
  NashConv_values=[]
  
  
  with tf.Session() as sess:
    # pylint: disable=g-complex-comprehension
    agents = [
        nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
                  FLAGS.reservoir_buffer_capacity, FLAGS.anticipatory_param,
                  **kwargs) for idx in range(num_players)
    ]
    expl_policies_avg = NFSPPolicies(env, agents, nfsp.MODE.average_policy)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    
    sess.run(tf.global_variables_initializer())
    #ENV veranderen naar een open_spiel friendly environment
    # random_agents = [
    #     random_agent.RandomAgent(idx, num_actions) for idx in range(num_players)
    # ]
    
    
    episode_num = 10000000 # set the episode number
    
    for episode in range(episode_num):
          time_step = env.reset()
          while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)
            #print("episode: ",episode," player: ",player_id," chose action: ",action_list)
            #Episode is over, step all agents with final info state.
          for agent in agents:
            agent.step(time_step)
    
    
          if (episode + 1) % 100 == 0:
            losses = [agent.loss for agent in agents]
            logging.info("Losses: %s", losses)
            expl = exploitability.nash_conv(env.game, expl_policies_avg)
            logging.info("[%s] NASH_CONV AVG %s", episode + 1, expl)
            logging.info("_____________________________________________")
            
            Step_values.append(episode+1)
            NashConv_values.append(expl)
            if (episode+1) % save_plot_every == 0 and episode > 0:
                  
                  x = np.array(Step_values)
                  y = np.array(NashConv_values)
                
                  plt.figure()
                  ax = plt.subplot(111)
                  plt.title("NFSP khun_poker NashConv convergence")
                  plt.xlabel('Step')
                  plt.ylabel('NashConv')
                  ax.set_xscale('log')
                  ax.set_yscale('log') 
                  #needs to 'wind up' the first 10 000 iterations
                  ax.set_xlim(min(Step_values),max(Step_values))  
                  ax.set_ylim(min(NashConv_values),1)  
                  ax.plot(x,y)
                  plt.savefig("leduc_poker/figures/"+str(episode)+'.png')
                  plt. clf()
                  

          

         
    x = np.array(Step_values)
    y = np.array(NashConv_values)
  
    plt.figure()
    ax = plt.subplot(111)
    plt.title("NFSP khun_poker NashConv convergence")
    plt.xlabel('Step')
    plt.ylabel('NashConv')
    ax.set_xscale('log')
    ax.set_yscale('log') 
    #needs to 'wind up' the first 10 000 iterations
    ax.set_xlim(min(Step_values),max(Step_values))  
    ax.set_ylim(min(NashConv_values),1)  
    ax.plot(x,y)
    plt.show()
    
if __name__ == "__main__":
  app.run(main)