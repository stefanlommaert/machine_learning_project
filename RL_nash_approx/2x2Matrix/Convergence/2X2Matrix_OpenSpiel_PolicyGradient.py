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

"""Python spiel example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from absl import app
from absl import flags
from matplotlib import pyplot as plt
import numpy as np
import pyspiel


import tensorflow.compat.v1 as tf
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import eva
from open_spiel.python.algorithms import policy_gradient
from open_spiel.python.environments import catch
from open_spiel.python import rl_environment


FLAGS = flags.FLAGS
flags.DEFINE_string("game", "matrix_pd", "Name of the game.") #WELKE GAME WILLEN WE TESTEN?
flags.DEFINE_integer("num_episodes", int(1e5), "Number of train episodes.") #HOEVEEL ITERATIES?
flags.DEFINE_integer("eval_every", int(1e2), #HOEVAAK BENCHMARKS RUNNEN?
                     "How often to evaluate the policy.")
flags.DEFINE_enum("algorithm", "a2c", ["dqn", "rpg", "qpg", "rm", "eva", "a2c"],
                  "Algorithms to run.") #WELKE ALGORITM VOOR DE PLAYER GEBRUIKEN?


P2_averages=[]
P1_averages=[]



def _eval_agent(env, agent1,agent2, num_episodes):#EVALUATION SCRIPT VOOR DE LEARNERS
  """Evaluates `agent` for `num_episodes`."""
  rewards = 0.0
  for _ in range(num_episodes):
    time_step = env.reset()
    episode_reward = 0
    while not time_step.last():
      agent_output = agent1.step(time_step, is_evaluation=True)
      agent2_output= agent2.step(time_step, is_evaluation=True)
      time_step = env.step((agent_output.action,agent2_output.action))
      print("rewards",time_step.rewards)
      episode_reward += time_step.rewards[0]
    rewards += episode_reward
  return rewards / num_episodes


def main_loop(unused_arg):
  """Trains a Policy Gradient agent in the catch environment."""
  #env = catch.Environment()
  
  env_configs = {}
  # plot_name = 'Dispersion game'
  # row_player = [[-1,1],[1,-1]]
  # vector_player = [[-1,1],[1,-1]]

  # plot_name = 'matching pennies'
  # row_player = [[1,-1],[-1,1]]
  # vector_player = [[-1,1],[1,-1]]

  # plot_name = 'Battle of the sexes'
  # row_player = [[3,0],[0,2]]
  # vector_player = [[2,0],[0,3]]

  plot_name = 'Subsidy game'
  row_player = [[10,0],[11,12]]
  vector_player = [[10,11],[0,12]]
  game = pyspiel.create_matrix_game(row_player, vector_player)
  
  env = rl_environment.Environment(game, **env_configs) 
  num_actions = env.action_spec()["num_actions"]

  # agents = [
  #     random_agent.RandomAgent(player_id=i, num_actions=num_actions)
  #     for i in range(FLAGS.num_players)
  # ]
  logging.info("Env specs: %s", env.observation_spec())
  logging.info("Action specs: %s", env.action_spec())
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  train_episodes = FLAGS.num_episodes

  with tf.Session() as sess:
    if FLAGS.algorithm in {"rpg", "qpg", "rm", "a2c"}:
      agent = policy_gradient.PolicyGradient(
          sess,
          player_id=0,
          info_state_size=info_state_size,
          num_actions=num_actions,
          loss_str=FLAGS.algorithm,
          hidden_layers_sizes=[128, 128],
          batch_size=128,
          entropy_cost=0.01,
          critic_learning_rate=0.1,
          pi_learning_rate=0.1,
          num_critic_before_pi=3)
      agent2 = policy_gradient.PolicyGradient(
          sess,
          player_id=1,
          info_state_size=info_state_size,
          num_actions=num_actions,
          loss_str=FLAGS.algorithm,
          hidden_layers_sizes=[128, 128],
          batch_size=128,
          entropy_cost=0.01,
          critic_learning_rate=0.1,
          pi_learning_rate=0.1,
          num_critic_before_pi=3)
    elif FLAGS.algorithm == "dqn": 
      agent = dqn.DQN( #DQN = Deep Q learning 
          sess,
          player_id=0,
          state_representation_size=info_state_size,
          num_actions=num_actions,
          learning_rate=0.1,
          replay_buffer_capacity=10000,
          hidden_layers_sizes=[32, 32],
          epsilon_decay_duration=2000,  # 10% total data
          update_target_network_every=250)
      agent2 = dqn.DQN(
          sess,
          player_id=1,
          state_representation_size=info_state_size,
          num_actions=num_actions,
          learning_rate=0.1,
          replay_buffer_capacity=10000,
          hidden_layers_sizes=[32, 32],
          epsilon_decay_duration=2000,  # 10% total data
          update_target_network_every=250)
    elif FLAGS.algorithm == "eva":
      agent = eva.EVAAgent(
          sess,
          env,
          player_id=0,
          state_size=info_state_size,
          num_actions=num_actions,
          learning_rate=1e-3,
          trajectory_len=2,
          num_neighbours=2,
          mixing_parameter=0.95,
          memory_capacity=10000,
          dqn_hidden_layers=[32, 32],
          epsilon_decay_duration=2000,  # 10% total data
          update_target_network_every=250)
      agent2 = eva.EVAAgent(
          sess,
          env,
          player_id=1,
          state_size=info_state_size,
          num_actions=num_actions,
          learning_rate=1e-3,
          trajectory_len=2,
          num_neighbours=2,
          mixing_parameter=0.95,
          memory_capacity=10000,
          dqn_hidden_layers=[32, 32],
          epsilon_decay_duration=2000,  # 10% total data
          update_target_network_every=250) 
    else:
      raise ValueError("Algorithm not implemented!")

    sess.run(tf.global_variables_initializer())

    # Train agent
    for ep in range(train_episodes):
      time_step = env.reset()
      while not time_step.last():
        agent_output = agent.step(time_step)
        
        agent2_output = agent2.step(time_step)
        
        # print("agent output: ", agent_output)
        # print("agent2 output: ", agent2_output)
        
        agent_action  = agent_output.action
        agent2_action = agent2_output.action
        
        # print("agent action: ", agent_action)
        # print("agent2 action: ", agent2_action)
         
        time_step = env.step((agent_action,agent2_action)) #hier gaan we er 2 moeten opgeven een action van beiden players
       
        
      # Episode is over, step agent with final info state.
      agent.step(time_step)
      agent2.step(time_step)
      

      if ep and ep % FLAGS.eval_every == 0:
        logging.info("-" * 80)
        logging.info("Episode %s", ep)
        # logging.info("Loss: %s", agent.loss)
        # avg_return = _eval_agent(env, agent,agent2, 5)
        # logging.info("Avg return: %s", avg_return)
        P1_averages.append(agent_output.probs[0]) #
        P2_averages.append(agent2_output.probs[0])
        print("player1 probability of action1: ",agent_output.probs[0])
        print("player2 probability of action2: ",agent2_output.probs[0])
    plt.axis('square')
    plt.title("Policy Gradient self-play: "+plot_name+ ", episodes = "+"1e5")
    plt.xlabel('Player 1, probability of action 1')
    plt.ylabel('Player 2, probability of action 1')
    plt.axis([0, 1, 0, 1])
    
    
    # create some x data and some integers for the y axis
    x = np.array(P1_averages)
    y = np.array(P2_averages)
    
    
    
    # plot the data
    plt.plot(x,y)
    
    # number_label=measurement_stepsize om de hoeveelste plot level aan te duiden maar is nogal scuffed
    
    
    label  = "start"
    
    plt.annotate(label, # this is the texst
                (x[0],y[0]), # these are the coordinates to position the label
                textcoords="offset points", # how to position the texst
                xytext=(0,1), # distance from texst to points (xs,ys)
                ha='center',
                fontsize=10) # horizontal alignment can be left, right or center
    label  = "end"
    
    plt.annotate(label, # this is the texst
                (x[-1],y[-1]), # these are the coordinates to position the label
                textcoords="offset points", # how to position the texst
                xytext=(0,1), # distance from texst to points (xs,ys)
                ha='center',
                fontsize=10) # horizontal alignment can be left, right or center
    plt.show()    


if __name__ == "__main__":
  app.run(main_loop)

    
