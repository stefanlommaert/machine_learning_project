#!/usr/bin/env python3
# encoding: utf-8
"""
fcpa_agent.py

Extend this class to provide an agent that can participate in a tournament.

Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2021 KU Leuven. All rights reserved.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import numpy as np



import pyspiel

from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr


def get_agent_for_tournament(player_id):
    """Change this function to initialize your agent.
    This function is called by the tournament code at the beginning of the
    tournament.

    :param player_id: The integer id of the player for this bot, e.g. `0` if
        acting as the first player.
    """
    my_player = Agent(player_id)
    return my_player


class Agent(pyspiel.Bot):
    """Agent template"""

    def __init__(self, player_id):
        #Training
        pyspiel.Bot.__init__(self)
        self.player_id = player_id
        self.game = pyspiel.load_game("leduc_poker")
        self.es_solver = external_sampling_mccfr.ExternalSamplingSolver(
        self.game, external_sampling_mccfr.AverageType.Simple)
        for _ in range(1000):
            self.es_solver.iteration()
        conv = exploitability.nash_conv(self.game, self.es_solver.average_policy())
        print("TRAINING DONE")
        self.state= self.game.new_initial_state()
        print("Iteration {} exploitability {}".format(10, conv))
        self.average_policy = self.es_solver.average_policy()
        np.save('leduc_poker/leduc_agent_infostats.npy', self.average_policy._infostates)

    def restart_at(self, state):
        self.state= state

    def inform_action(self, state, player_id, action):
        print("Inform action; for ", player_id)
        print("ACTION: ",action)
        print("STATE: ", state)
        
        self.state= state
        

    def step(self, state):
        print("average_policy_values :",self.es_solver.average_policy().action_probabilities(state))
        for key in self.es_solver.average_policy().action_probabilities(state):
            print("key :",key)
            value = self.es_solver.average_policy().action_probabilities(state)[key]
            print("value :",value)
            randfloat = random.random()
            if randfloat < value:
                print("RANDOM CHOSEN ACTION: ",key)
                return int(key)
        



