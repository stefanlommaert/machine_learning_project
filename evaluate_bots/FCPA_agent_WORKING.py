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

from open_spiel.python.algorithms import mccfr
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
        pyspiel.Bot.__init__(self)
        fcpa_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=2,blind=150 100,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,bettingAbstraction=fcpa)")
        self.game = pyspiel.load_game(fcpa_game_string)
        self.infostates=np.load("FCPA_poker/FCPA_tournament/fcpa_agent/infostates/full_fpca_agent_infostats.npy",allow_pickle=True)[()] #allow the magic pickle
        self.state= self.game.new_initial_state()
        
        self.average_policy = mccfr.AveragePolicy(self.game, list(range(self.game.num_players())),
                            self.infostates)
        self.solver = external_sampling_mccfr.ExternalSamplingSolver(
                    self.game, external_sampling_mccfr.AverageType.SIMPLE)
        self.solver._infostates= self.infostates
    def restart_at(self, state):
        self.state= state

    def inform_action(self, state, player_id, action):
     
        self.state= state
        
    def get_cards(self,info_key):
        #wss 2 en 1 index want zo werkt splits i guess 
        pubcards= info_key.split("]")[1].replace("[Public: ","")
        privcards= info_key.split("]")[0].replace("[Private: ","") 
        
        return privcards+pubcards 
    def create_valid_info_key(self,privcards,pubcards,actionsstring):
        return "[Private: "+privcards+"][Public: "+pubcards+"]"+actionsstring
        
    def step(self, state):
        print("STATE: ",state)
        cur_player = state.current_player()

        info_state_key = external_sampling_mccfr.simplify_info_key_fcpa(cur_player,self.state) 
        cards = self.get_cards(info_state_key)
        legal_actions = state.legal_actions()
        num_legal_actions = len(legal_actions)
        
        privcards = cards[:2]
        pubcards = cards[2:]
        print("INFO STATE: "+info_state_key)
        
        all_subcards=[]
        if len(pubcards)<=3:
            infostate_info = self.solver._lookup_infostate_info(info_state_key,
                                                    num_legal_actions)
            policy = self.solver._regret_matching(infostate_info[mccfr.REGRET_INDEX],
                                    num_legal_actions)
           
            action_idx = np.random.choice(np.arange(num_legal_actions), p=policy)
            print("CHOSE ACTION: ",state.action_to_string(cur_player, legal_actions[action_idx]).split(' ')[1][5:]," WITH ACTIONPROBABILITIES:")
            pol_indx=0
            for action in legal_actions:
                print(state.action_to_string(cur_player, action).split(' ')[1][5:]+" : ",policy[pol_indx])
                pol_indx+=1
            return legal_actions[action_idx]
        actionString = "[Actions: "
        for action in legal_actions:
            actionString += state.action_to_string(cur_player, action).split(' ')[1][5:] 
        actionString += "]"  
        if (len(pubcards)==4):
            #4 public cards
            for i in range(len(pubcards)):    
                all_subcards.append(pubcards.replace(pubcards[i],""))
                
                
        else:
            #5 public cards  
            
            for i in range(len(pubcards)):    
                first=i
                second= (i+1)//len(pubcards)
                newstring= pubcards.replace(pubcards[first],"")
                all_subcards.append(newstring.replace(pubcards[second],""))
        #go over all subentries of 3 cards to find the action with the highest probability (with the assumption that the highest probability will probility be a good option (a pair for example)).
        policys=[]
        maxes = []
        print("ALL SUBCARDS: ",all_subcards)
        for subcards in all_subcards:
            sub_info_key =  self.create_valid_info_key(privcards,subcards,actionString)  
            print("SUBINFO KEY: ",sub_info_key) 
            infostate_info = self.solver._lookup_infostate_info(sub_info_key,
                                                    num_legal_actions)
            policy = self.solver._regret_matching(infostate_info[mccfr.REGRET_INDEX],
                                    num_legal_actions)
            
            policys.append(policy)
            maxes.append(np.amax(policy))
        
        #choose the policy with a maximum chance      
        bestpolicy= policys[maxes.index((max(maxes)))]
      
        action_idx = np.random.choice(np.arange(num_legal_actions), p=bestpolicy)
        print("CHOSE ACTION: ",state.action_to_string(cur_player, legal_actions[action_idx]).split(' ')[1][5:]," WITH ACTIONPROBABILITIES:")
        pol_indx=0
        for action in legal_actions:
            print(state.action_to_string(cur_player, action).split(' ')[1][5:]+" : ", bestpolicy[pol_indx])
            pol_indx+=1
        return legal_actions[action_idx]
     
                 
           