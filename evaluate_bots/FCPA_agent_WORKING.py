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

def find_between( s, first, last ):
      try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start-1:end+1]
      except ValueError:
        return ""
def simplify_info_key_fcpa(cur_player,state):
    # example [Round 0][Player: 0][Pot: 40000][Money: 19850 0][Private: TsTd][Public: ][Sequences: r20000]
    #to [Private: TT][Public: ]  
    #delete rounds player pot money
    info_key = state.information_state_string(cur_player) 
    legal_actions = state.legal_actions()
    
    #what actions are available 
    actionString = "[Actions: "
    for action in legal_actions:
      
        actionString += state.action_to_string(cur_player, action).split(' ')[1][5:]
        
    actionString += "]"    
    
    for i in range(4):
      rounds = find_between(info_key,"[","]") 
      info_key=info_key.replace(rounds,"",1)
    newer_key="" 
    #save private and public
    for i in range(2):
      rounds = find_between(info_key,"[","]") 
      info_key=info_key.replace(rounds,"",1)  
      rounds = simplify_card_string(rounds)
      newer_key+=rounds
    return newer_key+actionString
def simplify_card_string(cards):
      splitted = cards.split(": ")
      cards= splitted[1][:-1]
      pre  = splitted[0][1:] 
      newcards=""
      for i in range(0,len(cards),2):
            card = cards[i]
            newcards+=card  
      return "["+pre+": "+newcards+"]"    
            
            
      
    

      
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
       
        self.solver = external_sampling_mccfr.ExternalSamplingSolver(
                    self.game, external_sampling_mccfr.AverageType.SIMPLE)
        self.solver._infostates= self.infostates
        
        self.average_policy   = self.solver.average_policy()
    def restart_at(self, state):
        self.state= state

    def inform_action(self, state, player_id, action):
        
        
        if player_id==0:
            print("OPPONENT PLAYED :",state.action_to_string(0, action).split(' ')[1][5:])     
        elif player_id==1:
            print("MEYYYY WTF??????")   
        else:
            print("CARD DEALD")      
        
        self.state= state
        
    def get_cards(self,info_key):
        #wss 2 en 1 index want zo werkt splits i guess 
        pubcards= info_key.split("]")[1].replace("[Public: ","",1)
        privcards= info_key.split("]")[0].replace("[Private: ","",1) 
        
        return privcards+pubcards 
    def create_valid_info_key(self,privcards,pubcards,actionsstring):
        return "[Private: "+privcards+"][Public: "+pubcards+"]"+actionsstring
    
    def get_average_probabilities(self,info_key,state):
        
        legal_actions = state.legal_actions()
        retrieved_infostate = self.solver._infostates.get(info_key, None)
        if retrieved_infostate is None:
            return {a: 1 / len(legal_actions) for a in legal_actions}
        avstrat = (
            retrieved_infostate[mccfr.AVG_POLICY_INDEX] /
            retrieved_infostate[mccfr.AVG_POLICY_INDEX].sum())
        return {legal_actions[i]: avstrat[i] for i in range(len(legal_actions))}   
    
    def step(self, state):
        print()
        print("––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––")
        print("START OUR AGENT TURN: ")
        
        cur_player = state.current_player()

        info_state_key = simplify_info_key_fcpa(cur_player,self.state) 
        info_key = state.information_state_string(cur_player) 
        cards = self.get_cards(info_state_key)
        legal_actions = state.legal_actions()
        num_legal_actions = len(legal_actions)
        
        privcards = cards[:2]
        pubcards = cards[2:]
        print("TOTAL INFO_STATE: ",info_key)
        print("INFO STATE: ",info_state_key)
        
        all_subcards=[]
        if len(pubcards)<=3:
            chances= []
            actions= []
            act_prob= self.get_average_probabilities(info_state_key,state) 
            
            for action in  act_prob:
                chance =  act_prob[action]
                chances.append(chance)
                actions.append(action)
                
                 
            chosen_action = np.random.choice(np.array(actions), p=np.array(chances))
            print("POLICY == ",np.array(chances))
            print("CHOSE ACTION: ",state.action_to_string(cur_player, chosen_action).split(' ')[1][5:]," WITH ACTIONPROBABILITIES:")
            pol_indx=0
            for action in actions:
                print(state.action_to_string(cur_player, action).split(' ')[1][5:]+" : ",np.array(chances)[pol_indx])
                pol_indx+=1
            print("END OUR AGENT TURN")
            print("––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––")       
            return chosen_action
        actionString = "[Actions: "
        for action in legal_actions:
            actionString += state.action_to_string(cur_player, action).split(' ')[1][5:] 
        actionString += "]"  
        if (len(pubcards)==4):
            #4 public cards
            for i in range(len(pubcards)):    
                all_subcards.append(pubcards.replace(pubcards[i],"",1))
                
                
        else:
            #5 public cards  
            for i in range(len(pubcards)):    
                first=i
                second= (i+1)//len(pubcards)
                newstring= pubcards.replace(pubcards[first],"",1)
                all_subcards.append(newstring.replace(pubcards[second],"",1))
        
        policys=[]
        maxes = []
        sub_keys=[]
        print("ALL SUBCARDS: ",all_subcards)
        for subcards in all_subcards:
            value=0

            print()
            sub_info_key =  self.create_valid_info_key(privcards,subcards,actionString)  
            sub_keys.append(sub_info_key)
            print("SUBINFO KEY: ",sub_info_key) 
            act_prob= self.get_average_probabilities(sub_info_key,state)
            
            chances=[]
            actions=[] 
            for action in  act_prob:
                chance =  act_prob[action]
                chances.append(chance)
                actions.append(action)
            policy=np.array(chances)
            
            policys.append(policy)
            #selection metric select on highest average regret if the policy is played out 
            for idx in range(num_legal_actions):
                infos=self.solver._lookup_infostate_info(sub_info_key,num_legal_actions) 
                avret = infos[mccfr.REGRET_INDEX][idx]*policy[idx]
                value+=avret
            value= value/infos[mccfr.AVG_POLICY_INDEX].sum()        
           
            print("WITh A VALUE OF: ",value)
            print("WITH POLICY: ",policy)
            print()
            maxes.append(value)
        
        #choose the policy with a maximum chance      
        bestpolicy= policys[maxes.index((max(maxes)))]
        best_key= sub_keys[maxes.index((max(maxes)))]
        chosen_action = np.random.choice(np.array(actions), p=np.array(chances))
        
        print("BEST SUBKEY: ",best_key)
        print("CHOSE ACTION: ",state.action_to_string(cur_player, chosen_action).split(' ')[1][5:]," WITH ACTIONPROBABILITIES:")
        
        pol_indx=0
        for action in legal_actions:
            print(state.action_to_string(cur_player, action).split(' ')[1][5:]+" : ", bestpolicy[pol_indx])
            pol_indx+=1
        print("END OUR AGENT TURN")
        print("––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––") 
        print()   
        return chosen_action
     
                 
           