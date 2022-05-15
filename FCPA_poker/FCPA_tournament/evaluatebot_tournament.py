#!/usr/bin/env python3
# encoding: utf-8
"""
tournament.py

Code to play a round robin tournament between fcpa agents.

Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2022 KU Leuven. All rights reserved.
"""
import importlib.util
import itertools
import logging
import os
import sys
from pathlib import Path

import click
import pandas as pd
import numpy as np
from tqdm import tqdm
import trail_agents.fcpa_random_agent as ra

import pyspiel
from open_spiel.python.algorithms.evaluate_bots import evaluate_bots

logger = logging.getLogger('be.kuleuven.cs.dtai.fcpa.tournament')


def load_agent(path, player_id):
    """Inintialize an agent from a 'fcpa_agent.py' file.
    """
    module_dir = os.path.dirname(os.path.abspath(path))
    sys.path.insert(1, module_dir)
    spec = importlib.util.spec_from_file_location("fcpa_agent", path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo.get_agent_for_tournament(player_id)


def load_agent_from_dir(agent_id, path):
    """Scrapes a directory for an fcpa agent.

    This function searches all subdirectories for an 'fcpa_agent.py' file and
    calls the ``get_agent_for_tournament`` method to create an instance of
    a player 1 and player 2 agent. If multiple matching files are found,
    a random one will be used.
    """
    agent_path = next(Path(path).glob('**/fcpa_agent.py'))
    print("AGENT PATH ISSSSS: ",agent_path)
    try:
        return {
            'id':  agent_id,
            'agent_p1': load_agent(agent_path, 0),
            'agent_p2': load_agent(agent_path, 1),
        }
    except Exception as e:
        logger.exception("Failed to load %s" % agent_id)


def play_match(game, agent1, agent2, seed=1234, rounds=100):
    """Play a set of games between two agents.
    """
    rng = np.random.RandomState(seed)
    results = []
    for _ in tqdm(range(rounds)):
       
        # Alternate between the two agents as p1 and p2
        for (p1, p2) in [(agent1, agent2), (agent2, agent1)]:
            try:
                returns = evaluate_bots(
                        game.new_initial_state(),
                        [p1['agent_p1'], p2['agent_p2']],
                        rng)
                #print("GAME OVER: GAINS P1: ", returns[0],' GAINS P2: ',returns[1])
                error = None
            except Exception as ex:
                logger.exception("Failed to play between %s and %s" % (agent1['id'], agent2['id']))
                template = "An exception of type {0} occurred. Message: {1}"
                error = template.format(type(ex).__name__, ex)
                returns = [None, None]
            finally:
                results.append({
                    "agent_p1": p1['id'],
                    "agent_p2": p2['id'],
                    "return_p1": returns[0],
                    "return_p2": returns[1],
                    "error": error
                })
    return results


def play_tournament(game, agents, seed=1234, rounds=100):
    """Play a round robin tournament between multiple agents.
    """
    rng = np.random.RandomState(seed)
    # Load each team's agent
    results = []
    for _ in tqdm(range(rounds)):
        for (agent1, agent2) in list(itertools.permutations(agents.keys(), 2)):
            returns = evaluate_bots(
                    game.new_initial_state(), 
                    [agents[agent1]['agent_p1'], agents[agent2]['agent_p2']], 
                    rng)
            results.append({
                "agent_p1": agent1,
                "agent_p2": agent2,
                "return_p1": returns[0],
                "return_p2": returns[1]
            })
    return results

def main():
    myagent = load_agent_from_dir("RO762346_DESTROYER","")
    print("BOT IS RETURNED")
    random_agent = {
            'id':  "opponent",
            'agent_p1': ra.get_agent_for_tournament(0),
            'agent_p2': ra.get_agent_for_tournament(1),
        } 
    
    
    fcpa_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=150 100,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,bettingAbstraction=fcpa)")
    game = pyspiel.load_game(fcpa_game_string)
    policy = pyspiel.PreferredActionPolicy([1, 0])
    check_call_agent ={
            'id':  "opponent",
            'agent_p1':  pyspiel.make_policy_bot(game, 0, 1234, policy),
            'agent_p2':  pyspiel.make_policy_bot(game, 1, 1234, policy),
        } 
    policy = pyspiel.PreferredActionPolicy([0, 1])
    

    fold_agent ={
            'id':  "opponent",
            'agent_p1':  pyspiel.make_policy_bot(game, 0, 1234, policy),
            'agent_p2':  pyspiel.make_policy_bot(game, 1, 1234, policy),
        } 
    print("KUNNEN GAME STARTEN")
    totalgames = 500000
    
    gameresults = play_match(game,myagent,check_call_agent,1234,totalgames)
    total = {
       'RO762346_DESTROYER':  0,
            'opponent': 0, 
    } 
    wins= {
        'RO762346_DESTROYER':  0,
            'opponent': 0,  
    }   
    
    for entry in gameresults:
        total[entry["agent_p2"]]+= int(entry["return_p2"])
        total[entry["agent_p1"]]+= int(entry["return_p1"])
        if int(entry["return_p2"])>int(entry["return_p1"]):
             
            wins[entry["agent_p2"]]+= 1
        else:
            wins[entry["agent_p1"]]+=1 
    #playing 2 games/iteration !         
    totalgames*=2        
    print('RO762346_DESTROYER gain: ',total['RO762346_DESTROYER'])
    print('OPPONENT gain: ',total['opponent'])       
    print()
    print('RO762346_DESTROYER average gain: ',total['RO762346_DESTROYER']/totalgames)
    print('OPPONENT average gain: ',total['opponent']/totalgames)       
    print()
    print('RO762346_DESTROYER wins: ',wins['RO762346_DESTROYER'],"/",totalgames)
    print('OPPONENT wins: ',wins['opponent'],"/",totalgames) 
         
        

if __name__ == '__main__':
    main()
