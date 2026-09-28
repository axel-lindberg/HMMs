#!/usr/bin/env python3

import math
import random

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random

from hmms.hmm1 import forward_algorithm
from hmms.hmm3 import baum_welch
from hmms.helper import log_likelihood

N_STATES = 4
START_STEP = 100
MAX_ITERS = 20
TOL = 1e-3

def random_row(n, rng):
    row = [rng.random() + 0.5 for _ in range(n)]
    s = sum(row)
    return [v / s for v in row]

class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """  
        rng = random.Random(0)
        
        self.models = [None] * N_SPECIES
        self.labelled = [[] for _ in range(N_SPECIES)]
        self.history = [[] for _ in range(N_FISH)]
        self.guessed = [False] * N_FISH
        self.rng = rng

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """

        # This code would make a random guess on each step:
        # return (step % N_FISH, random.randint(0, N_SPECIES - 1))
        
        for fish_id in range(N_FISH):
            self.history[fish_id].append(observations[fish_id])
 
        if step < START_STEP:
            return None
 
        for fish_id in range(N_FISH):
            if not self.guessed[fish_id]:
                self.guessed[fish_id] = True
                return (fish_id, self.find_best_type(self.history[fish_id]))
        return None
    
    def find_best_type(self, obs):
        best_type = None
        best_ll = float('-inf')
        for s in range(N_SPECIES):
            if self.models[s] is None:
                continue
            A, B, pi = self.models[s]
            try:
                _, c = forward_algorithm(A, B, pi, obs)
                ll = log_likelihood(c)
            except ZeroDivisionError:
                ll = float('-inf')
            if ll > best_ll:
                best_ll = ll
                best_type = s
 
        if best_type is None:
            return None
        return best_type

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """
        self.labelled[true_type].append(list(self.history[fish_id]))
        obs = [o for seq in self.labelled[true_type] for o in seq]
 
        if self.models[true_type] is None:
            A = [random_row(N_STATES, self.rng) for _ in range(N_STATES)]
            B = [random_row(N_EMISSIONS, self.rng) for _ in range(N_STATES)]
            pi = random_row(N_STATES, self.rng)
        else:
            A, B, pi = self.models[true_type]
 
        try:
            A, B, pi, _ = baum_welch(A, B, pi, obs, MAX_ITERS, TOL)
            self.models[true_type] = (A, B, pi)
        except ZeroDivisionError:
            pass
