import logging
import math
import torch
import numpy as np

EPS = torch.tensor(1e-8)

log = logging.getLogger(__name__)


class FastMCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, model, cpuct=0.5):
        self.game = game
        self.model = model
        self.cpuct = cpuct
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Psa = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vsa = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonical_state, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        s = self.game.stringRepresentation(canonical_state)

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        count_logits = self.Nsa[s] ** (1. / temp)
        probs = count_logits / count_logits.sum()
        return probs

    def search(self, canonical_state):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the value of the current canonical_state
        """

        s = self.game.stringRepresentation(canonical_state)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonical_state)
        if self.Es[s] != 0:
            # terminal node
            return self.Es[s]

        if s not in self.Psa:
            # leaf node
            model_input = self.game.prepStateForModel(canonical_state)
            self.Psa[s], v = self.model.predict(model_input)
            valid_actions = self.game.getValidMoves(canonical_state)
            self.Psa[s] = self.Psa[s] * valid_actions  # masking invalid moves
            sum_Psa_s = torch.sum(self.Psa[s])
            if sum_Psa_s > 0:
                self.Psa[s] /= sum_Psa_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Psa[s] = self.Psa[s] * valid_actions
                self.Psa[s] /= torch.sum(self.Psa[s])

            self.Vsa[s] = valid_actions
            self.Ns[s] = torch.tensor(0.0)#, device=self.model.device)
            self.Nsa[s] = torch.zeros(self.game.getActionSize())#, device=self.model.device)
            self.Qsa[s] = torch.zeros(self.game.getActionSize())#, device=self.model.device)
            return v

        valids = self.Vsa[s]
        cur_best = -float('inf')
        best_act = -1
        
        # UCB -> upper confidence bound
        UCB = self.cpuct * self.Psa[s] * torch.sqrt(self.Ns[s] + EPS)
        
        prev_taken_a_mask = self.Qsa[s] != 0
        UCB[prev_taken_a_mask] = self.Qsa[s][prev_taken_a_mask] + self.cpuct * self.Psa[s][prev_taken_a_mask] * torch.sqrt(self.Ns[s]) / (1 + self.Nsa[s][prev_taken_a_mask])

        # pick the action with the highest upper confidence bound
        a = torch.argmax(UCB)
        next_s = self.game.getNextState(canonical_state, a)
        
        v = self.search(next_s)
        
        self.Nsa[s][a] += 1
        self.Qsa[s][a] = (self.Nsa[s][a] * self.Qsa[s][a] + v) / (self.Nsa[s][a] + 1)
        self.Ns[s] += 1

        return v
    
    def pi(self, s):
        s = self.game.stringRepresentation(s)
        return self.Psa[s]