import logging
import math
import torch
import numpy as np
from itertools import compress as mask

EPS = torch.tensor(1e-8)

log = logging.getLogger(__name__)


class FasterMCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, model, env):
        self.env = env
        self.model = model
        self.cache = {}  #{'Qa', 'N', 'Na'}

    def getActionProb(self, current_states, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        string_reps = self.env.to_hash(current_states)

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs
        
        count_logits = torch.stack([self.cache[s]['Na'] for s in string_reps])
        probs = torch.softmax(count_logits/temp, dim=1).clone()
        return probs
    
    def search(self, current_states, target_strings, c1=1.25, c2=19652, sim_action_select_temp=1, sample=False, forced_trajectory=None):
        '''
        current_states: padded Long tensor: [N, seq_len]
        target_states: [str]: [N]
        forced_trajectory: Long tensor: [N, traj_len], a sequence of actions to follow for each sequence.
        '''
        N = current_states.shape[0]
        vocab_size = self.env.getActionSize()
        hashes = self.env.to_hash(current_states) # [N] type str, using numpy since can address with index mask later
        
        policies, values = self.model.predict(current_states) # policies softmax distribution [N, vocab_size], values [N,1]
        valid_actions = self.env.validActions(current_states)
        policies[~valid_actions] = -float('Inf')
        values = values.squeeze(1) # [N]
        
        is_leaf_mask = torch.tensor([h not in self.cache for h in hashes])
        leaf_idxs = torch.arange(N)[is_leaf_mask]
        is_branch_mask = ~is_leaf_mask
        branch_idxs = torch.arange(N)[is_branch_mask]
        
        for leaf_idx in leaf_idxs:
            self.cache[hashes[leaf_idx]] = {
                'state': current_states[leaf_idx],
                'N': torch.tensor([1.0]),
                'Qa': torch.zeros(vocab_size),
                'Na': torch.zeros(vocab_size),
                'pred_V': values[leaf_idx],
                'avg_V':torch.tensor([0.0])
            }
        if is_leaf_mask.all():
            return values
        
        Ns = torch.stack([self.cache[hashes[i]]['N'] for i in branch_idxs])
        Ns_per_a = Ns.repeat(1,vocab_size)
        Qas = torch.stack([self.cache[hashes[i]]['Qa'] for i in branch_idxs])
        Nas = torch.stack([self.cache[hashes[i]]['Na'] for i in branch_idxs])
        
        # initialise all to be in exploration mode since the actions haven't been taken yet to exploit
#         UCB = self.c * policies[is_branch_mask] * Ns.sqrt()
        UCB = policies[is_branch_mask] * Ns.sqrt() * (c1 + torch.log((Ns+c2+1)/c2))
        prev_taken_a_mask = Qas != 0 # select previouslly taken actions since it will have a valid exploitation value
        UCB[prev_taken_a_mask] = Qas[prev_taken_a_mask] + policies[is_branch_mask][prev_taken_a_mask] * (Ns_per_a.sqrt()[prev_taken_a_mask] / (1 + Nas[prev_taken_a_mask])) * (c1 + torch.log((Ns_per_a.sqrt()[prev_taken_a_mask]+c2+1)/c2))
#         print(UCB)
        
        
        # doing a softmax will only matter for sampling since exponential is monotonic
#         UCB = torch.softmax(UCB/sim_action_select_temp, dim=1) # then renormalise     temp~0 sharpen maximally, almost argmax     temp > 1 flatten
        actions = torch.zeros((N,1), dtype=torch.long)
        actions[is_branch_mask] = UCB.multinomial(1) if sample else UCB.argmax(1).unsqueeze(1)
        
        
        if type(forced_trajectory) != type(None):
            actions[is_branch_mask] = forced_trajectory[is_branch_mask,0]
            forced_trajectory = forced_trajectory[:,1:]
            if forced_trajectory.shape[1] == 0:
                forced_trajectory = None
            
        is_terminal_mask = torch.zeros(N, dtype=torch.bool)
        rewards = torch.zeros(N)
                
        only_next_states, rewards[is_branch_mask], is_terminal_mask[is_branch_mask] = self.env.step(current_states[is_branch_mask], list(mask(target_strings,is_branch_mask)), actions[is_branch_mask])
        
        next_states = torch.zeros(N, only_next_states.shape[1], dtype=torch.long)
        next_states[is_branch_mask] = only_next_states
        
        values[is_terminal_mask] = rewards[is_terminal_mask]
        
        non_term_branch_idxs = torch.arange(N)[~is_terminal_mask & is_branch_mask]
        if len(non_term_branch_idxs) != 0:
            values[non_term_branch_idxs] = self.search(next_states[non_term_branch_idxs], 
                                                       list(mask(target_strings,~is_terminal_mask & is_branch_mask)), 
                                                       sim_action_select_temp=sim_action_select_temp, 
                                                       sample=sample,
                                                       forced_trajectory=forced_trajectory)
        
        for i in branch_idxs:
            self.cache[hashes[i]]['N'] += 1
            dense_Na = self.cache[hashes[i]]['Na']
            dense_Qa = self.cache[hashes[i]]['Qa']
            
            dense_Qa[actions[i]] = (dense_Na[actions[i]] * dense_Qa[actions[i]] + values[i]) / (dense_Na[actions[i]] + 1)
            self.cache[hashes[i]]['Qa'] = dense_Qa
            
            self.cache[hashes[i]]['avg_V'] += (values[i] - self.cache[hashes[i]]['avg_V']) / self.cache[hashes[i]]['N']

            dense_Na[actions[i]] += 1
            self.cache[hashes[i]]['Na'] = dense_Na
            
            del dense_Qa
            del dense_Na
        del policies
        
        return values
        
    
    def batchSearch(self, current_states, target_strings, sim_action_select_temp=1, sample=True):
        '''
        current_states: padded Long tensor: [N, seq_len]
        target_states: [str]: [N]
        '''
        N = current_states.shape[0]
        vocab_size = self.env.getActionSize()
        string_reps = self.env.to_hash(current_states) # [str*N]
        game_states = self.env.batchGameEnded(current_states, target_strings) # torch.tensor([N]) -1 or 0(game not finished) or 1
#         print('visited')
        # policies [N, vocab_size], values [N]
        policies, values = self.model.predict(current_states)
        values = values.squeeze(1)
                
        # assume all states are new
        states_visited = torch.zeros_like(game_states)
        for i in range(N):
            if string_reps[i] in self.cache:
                states_visited[i] = 1
        
        returned_values = game_states # set finished states into returned_values
        
        # game_states that are terminal (-1 or +1) or non visited states
        is_leaf_node_mask = (game_states!=0) | (states_visited==0) 
        # game_states that are still going (0) or non visited states
        non_terminal_leaf_states_mask = (game_states==0) & (states_visited==0) 
        # add non terminal leaf node values to returned_values
        returned_values[non_terminal_leaf_states_mask] = values[non_terminal_leaf_states_mask] 
        
        leaf_node_indices = torch.arange(N)[is_leaf_node_mask]
        for i in leaf_node_indices:
            self.cache[string_reps[i]] = {
                'state': current_states[i],
                'N': torch.tensor([1.0]),
                'Qa': torch.zeros(vocab_size).to_sparse(),
                'Na': torch.zeros(vocab_size).to_sparse(),
                'eV': values[i]
            }
        # return since all states have values
        if is_leaf_node_mask.all():
            return returned_values
        
        is_branch_node_mask = ~is_leaf_node_mask
        Ns = torch.stack([self.cache[string_reps[i]]['N'] for i in torch.arange(N)[is_branch_node_mask]])
        Qas = torch.stack([self.cache[string_reps[i]]['Qa'].to_dense() for i in torch.arange(N)[is_branch_node_mask]])
        Nas = torch.stack([self.cache[string_reps[i]]['Na'].to_dense() for i in torch.arange(N)[is_branch_node_mask]])
        
        # initialise all to be in exploration mode since the actions haven't been taken yet to exploit
        UCB = self.cpuct * policies[is_branch_node_mask] * Ns.sqrt()
        prev_taken_a_mask = Qas != 0 # select previouslly taken actions since it will have a valid exploitation value
        UCB[prev_taken_a_mask] = Qas[prev_taken_a_mask] + \
        self.cpuct * policies[prev_taken_a_mask] * Ns.sqrt().repeat(1,vocab_size)[prev_taken_a_mask] / (1 + Nas[prev_taken_a_mask])
#         print(Ns.repeat(1,vocab_size))
#         print(UCB)
        
        # Before sampling we can sharrpen the prob of selecting the best actions
        UCB = torch.softmax(UCB/sim_action_select_temp, dim=1) # then renormalise     temp~0 sharpen maximally, almost argmax     temp > 1 flatten
#         print(f'top nex actions {UCB.argsort(descending=True)[:,:10]} for states {current_states}')
        # a [N,1]
        a = UCB.multinomial(1) if sample else UCB.argmax(1).unsqueeze(1) # sellect next action for each state by either sampling or argmaxing
#         print(f'selecting action {a} for states {current_states}')
#         print(f'top scores from UCB: {UCB}')
        branch_next_states = self.env.batchNextStates(current_states[is_branch_node_mask], a) # get next state for the branch nodes
        
        branch_target_strings = [target_strings[i] for i in torch.arange(N)[is_branch_node_mask]]
        
        returned_values[is_branch_node_mask] = self.batchSearch(branch_next_states, 
                                                                branch_target_strings,
                                                                sim_action_select_temp=sim_action_select_temp,
                                                                sample=sample)
#         print('returned_values', returned_values)
        for i in torch.arange(N)[is_branch_node_mask]:
            self.cache[string_reps[i]]['N'] += 1
            dense_Na = self.cache[string_reps[i]]['Na'].to_dense()
            
            dense_Qa = self.cache[string_reps[i]]['Qa'].to_dense()
            
            dense_Qa[a[i]] = (dense_Na[a[i]] * dense_Qa[a[i]] + returned_values[i]) / (dense_Na[a[i]] + 1)
#             print(f'dense_Qa[a[i]] = ({dense_Na[a[i]]} * {dense_Qa[a[i]]} + {returned_values[i]}) / ({dense_Na[a[i]]} + 1) = {(dense_Na[a[i]] * dense_Qa[a[i]] + values[i]) / (dense_Na[a[i]] + 1)} Qa->= {dense_Qa[a[i]]}')
            self.cache[string_reps[i]]['Qa'] = dense_Qa.to_sparse()
#             print('Qa', self.cache[string_reps[i]]['Qa'], 'for a', a[i])
    
#             print('pre Na', self.cache[string_reps[i]]['Na'], 'for a', a[i])
            dense_Na[a[i]] += 1
            self.cache[string_reps[i]]['Na'] = dense_Na.to_sparse()
#             print('post Na', self.cache[string_reps[i]]['Na'], 'for a', a[i])
            
            del dense_Qa
            del dense_Na
#         del policies
        return returned_values
        
    
    def pi(self, s):
        s = self.env.stringRepresentation(s)
        return self.Psa[s]