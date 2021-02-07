import logging
import math
import torch
import numpy as np
from itertools import compress as mask
import networkx as nx
from IPython.display import Image, display

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
        
    def plot(self, from_state=None):
        OG=nx.OrderedGraph()
        added_chunks = set()
        states = sorted([node['state'].tolist() for node in self.cache.values()], key=len)
        if type(from_state)!=type(None):
            base_state_hash = ''.join(map(str, from_state.tolist()))
            def contains(lst1, lst2):
                ls1 = set(element for element in lst1 if element in lst2)
                ls2 = set(element for element in lst2 if element in lst1)
                return ls1 == ls2
            states = [s for s in states if base_state_hash in ''.join(map(str, s))]
        OG.add_node('[BOS]\n[1]')
        for s in states:
            s = [idx for idx in s if idx != 0]
            for i in range(len(s), 0, -1):
                parent_chunk = s[:i]
                new_chunk = s[i:]
                if tuple(parent_chunk) in added_chunks:
                    break
            added_chunks.add(tuple(s))
            to_str = lambda x: self.env.to_hash(torch.tensor([x]))[0]
            OG.add_node(to_str(s)+"\n"+str(s))
            OG.add_edge(to_str(parent_chunk)+"\n"+str(parent_chunk), to_str(s)+"\n"+str(s))
        
        pdot = nx.drawing.nx_pydot.to_pydot(OG)
        png_str = pdot.write_png('/tmp/graph.png')
        display(Image(filename='/tmp/graph.png'))