import random
import re
import copy
import torch
from time import perf_counter
import tqdm
from src.FastMCTS import FastMCTS
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer, Callback, seed_everything
from src.models_and_transforms.text_transforms import Numericalise_Transform, Scratch_Pad_Sequence_Policy_Creator_Transform, \
                                                        Rename_Transform, Class_Rebalance_Transform
from src.pipe_datasets import Scratch_Pad_Policy_Dataset


class ScratchPadGame():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.
    Use 1 for player1 and -1 for player2.
    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self, tokenizer, prompt_target_pairs=None, device='cpu'):
        self.tokenizer = tokenizer
        self.device=device
        self.action_size = len(self.tokenizer.get_vocab())
        self.prompt_target_pairs = prompt_target_pairs
        if prompt_target_pairs is None:
            self.prompt_target_pairs = [('What is 3+4?','What is 3+4?7[EOS]'), ('What is 2+3?','What is 2+3?5[EOS]')]
        self.max_token_length = 20
        self.exec_token_id = self.tokenizer.get_vocab()['>>>']
        self.nl_token_id = self.tokenizer.get_vocab()['\n']
        
        self.start_scratch_pad_id = self.tokenizer.get_vocab()['[SP]']
        self.end_scratch_pad_id = self.tokenizer.get_vocab()['[ESP]']
        self.value_id = self.tokenizer.get_vocab()['[VALUE]']
        self.mask_id = self.tokenizer.get_vocab()['[MASK]']
        self.eos_id = self.tokenizer.get_vocab()['[EOS]']
        self.pad_id = self.tokenizer.get_vocab()['[PAD]']

    def randInitProblem(self):
        """
        Sample a 
        Returns:
            canonical_state: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        start_string, target_string = random.choice(self.prompt_target_pairs)
        canonical_state = self.toCanonical(start_string, target_string)
        return canonical_state
    
    def batchInitialProblems(self, batch_size):
        pairs = random.choices(self.prompt_target_pairs, k=batch_size)
        starting_states = [ex[0] for ex in pairs]
        target_states = [ex[1] for ex in pairs]
        tokenized_starting_states = [ex.ids for ex in self.tokenizer.encode_batch(starting_states)]
        tensor_starting_states = torch.tensor(tokenized_starting_states, device=self.device)
        return tensor_starting_states, target_strings
        
    
    def toCanonical(self, start_string, target_string):
        start_state = torch.tensor(self.tokenizer.encode(stmnt_out + '\n'), device=self.device)
        target_state = torch.tensor(self.tokenizer.encode(stmnt_out + '\n'), device=self.device)
        return {'current_state':start_state, 'target_state':target_state}

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return self.action_size
    
    def getNextStateTransform(self, canonical_state_batch):
        next_states = []
        for canonical_state in canonical_state_batch:
            next_states.append(self.getNextState(canonical_state, canonical_state['action']))
        return next_states
    
    def batchNextStates(self, current_states, actions):
        """
        Input:
            current_states: Long tensor: [N, seq_len]
            actions: action taken: Long tensor: [N,1]
        Returns:
            nextState: state after applying action: torch.tensor([1,2,3,4])
        """
        new_states = []
        for current_state, action in zip(current_states, actions):
            in_sp_mask = self.scratchPadMask(current_state.unsqueeze(0)).squeeze(0) # True means in sp token, False means out
            if action != self.exec_token_id or not in_sp_mask[-1] == True:
                new_state = torch.cat((current_state, action))
                new_states.append(new_state)
                continue
            # this means we are in an execution situation 

            sequence = self.tokenizer.decode(torch.cat((current_state, action)).tolist())
            prior_scratch_pad_sequence, _, last_scratch_pad_sequence = sequence.rpartition('[SP]')
            prior_scratch_pad_sequences = re.findall(r'\[SP\]([^]]*)\[ESP\]', prior_scratch_pad_sequence)
            all_statements = ''.join(prior_scratch_pad_sequences + [last_scratch_pad_sequence])
            individual_statements = re.split(r'>>>.*\n|>>>', all_statements)
            code = '\n'.join([s for s in individual_statements if s])
            stmnt_out = self.scratch_pad_exec(code)
            tokenized_stmnt_out = torch.tensor(self.tokenizer.encode(stmnt_out + '\n'), device=self.device)
            new_state = torch.cat((current_state, action.reshape(1), tokenized_stmnt_out))
            new_states.append(new_state)
        
        next_states = torch.nn.utils.rnn.pad_sequence([new_state.flip(0) for new_state in new_states], 
                                                 padding_value=self.pad_id, batch_first=True).flip(1)
        return next_states

    def getNextState(self, canonical_state, action):
        """
        Input:
            state: current sequence: torch.tensor([1,2,3])
            action_index: action taken: torch.tensor(4)
        Returns:
            nextState: state after applying action: torch.tensor([1,2,3,4])
        """
        current_state = canonical_state['current_state']
        in_sp_mask = self.scratchPadMask(current_state)
        
        if action != self.exec_token_id or not in_sp_mask[-1] == True:
            new_state = torch.cat((current_state, action.reshape(1)))
            return {'current_state':new_state, 'target_state':canonical_state['target_state']}
        # this means we are in an execution situation 
        
        sequence = self.tokenizer.decode(current_state)
        prior_scratch_pad_sequence, _, last_scratch_pad_sequence = sequence.rpartition('<ScratchPad>')
        prior_scratch_pad_sequences = re.findall(r'\<ScratchPad\>([^]]*)\</ScratchPad\>', prior_scratch_pad_sequence)
        all_statements = ''.join(prior_scratch_pad_sequences + [last_scratch_pad_sequence])
        individual_statements = re.split(r'>>>.*\n|>>>', all_statements)
        stmnt_out = self.scratch_pad_exec('\n'.join(individual_statements))
        tokenized_stmnt_out = self.tokenizer.encode(stmnt_out + '\n', return_tensors='pt')[0]
        current_state = canonical_state['current_state']
        new_state = torch.cat((current_state, action.reshape(1), tokenized_stmnt_out))
        return {'current_state':new_state, 'target_state':canonical_state['target_state']}
        
    def in_scratch_pad(self, state):
        if '<ScratchPad>' not in state:
            return False
        prior_scratch_pad_sequence, _, last_scratch_pad_sequence = state.rpartition('<ScratchPad>')
        if '</ScratchPad>' in last_scratch_pad_sequence:
            return False
        return True
    
    def scratch_pad_exec(self, code):
        if not code:
            return ''
        try:
            prior_code, _, last_line = code.rpartition('\n')
            exec(f'{prior_code}\nglobal __i__; __i__ = {last_line}')
            global __i__
            return str(__i__)
        except Exception as e:
            if hasattr(e,'msg'):
                return "ERROR: " + e.msg
            return "ERROR: " + str(e)
    
    def getValidMovesTransform(self, canonical_state):
        for canonical_state in canonical_state_batch:
            canonical_state['valid_moves'] = self.getValidMoves(canonical_state)
        return canonical_state_batch
    
    def getValidMoves(self, canonical_state):
        """
        Input:
            canonical_state: current board
        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        return torch.ones(self.getActionSize(), device=self.device)
    
    def scratchPadMaskTransform(self, current_state_batch):
        for canonical_state in canonical_state_batch:
            canonical_state['scratch_pad_mask'] = self.scratchPadMask(canonical_state['current_state'])
        return canonical_state_batch
    
    def scratchPadMask(self, current_states):
        N = current_states.shape[0]
        is_SP_start_tok = current_states == self.start_scratch_pad_id
        is_SP_end_tok = current_states == self.end_scratch_pad_id
        is_SP_end_tok = torch.cat((torch.tensor([[False]]*N, device=self.device), is_SP_end_tok[:,:-1]), dim=1)
        
        SP_toks = is_SP_start_tok + is_SP_end_tok
        SP_mask = torch.cumsum(SP_toks.to(torch.int), dim=1) % 2 == 1
        return SP_mask
    
    def autoGeneratedMaskTransform(self, current_state_batch):
        for canonical_state in canonical_state_batch:
            canonical_state['auto_gen_mask'] = self.autoGeneratedMask(canonical_state['current_state'])
        return canonical_state_batch
    
    def autoGeneratedMask(self, current_state):
        state_len = current_state.shape[0]
        is_auto_gen_start_tok = current_state == self.exec_token_id
        is_auto_gen_start_tok = torch.cat((torch.tensor([False], device=self.device), is_auto_gen_start_tok[:-1]))
        
        is_auto_gen_end_tok = current_state == self.nl_token_id
        is_auto_gen_end_tok = torch.cat((torch.tensor([False], device=self.device), is_auto_gen_end_tok[:-1]))
        
        SP_toks = is_auto_gen_start_tok + is_auto_gen_end_tok
        SP_mask = torch.cumsum(SP_toks.to(torch.int), dim=0) % 2 == 1
        return SP_mask
    
    def batchGameEnded(self, states, target_strings):
        '''
        states: Long tensor: [N, seq_len]
        target_states: list of target string states: [N]
        reutrns: Long tensor: [N]
        '''
        N = states.shape[0]
        SP_mask = self.scratchPadMask(states) # scratchPad tokens are True
        pad_mask = states==self.pad_id # pad tokens are True
        true_token_mask = ~(SP_mask | pad_mask) # scratchPad and pad tokens are False
        
        game_states = ((states!=self.pad_id).sum(dim=1)<self.max_token_length).to(torch.float)-1
        for i in torch.arange(N)[game_states==0]:
#             print(states[i], SP_mask[i], pad_mask[i])
            if self.tokenizer.decode(states[i][true_token_mask[i]].tolist())==target_strings[i]:
                game_states[i] = 1
            
        return game_states
    
    def getGameEndedTransform(self, canonical_state_batch):
        """
        Input:
            canonical_state_batch:[{'curernt_state':torch.tensor([1,2]), 'target_state':torch.tensor([1,2,3])}, {'curernt_state':, 'target_state':}]
        Returns: [{'curernt_state':torch.tensor([1,2]), 'target_state':torch.tensor([1,2,3]), 'game_ended':tensor(0)}, ]
            0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
        """
        for canonical_state in canonical_state_batch:
            canonical_state['game_ended'] = torch.tensor(self.getGameEnded(canonical_state))
        return canonical_state_batch

    def getGameEnded(self, canonical_state):
        """
        Input:
            canonical_state: {'curernt_state':torch.tensor([1,2]), 'target_state':torch.tensor([1,2,3])}
        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
        """
        current_state = canonical_state['current_state']
        state_len = current_state.shape[0]
        if state_len > self.max_token_length:
            return -1
        
        sp_mask = self.scratchPadMask(current_state)
        
        string_current_state = self.tokenizer.decode(current_state[~sp_mask].tolist())
        string_target_state = self.tokenizer.decode(canonical_state['target_state'].tolist())
        if string_current_state == string_target_state:
            return 1
        
        if len(current_state) == 0:
            return 0
        if current_state[-1] == self.eos_id:
            return -1
        return 0
    
    def prepStateForModelTransform(self, canonical_state_batch):
        for canonical_state in canonical_state_batch:
            canonical_state['input_ids'] = self.prepStateForModel(canonical_state)
        return canonical_state_batch
    
    def prepStateForModel(self, canonical_state):
        current_state = canonical_state['current_state']
        extra_tokens = torch.tensor([self.mask_id, self.value_id], device=self.device)
        input_state = torch.cat((current_state, extra_tokens)).unsqueeze(0)
        return input_state
    
    def batchStringRep(self, states):
        '''
        states: Long tensor: [N, seq_len]
        returns: list of string representations for the states [str]
        '''
        return [self.tokenizer.decode(state[state!=self.pad_id].tolist(), skip_special_tokens=False) for state in states]
    
    def stringRepresentationTransform(self, canonical_state_batch):
        """
        Input:
            canonical_state_batch: current state: [{'curernt_state':'foo bar', 'target_state':'foo bar baz'}]
        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        for canonical_state in canonical_state_batch:
            canonical_state['string_rep'] = self.stringRepresentation(canonical_state)
        return canonical_state_batch

    def stringRepresentation(self, canonical_state):
        """
        Input:
            canonical_state: current state: {'curernt_state':'foo bar', 'target_state':'foo bar baz'}
        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return self.tokenizer.decode(canonical_state['current_state'], skip_special_tokens=False)
    
    

class EpisodeUtility():
    def __init__(self, game, num_episodes=10, num_sims_per_turn=20, policy_exploration_weight=0.5, **kwargs):
        '''
        Wrapper class for experience gathering.
        '''
        self.game = game
        self.num_episodes = num_episodes
        self.num_sims_per_turn = num_sims_per_turn
        self.policy_exploration_weight = policy_exploration_weight
        self.episode_counter = 0
        
    def explore(self, model):
        examples = []
        t1 = perf_counter()  
        for _ in tqdm.tqdm(range(self.num_episodes), desc='Exploring env'):
            examples += self.executeEpisode(model)
        t2 = perf_counter()  
        positive_examples = [ex for ex in examples if ex['reward'] == 1]
        print(f'Performed {self.num_episodes} episodes in {t2-t1:.2f}s. {len(positive_examples)} positive\n------------------')
        if positive_examples != []:
            for i in range(2):
                ex = random.choice(positive_examples)
                self.prettyPrint(ex)
        for i in range(2):
            ex = random.choice(examples)
            self.prettyPrint(ex)
        return examples
    
    def prettyPrint(self, ex):
        canon_state, pi, r = ex['canonical_state'], ex['pi'], ex['reward']
        print(self.game.tokenizer.decode(canon_state['target_state']))
        print(self.game.tokenizer.decode(canon_state['current_state']))
        top_actions = torch.argsort(pi, descending=True)
        print(f'Top 3 next actions: {[self.game.tokenizer.decode([a]) for a in top_actions[:3]]}')
        print(f'Current reward: {r}')
        print()
        
    def assignRewards(self, examples, reward):
        for ex in examples:
            ex['reward'] = reward
        return examples
        
    def executeEpisode(self, model):
        examples = []
        canonical_state = self.game.randInitProblem()
        mcts = FastMCTS(self.game, model, cpuct=self.policy_exploration_weight)     # initialise search tree
        while True:
            for _ in range(self.num_sims_per_turn):
                mcts.search(canonical_state)
            pi_s = mcts.getActionProb(canonical_state)
            examples.append({'canonical_state':canonical_state, 
                             'pi':pi_s, 
                             'id': self.episode_counter,
                             'reward':None})              # rewards can not be determined yet 
            self.episode_counter += 1
            a = torch.multinomial(pi_s,1)[0]    # sample action from improved policy
            canonical_state = self.game.getNextState(canonical_state,a)
            reward = self.game.getGameEnded(canonical_state)
            if reward in [1,-1]:
                examples = self.assignRewards(examples, reward) 
                return examples
            
    def decomposePositiveEpisode(self, canonical_state):
        examples = []
        current_state = canonical_state['current_state']
        sp_mask = self.game.autoGeneratedMask(current_state)
        for i in range(current_state.shape[0]):
            if sp_mask[i] == True:
                continue
            one_hot_pi = torch.zeros(self.game.getActionSize(), dtype=torch.float)
            one_hot_pi[current_state[i]] = 1
            new_example = {'canonical_state': {
                               'current_state':current_state[:i], 
                               'target_state':canonical_state['target_state']
                           },
                           'pi': one_hot_pi,
                           'id': 'custom',
                           'reward':1
                          }
            examples.append(new_example)
        return examples
    
    def prepExamplesForModel_Transform(self, samples):
        for sample_obj in samples:
            sample_obj['input_ids'] = self.game.prepStateForModel(sample_obj['canonical_state'])
        return samples
    
    def fullLoopTrainer(self, model, steps=10, custom_examples=[], max_epochs=500):
        explored_examples = []
        for step in range(steps):
            # Training phase
            train_samples = copy.deepcopy(explored_examples + custom_examples)
            train_samples = self.prepExamplesForModel_Transform(train_samples)
            train_samples = Class_Rebalance_Transform(field='target_value')(train_samples)
            train_dataset = Scratch_Pad_Policy_Dataset(train_samples, slow_pipe=[], real_time_pipe=[])
            train_dataloader = train_dataset.to_dataloader(3, shuffle=True)
            
            trainer = Trainer(gradient_clip_val=0.5, amp_level='O1', max_epochs=max_epochs)
            model.train()
            trainer.fit(model, train_dataloader)
            
            # Exploration phase
            model.eval()
            new_explored_examples = self.explore(model)
            successful_explorations = [ex for ex in new_explored_examples if ex['reward'] == 1]
            print(f'{len(successful_explorations)}/{len(new_explored_examples)} successful explorations:')
            for ex in successful_explorations[:5]:
                self.prettyPrint(ex)
            print("Failed explorations:")
            for ex in new_explored_examples[:5]:
                self.prettyPrint(ex)
            new_explored_examples = Rename_Transform(fields=[('pi','target_policy'),('reward','target_value')])(new_explored_examples)
                
            new_exp_size = int(len(explored_examples)/2)
            print(f"Trimming {len(explored_examples) - new_exp_size} examples from the exploration dataset")
            explored_examples = random.choices(explored_examples, k=new_exp_size)
            explored_examples += new_explored_examples
            