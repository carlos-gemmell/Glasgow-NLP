import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
from heapq import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import itertools

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 50


class BeamSearchNode(object):
    def __init__(self, decoder_output, encoder_input, prev_node, last_id, logProb, length,batch_num):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.decoder_output = decoder_output
        self.encoder_input = encoder_input
        self.prev_node = prev_node
        self.last_id = last_id
        self.logp = logProb
        self.leng = length
        self.batch_num = batch_num
    
    def __lt__(self, other):
        return True

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward



def beam_search_decode(model, batch_encoder_ids, beam_size=3, num_out=3, max_length=10, SOS_token=1,EOS_token=2, PAD_token=3):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoded_batch = []
    batch_size = batch_encoder_ids.shape[1]
    batch_endnodes = [[] for i in range(batch_size)]
    batch_trees = [[] for i in range(batch_size)]
    
    beam_sizes = [beam_size] * batch_size
    
    first_decoder_id = torch.LongTensor([[SOS_token]]).to(device)
    
    for i, decode_tree in enumerate(batch_trees):
        root_node = BeamSearchNode(first_decoder_id, batch_encoder_ids[:,i].view(-1,1), None, SOS_token, 0, 1,i)
        heappush(decode_tree, (-root_node.eval(), root_node))
    
    while True:
        # print("STEP")
        working_nodes = []
        for tree, sample_endnode_list, sample_beam_size in zip(batch_trees, batch_endnodes, beam_sizes):
            for i in range(sample_beam_size):
                if len(tree) == 0:
                    break
                score, node = heappop(tree)
                if node.last_id == EOS_token or node.leng >= max_length:
                    if len(sample_endnode_list) < num_out:
                        # print("FOUND,",node.logp.item())
                        sample_endnode_list.append(node)
                    sample_beam_size -= 1
                else:
                    working_nodes.append((score, node))
            
            del tree[:]
                
                    
        if working_nodes == []:
            break
        
        # making the padded input from different batches and sequence lengths
        rough_input = [n.decoder_output for (score, n) in working_nodes]
        max_decoder_size = max([t.shape[0] for t in rough_input])
        padded_decoder_input = torch.zeros((max_decoder_size,len(working_nodes)), dtype=torch.long, device=device).fill_(PAD_token)
        
        # we create a correctly sized tensor with all <pad> symbols and fill with the according tokens
        for i in range(len(working_nodes)):
            length = rough_input[i].shape[0]
            padded_decoder_input[:length,i] = rough_input[i].view(-1)
        
        encoder_input = torch.cat([n.encoder_input for (score,n) in working_nodes], dim=1)
        
        decoder_predictions = model(encoder_input, padded_decoder_input)
#         print(ex["p_gens"])
        
        for (score, node), logits in zip(working_nodes, decoder_predictions.transpose(0,1)):
            last_token_pos = node.decoder_output.shape[0] - 1
            last_token_logits = logits[last_token_pos] 
            last_token_log_probs = last_token_logits.log_softmax(0)
            log_probs, indexes = torch.topk(last_token_log_probs, beam_sizes[node.batch_num])
            for log_prob, idx in zip(log_probs, indexes):
                new_decoder_output = torch.cat([node.decoder_output, idx.view(-1,1)])
#                 print([TGT_TEXT.vocab.itos[f] for f in new_decoder_output.view(-1)], -(node.logp+log_prob).item())
                new_node = BeamSearchNode(new_decoder_output, node.encoder_input, node, idx, node.logp+log_prob, node.leng+1,node.batch_num)
                heappush(batch_trees[node.batch_num], (-(node.logp+log_prob),new_node))
            batch_trees[node.batch_num] = nsmallest(beam_sizes[node.batch_num],batch_trees[node.batch_num])
        
       
    for i in range(len(batch_endnodes)):
        batch_endnodes[i] = [n.decoder_output for n in sorted(batch_endnodes[i], key=lambda node: -node.logp)]
    return batch_endnodes
                