import json
import torch

def BERT_self_attention_combiner(bert_state_dict, target_state_dict, layer, self_att_map):
#     with open("bert_pretrain_convert/self_attention_map.json", "r") as f:
#         key_map = json.load(f)
    key_map = self_att_map
    def insert_layer(my_list, layer):
        return [el.replace("0", str(layer)) for el in my_list]

    weight_source_keys = insert_layer(key_map['source_keys'][:3], layer)
    bias_source_keys = insert_layer(key_map['source_keys'][3:], layer)
    target_keys = insert_layer(key_map['target_keys'], layer)
    
    weight_source = [bert_state_dict[key] for key in weight_source_keys]
    bias_source = [bert_state_dict[key] for key in bias_source_keys]
    
    weight_source = torch.cat(weight_source, dim=0)
    bias_source = torch.cat(bias_source, dim=0)
    
    # WEIGHTS
    target_state_dict[target_keys[0]] = weight_source
    
    # BIAS
    target_state_dict[target_keys[1]] = bias_source
    

def translate_one_layer(bert_state_dict, target_state_dict, layer, layer_map, self_att_map):
#     with open("bert_pretrain_convert/layer_map.json", "r") as f:
#         key_map = json.load(f)
    key_map = layer_map
    layer_map = {}
    for key, value in key_map.items():
        layer_map[key.replace("0", str(layer))] = value.replace("0", str(layer))
        
    BERT_self_attention_combiner(bert_state_dict, target_state_dict, layer, self_att_map)
    
    for key, value in layer_map.items():
        target_state_dict[value] = bert_state_dict[key]
    
    return target_state_dict

def translate_from_hugginface_to_torch_BERT(bert_state_dict, extras_map, layer_map, self_att_map, num_layers = 12):
    
    target = {}
    
    for id_layer in range(num_layers):
        translate_one_layer(bert_state_dict, target, id_layer, layer_map, self_att_map)
        
    #Missing Embedder and Pooler
#     with open("bert_pretrain_convert/extras_map.json", "r") as f:
#         key_map = json.load(f)
    key_map = extras_map
    
    for key, value in key_map.items():
        if value is None:
            continue
            
        target[value] = bert_state_dict[key]
        
    return target