import sys
import os

assert len(sys.argv) == 3, 'Args are wrong.'

input_path = sys.argv[1]
output_path = sys.argv[2]

assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
from share import *
from cldm.model import create_model


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


model = create_model(config_path='./models/cldm_v15.yaml')

pretrained_weights = torch.load(input_path, weights_only=False)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    if is_control:
        copy_k = 'model.diffusion_' + name
    else:
        copy_k = k
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

# Our model adds additional channels to the first layer to condition on an input image.
# For the first layer, copy existing channel weights and initialize new channel weights to zero.
# input_keys = [
#     "model.diffusion_model.input_blocks.0.0.weight",
#     "control_model.input_blocks.0.0.weight",
#     "model_ema.diffusion_modelinput_blocks00weight",
# ]
#
# for input_key in input_keys:
#     if input_key not in scratch_dict:
#         continue
#     is_control, name = get_node_name(input_key, 'control_')
#     if is_control:
#         copy_k = 'model.diffusion_' + name
#     else:
#         copy_k = input_key
#
#     input_weight = scratch_dict[input_key]
#
#     if input_weight.size() != pretrained_weights[copy_k].size():
#         print(f"Manual init: {input_key}")
#         input_weight.zero_()
#         input_weight[:, :4, :, :].copy_(pretrained_weights[copy_k])
#
#     target_dict[input_key] = input_weight


model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')
