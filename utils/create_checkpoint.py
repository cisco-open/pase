import torch
from omegaconf import OmegaConf


train_cfg_path = "/work/user_data/xiaobin/Experiments/exp_PASE/exp_vocoder_L24_2025-12-04-11h43m/config.yaml"
train_ckpt_path = "/work/user_data/xiaobin/Experiments/exp_PASE/exp_vocoder_L24_2025-12-04-11h43m/checkpoints/model_001.tar"
save_ckpt_path = "/work/user_data/xiaobin/Pre-trained/Vocoder_L24_ep001.tar"

config = OmegaConf.load(train_cfg_path)
config =  OmegaConf.to_container(config)
state_dict = torch.load(train_ckpt_path, map_location='cpu')

new_dict = {}

if 'generator' in state_dict.keys():
    model_dict = state_dict['generator']
elif 'model' in state_dict.keys():
    model_dict = state_dict['model']
else:
    raise ValueError("Keys mismatch!")

new_dict['model'] = model_dict
new_dict['cfg'] = config['decoder_config']

torch.save(new_dict, save_ckpt_path)
print("Successfully created a pre-trained checkpoint:", save_ckpt_path)
print("Keys:", new_dict.keys())


