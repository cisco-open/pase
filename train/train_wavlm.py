# Copyright (c) 2025 Xiaobin-Rong.
# Adapted under MIT LICENSE.
# Source: https://github.com/Xiaobin-Rong/SEtrain
# License included under licenses/LICENSE_setrain.

"""
Fine-tuning WavLM with knowledge distillation (KD) objective
"""
import os
import torch
import shutil
import random
import argparse
import numpy as np
import torch.distributed as dist
import soundfile as sf
from datetime import datetime
from tqdm import tqdm
from glob import glob
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from utils.distributed_utils import reduce_value

from loaders.dataloader import URGENT2Dataset as WavLMDataset
from models.wavlm.feature_extractor import WavLM_feat
from models.vocoder.wavlmdec import WavLMDec
from utils.scheduler import LinearWarmupCosineAnnealingLR as WarmupLR


seed = 34
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic =True


def run(rank, config, args):
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12357'
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)
        dist.barrier()

    args.rank = rank
    args.device = torch.device(f"cuda:{rank}")
    
    collate_fn = WavLMDataset.collate_fn if hasattr(WavLMDataset, "collate_fn") else None
    shuffle = False if args.world_size > 1 else True

    train_dataset = WavLMDataset(**config['train_dataset'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.world_size > 1 else None
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    sampler=train_sampler,
                                                    **config['train_dataloader'],
                                                    shuffle=shuffle,
                                                    collate_fn=collate_fn)
    
    validation_dataset = WavLMDataset(**config['validation_dataset'])
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset) if args.world_size > 1 else None
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                        sampler=validation_sampler,
                                                        **config['validation_dataloader'], 
                                                        shuffle=False,
                                                        collate_fn=collate_fn)
    
    vocoder = WavLMDec.from_pretrained(**config['vocoder_config']).to(args.device)

    teacher = WavLM_feat(**config['teacher_config']).to(args.device)
    model = WavLM_feat(**config['student_config']).to(args.device)

    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['optimizer']['lr'], betas=(0.8, 0.99), weight_decay=0.01)    
    scheduler = WarmupLR(optimizer, **config['scheduler'])
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **config['scheduler'])

    loss = torch.nn.MSELoss().to(args.device)

    trainer = Trainer(config=config, model=[model, teacher, vocoder],
                      optimizer=optimizer, scheduler=scheduler, loss_func=loss,
                      train_dataloader=train_dataloader, validation_dataloader=validation_dataloader, 
                      train_sampler=train_sampler, args=args)

    trainer.train()

    if args.world_size > 1:
        dist.destroy_process_group()


class Trainer:
    def __init__(self, config, model, optimizer, scheduler, loss_func,
                 train_dataloader, validation_dataloader, train_sampler, args):
        self.config = config
        self.model = model[0]
        self.teacher = model[1]
        self.vocoder = model[2]
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        self.train_sampler = train_sampler
        self.rank = args.rank
        self.device = args.device
        self.world_size = args.world_size
        
        self.default_fs = config['samplerate']

        # training config
        config['DDP']['world_size'] = args.world_size
        self.trainer_config = config['trainer']
        self.epochs = self.trainer_config['epochs']
        self.save_checkpoint_interval = self.trainer_config['save_checkpoint_interval']
        self.clip_grad_norm_value = self.trainer_config['clip_grad_norm_value']
        self.resume = self.trainer_config['resume']

        if not self.resume:
            self.exp_path = self.trainer_config['exp_path'] + '_' + datetime.now().strftime("%Y-%m-%d-%Hh%Mm")
 
        else:
            self.exp_path = self.trainer_config['exp_path'] + '_' + self.trainer_config['resume_datetime']

        self.log_path = os.path.join(self.exp_path, 'logs')
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoints')
        self.sample_path = os.path.join(self.exp_path, 'val_samples')
        self.code_path = os.path.join(self.exp_path, 'codes')

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)
        os.makedirs(self.code_path, exist_ok=True)

        # save the config
        if self.rank == 0:
            shutil.copy2(__file__, self.exp_path)
            shutil.copy2(args.config, Path(self.exp_path) / 'config.yaml')
            
            for file in Path(__file__).parent.parent.iterdir():
                if file.is_file():
                    shutil.copy2(file, self.code_path)
            for d in ['configs', 'loaders', 'models', 'train', 'inference', 'utils']:
                shutil.copytree(Path(__file__).parent.parent / d, Path(self.code_path) / d, dirs_exist_ok=True)
            self.writer = SummaryWriter(self.log_path)

        self.start_epoch = 1
        self.best_score = 1e8

        if self.resume:
            self._resume_checkpoint()

    def _set_train_mode(self):
        self.model.train()

    def _set_eval_mode(self):
        self.model.eval()

    def _save_checkpoint(self, epoch, score):
        model_dict = self.model.module.wavlm.state_dict() if self.world_size > 1 else self.model.wavlm.state_dict()
        state_dict = {'epoch': epoch,
                      'optimizer': self.optimizer.state_dict(),
                      'scheduler': self.scheduler.state_dict(),
                      'model': model_dict}

        torch.save(state_dict, os.path.join(self.checkpoint_path, f'model_{str(epoch).zfill(3)}.tar'))
        if score < self.best_score:
            self.state_dict_best = state_dict.copy()
            self.best_score = score

    def _del_checkpoint(self, epoch, score):
        if (epoch - 1) % self.save_checkpoint_interval != 0:
            prev_epoch = epoch - 1
            checkpoint_file = os.path.join(self.checkpoint_path, f'model_{str(prev_epoch).zfill(3)}.tar')
        
            if os.path.exists(checkpoint_file):
                try:
                    os.remove(checkpoint_file)
                    print(f"Deleted checkpoint: {checkpoint_file}")
                except Exception as e:
                    print(f"Failed to delete checkpoint {checkpoint_file}: {e}")

    def _resume_checkpoint(self):
        latest_checkpoints = sorted(glob(os.path.join(self.checkpoint_path, 'model_*.tar')))[-1]

        map_location = self.device
        checkpoint = torch.load(latest_checkpoints, map_location=map_location)

        self.start_epoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        if self.rank == 0:
            print(self.scheduler.state_dict())
        if self.world_size > 1:
            self.model.module.wavlm.load_state_dict(checkpoint['model'])
        else:
            self.model.wavlm.load_state_dict(checkpoint['model'])


    def _train_epoch(self, epoch):
        total_loss = 0
        self.train_dataloader.dataset.sample_data_per_epoch()
        self.train_bar = tqdm(self.train_dataloader, ncols=120)

        for step, (noisy, clean, info) in enumerate(self.train_bar, 1):
            noisy = noisy.to(self.device)     # (B, L)
            clean = clean.to(self.device)

            noisy_feat = self.model(noisy)

            with torch.no_grad():
                clean_feat = self.teacher(clean)

            loss = self.loss_func(noisy_feat, clean_feat)
            
            if self.world_size > 1:
                loss = reduce_value(loss)
                
            total_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.optimizer.step()
            
            self.scheduler.step()
            
            self.train_bar.desc = '   train[{}/{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.train_bar.postfix = 'train_loss={:.4f}'.format(total_loss / step)
            
        if self.world_size > 1 and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch)
            self.writer.add_scalars('train_loss', {'loss': total_loss / step}, epoch)


    @torch.no_grad()
    def _validation_epoch(self, epoch):
        total_loss = 0
        self.validation_bar = tqdm(self.validation_dataloader, ncols=120)
        for step, (noisy, clean, info) in enumerate(self.validation_bar, 1):
            noisy = noisy.to(self.device)     # (B, L)
            clean = clean.to(self.device)
            
            noisy_feat = self.model(noisy)
            clean_feat = self.teacher(clean)

            loss = self.loss_func(noisy_feat, clean_feat)
            
            if self.world_size > 1:
                loss = reduce_value(loss)
                
            total_loss += loss.item()

            if self.rank == 0 and (epoch<10 or epoch%10 == 0) and step <= 10:
                clean_feat = self.model(clean)

                noisy_recon = self.vocoder(noisy_feat)
                clean_recon = self.vocoder(clean_feat)
            
                uid = info['id'][0]
                noisy_gt_path = os.path.join(self.sample_path, '{}_noisy_gt.wav'.format(uid))
                clean_gt_path = os.path.join(self.sample_path, '{}_clean_gt.wav'.format(uid))
                noisy_recon_path = os.path.join(self.sample_path, '{}_noisy_recon_epoch{}.wav'.format(uid, str(epoch).zfill(3)))
                clean_recon_path = os.path.join(self.sample_path, '{}_clean_recon_epoch{}.wav'.format(uid, str(epoch).zfill(3)))

                noisy_gt = noisy.cpu().squeeze().numpy()
                clean_gt = clean.cpu().squeeze().numpy()
                noisy_recon = noisy_recon.cpu().squeeze().numpy()
                clean_recon = clean_recon.cpu().squeeze().numpy()
                
                sf.write(noisy_gt_path, noisy_gt, self.default_fs)
                sf.write(clean_gt_path, clean_gt, self.default_fs)
                sf.write(noisy_recon_path, noisy_recon, self.default_fs)
                sf.write(clean_recon_path, clean_recon, self.default_fs)

            self.validation_bar.desc = 'validate[{}/{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.validation_bar.postfix = 'valid_loss={:.4f}'.format(total_loss / step)

        if (self.world_size > 1) and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars('val_loss', {'loss': total_loss / step}, epoch)

        return total_loss / step


    def train(self):
        if self.resume:
            self._resume_checkpoint()

        if self.start_epoch == 1:
            self._set_eval_mode()
            _ = self._validation_epoch(0)

        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self._set_train_mode()
            self._train_epoch(epoch)

            self._set_eval_mode()
            valid_loss = self._validation_epoch(epoch)
            torch.cuda.empty_cache()

            if self.rank == 0:
                self._save_checkpoint(epoch, valid_loss)
                self._del_checkpoint(epoch, valid_loss)

        if self.rank == 0:
            torch.save(self.state_dict_best,
                    os.path.join(self.checkpoint_path,
                    'best_model_{}.tar'.format(str(self.state_dict_best['epoch']).zfill(3))))

            print('------------Training for {} epochs has done!------------'.format(self.epochs))



if __name__ == '__main__':
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='configs/cfg_train_wavlm.yaml')
    parser.add_argument('-D', '--device', default='0', help='The index of the available devices, e.g. 0,1,2,3')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.world_size = len(args.device.split(','))
    config = OmegaConf.load(args.config)
    
    if args.world_size > 1:
        torch.multiprocessing.spawn(
            run, args=(config, args,), nprocs=args.world_size, join=True)
    else:
        run(0, config, args)
