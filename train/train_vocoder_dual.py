# Copyright (c) 2025 Xiaobin-Rong.
# Adapted under MIT LICENSE.
# Source: https://github.com/Xiaobin-Rong/SEtrain
# License included under licenses/LICENSE_setrain.

import os
import torch
import shutil
import random
import argparse
import numpy as np
import torch.distributed as dist
from datetime import datetime
from omegaconf import OmegaConf
from tqdm import tqdm
from glob import glob
from pathlib import Path
import soundfile as sf
from torch.utils.tensorboard import SummaryWriter
from utils.distributed_utils import reduce_value

from .dataloader import URGENT2Dataset as Dataset
from models.wavlm.feature_extractor import WavLM_feat as Encoder
from models.vocoder.wavlmdec_dual import WavLMDec as Generator
from models.vocoder.discriminators import (
    MultiPeriodDiscriminator,
    MultiBandDiscriminator,
    CombinedDiscriminator
)
from .loss import (
    feature_loss,
    generator_loss,
    discriminator_loss,
    MultiScaleMelSpectrogramLoss,
)
from utils.scheduler import LinearWarmupCosineAnnealingLR as WarmupLR


seed = 43
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
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)
        dist.barrier()

    args.rank = rank
    args.device = torch.device(f"cuda:{rank}")
    
    collate_fn = Dataset.collate_fn if hasattr(Dataset, "collate_fn") else None
    shuffle = False if args.world_size > 1 else True

    train_dataset = Dataset(**config['train_dataset'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.world_size > 1 else None
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    sampler=train_sampler,
                                                    **config['train_dataloader'],
                                                    shuffle=shuffle,
                                                    collate_fn=collate_fn)
    
    validation_dataset = Dataset(**config['validation_dataset'])
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset) if args.world_size > 1 else None
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                        sampler=validation_sampler,
                                                        **config['validation_dataloader'], 
                                                        shuffle=False,
                                                        collate_fn=collate_fn)
    
    encoder = Encoder(**config['encoder_config']).to(args.device)
    generator = Generator(**config['decoder_config']).to(args.device)
    
    mpd = MultiPeriodDiscriminator(**config['discriminator_config']['mpd']).to(args.device)
    mbd = MultiBandDiscriminator(**config['discriminator_config']['mbd']).to(args.device)
    discriminator = CombinedDiscriminator([mpd, mbd]).to(args.device)
        
    if args.world_size > 1:
        generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[rank])
        discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[rank])

    optimizer_g = torch.optim.AdamW(params=generator.parameters(), lr=config['optimizer']['lr'], betas=(0.8, 0.99), weight_decay=0.01)
    optimizer_d = torch.optim.AdamW(params=discriminator.parameters(), lr=config['optimizer']['lr'], betas=(0.8, 0.99), weight_decay=0.01)
    
    scheduler_g = WarmupLR(optimizer_g, **config['scheduler'])
    scheduler_d = WarmupLR(optimizer_d, **config['scheduler'])

    loss = MultiScaleMelSpectrogramLoss(sampling_rate=config['samplerate']).to(args.device)

    trainer = Trainer(config=config, model=[encoder, generator, discriminator],
                      optimizer=[optimizer_g, optimizer_d], scheduler=[scheduler_g, scheduler_d],
                      loss_func=loss, train_dataloader=train_dataloader, validation_dataloader=validation_dataloader, 
                      train_sampler=train_sampler, args=args)

    trainer.train()

    if args.world_size > 1:
        dist.destroy_process_group()


class Trainer:
    def __init__(self, config, model, optimizer, scheduler, loss_func,
                 train_dataloader, validation_dataloader, train_sampler, args):
        self.config = config
        self.encoder = model[0]
        self.generator = model[1]
        self.discriminator = model[2]
        self.optimizer_g = optimizer[0]
        self.optimizer_d = optimizer[1]
        self.scheduler_g = scheduler[0]
        self.scheduler_d = scheduler[1]
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
        
        self.lamda = config['coeff']['recons']
        self.lamda_feat = config['coeff']['feat']
        self.lamda_adv = config['coeff']['adv']
        
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
            shutil.copytree(Path(__file__).parent.parent / 'models', Path(self.code_path) / 'models', dirs_exist_ok=True)
            self.writer = SummaryWriter(self.log_path)

        self.start_epoch = 1
        self.best_score = 1e8

        if self.resume:
            self._resume_checkpoint()


    def _set_train_mode(self):
        self.generator.train()
        self.discriminator.train()

    def _set_eval_mode(self):
        self.generator.eval()
        self.discriminator.eval()

    def _save_checkpoint(self, epoch, score):
        generator_dict = self.generator.module.state_dict() if self.world_size > 1 else self.generator.state_dict()
        discriminator_dict = self.discriminator.module.state_dict() if self.world_size > 1 else self.discriminator.state_dict()

        state_dict = {'epoch': epoch,
                      'optimizer_g': self.optimizer_g.state_dict(),
                      'optimizer_d': self.optimizer_d.state_dict(),
                      'scheduler_g': self.scheduler_g.state_dict(),
                      'scheduler_d': self.scheduler_d.state_dict(),
                      'generator': generator_dict,
                      'discriminator': discriminator_dict}

        torch.save(state_dict, os.path.join(self.checkpoint_path, f'model_{str(epoch).zfill(3)}.tar'))

        if score < self.best_score:
            self.state_dict_best = state_dict.copy()
            self.best_score = score

    def _del_checkpoint(self, epoch, score):
        # Condition 1: epoch-1 is not a multiple of self.save_checkpoint_interval;
        # Condition 2: current score is best score.
        # if (epoch - 1) % self.save_checkpoint_interval != 0 or score == self.best_score:
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
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d'])
        if self.world_size > 1:
            self.generator.module.load_state_dict(checkpoint['generator'])
            self.discriminator.module.load_state_dict(checkpoint['discriminator'])
            
        else:
            self.generator.load_state_dict(checkpoint['generator'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])


    def _train_epoch(self, epoch):
        total_loss = 0
        total_loss_adv = 0
        total_loss_feat = 0
        total_loss_dis = 0
        self.train_dataloader.dataset.sample_data_per_epoch()
        self.train_bar = tqdm(self.train_dataloader, ncols=130)

        for step, (noisy, clean, info) in enumerate(self.train_bar, 1):
            fs = self.default_fs
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)     # (B, 1, T) 

            # For generator
            with torch.no_grad():
                feat_a_noisy, feat_p_noisy = self.encoder(noisy)  # (B, T, D)
            
            esti_wav = self.generator(feat_p_noisy, feat_a_noisy)
            
            loss_mel = self.lamda * self.loss_func(esti_wav, clean)

            true_metric, esti_metric, true_fmap, esti_fmap = self.discriminator(clean, esti_wav)
            
            loss_adv = self.lamda_adv * generator_loss(esti_metric)[0]
            loss_feat = self.lamda_feat * feature_loss(true_fmap, esti_fmap)
            
            loss = loss_mel + loss_adv + loss_feat
            
            if self.world_size > 1:
                loss = reduce_value(loss)
                loss_adv = reduce_value(loss_adv)
                loss_feat = reduce_value(loss_feat)
            total_loss += loss.item()
            total_loss_adv += loss_adv.item()
            total_loss_feat += loss_feat.item()
            
            self.optimizer_g.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.clip_grad_norm_value)
            self.optimizer_g.step()

            # For discriminator
            true_metric, esti_metric, _, _ = self.discriminator(clean, esti_wav.detach())

            loss_dis = discriminator_loss(true_metric, esti_metric)[0]
            if self.world_size > 1:
                loss_dis = reduce_value(loss_dis)
            total_loss_dis += loss_dis.item()
                
            self.optimizer_d.zero_grad()
            loss_dis.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clip_grad_norm_value)
            self.optimizer_d.step()
            
            self.scheduler_g.step()
            self.scheduler_d.step()
            
            self.train_bar.desc = '   train[{}/{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.train_bar.postfix = 'L={:.2f}, Lg={:.2f}, Lf={:.2f}, Ld={:.2f}'.format(total_loss / step,
                                                                                        total_loss_adv / step,
                                                                                        total_loss_feat / step,
                                                                                        total_loss_dis / step)
        if self.world_size > 1 and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars('lr', {'lr': self.optimizer_g.param_groups[0]['lr']}, epoch)
            self.writer.add_scalars('train_loss', {'loss': total_loss / step,
                                                   'loss_Adv': total_loss_adv / step,
                                                   'loss_Feat': total_loss_feat / step, 
                                                   'loss_Dis': total_loss_dis / step}, epoch)


    @torch.inference_mode()
    def _validation_epoch(self, epoch):
        total_loss = 0
        total_loss_adv = 0
        total_loss_feat = 0
        total_loss_dis = 0
        self.validation_bar = tqdm(self.validation_dataloader, ncols=130)
        for step, (noisy, clean, info) in enumerate(self.validation_bar, 1):
            fs = self.default_fs
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)     # (B, 1, T) 

            # For generator
            feat_a_noisy, feat_p_noisy = self.encoder(noisy)
            
            esti_wav = self.generator(feat_p_noisy, feat_a_noisy)
            
            loss_mel = self.lamda * self.loss_func(esti_wav, clean)
            
            true_metric, esti_metric, true_fmap, esti_fmap = self.discriminator(clean, esti_wav)
            
            loss_adv = self.lamda_adv * generator_loss(esti_metric)[0]
            loss_feat = self.lamda_feat * feature_loss(true_fmap, esti_fmap)
            
            loss = loss_mel + loss_adv + loss_feat
            
            if self.world_size > 1:
                loss = reduce_value(loss)
                loss_adv = reduce_value(loss_adv)
                loss_feat = reduce_value(loss_feat)
            total_loss += loss.item()
            total_loss_adv += loss_adv.item()
            total_loss_feat += loss_feat.item()

            # For discriminator
            true_metric, esti_metric, _, _ = self.discriminator(clean, esti_wav.detach())

            loss_dis = discriminator_loss(true_metric, esti_metric)[0]
            if self.world_size > 1:
                loss_dis = reduce_value(loss_dis)
            total_loss_dis += loss_dis.item()

            if self.rank == 0 and (epoch<10 or epoch%10 == 0) and step <= 5:
                uid = info['id'][0]
                noisy_path = os.path.join(self.sample_path, '{}_noisy.wav'.format(uid))
                clean_path = os.path.join(self.sample_path, '{}_clean.wav'.format(uid))
                enhanced_path = os.path.join(self.sample_path, '{}_enh_epoch{}.wav'.format(uid, str(epoch).zfill(3)))

                noisy = noisy.cpu().numpy().squeeze()
                clean = clean.cpu().numpy().squeeze()
                esti_wav = esti_wav.detach().cpu().numpy().squeeze()
                
                sf.write(noisy_path, noisy, samplerate=self.default_fs)
                sf.write(clean_path, clean, samplerate=self.default_fs)
                sf.write(enhanced_path, esti_wav, samplerate=self.default_fs)
                
            self.validation_bar.desc = 'validate[{}/{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.validation_bar.postfix = 'L={:.2f}, Lg={:.2f}, Lf={:.2f}, Ld={:.2f}'.format(total_loss / step,
                                                                                        total_loss_adv / step,
                                                                                        total_loss_feat / step,
                                                                                        total_loss_dis / step)

        if (self.world_size > 1) and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars('val_loss', {'loss': total_loss / step,
                                                   'loss_Adv': total_loss_adv / step,
                                                   'loss_Feat': total_loss_feat / step, 
                                                   'loss_Dis': total_loss_dis / step}, epoch)

        return total_loss / step


    def train(self):
        if self.resume:
            self._resume_checkpoint()

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
    parser.add_argument('-C', '--config', default='configs/cfg_train_vocoder_dual.yaml')
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
