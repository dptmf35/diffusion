import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        self.cond_ddpm = self.set_device(networks.define_DDPM(opt))
        self.schedule_phase = None

        # set loss and load resume state
        self.set_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
        self.set_loss()


        if self.opt['phase'] == 'train':
            self.cond_ddpm.train() 
            optim_params = list(self.cond_ddpm.parameters())
            self.optDDPM = torch.optim.Adam(optim_params,lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()

        self.load_network()
        self.print_network()
        

    def feed_data(self, data):
        self.data = self.set_device(data) 
        self.gt = self.data['GT']
        self.condition = self.data['condition']

    def optimize_parameters(self):
        b, c, h, w = self.gt.shape
        
        self.optDDPM.zero_grad()
        ddpm_loss = self.cond_ddpm(self.gt, self.condition)
        ddpm_loss = ddpm_loss.sum()/int(b*c*h*w)
        ddpm_loss.backward()
        self.optDDPM.step()
        self.log_dict['ddpm_loss'] = ddpm_loss.item()

    def test(self, continuous=True):
        self.cond_ddpm.eval()
        with torch.no_grad():
            if isinstance(self.cond_ddpm, nn.DataParallel):
                self.result = self.cond_ddpm.module.sampling(self.gt, self.condition, continuous)
            else:
                self.result = self.cond_ddpm.sampling(self.gt, self.condition, continuous)
        self.cond_ddpm.train()

    def sample(self, batch_size=1, continuous=False):
        self.cond_ddpm.eval()
        with torch.no_grad():
            if isinstance(self.cond_ddpm, nn.DataParallel):
                self.result = self.cond_ddpm.module.sample(batch_size, continuous)
            else:
                self.result = self.cond_ddpm.sample(batch_size, continuous)
        self.cond_ddpm.train()

    def set_loss(self):
        if isinstance(self.cond_ddpm, nn.DataParallel):
            self.cond_ddpm.module.set_loss(self.device)
        else:
            self.cond_ddpm.set_loss(self.device)

    def set_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.cond_ddpm, nn.DataParallel):
                self.cond_ddpm.module.set_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.cond_ddpm.set_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict


    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['result'] = self.result.detach().float().cpu()                 
        out_dict['GT'] = self.data['GT'].detach().float().cpu()                 
        out_dict['condition'] = self.condition.detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.cond_ddpm)
        if isinstance(self.cond_ddpm, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.cond_ddpm.__class__.__name__,
                                             self.cond_ddpm.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.cond_ddpm.__class__.__name__)

        logger.info(
            'Network conditional DDPM structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        diffusion_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_diff.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        diffusion_network = self.cond_ddpm
        if isinstance(self.cond_ddpm, nn.DataParallel):
            diffusion_network = diffusion_network.module
        state_dict = diffusion_network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, diffusion_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optDDPM.state_dict()
        torch.save(opt_state, opt_path)

        logger.info('Saved model in [{:s}] ...'.format(diffusion_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info('Loading pretrained model for conditional DDPM [{:s}] ...'.format(load_path))
            diffusion_path = '{}_diff.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            diffusion_network = self.cond_ddpm
            if isinstance(self.cond_ddpm, nn.DataParallel):
                diffusion_network = diffusion_network.module
            diffusion_network.load_state_dict(torch.load(diffusion_path),  strict=(not self.opt['model']['finetune_norm']))
            
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optDDPM.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
