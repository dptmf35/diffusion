import torch
import os
import model as Model
import numpy as np
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
from dataset import Data_set
from metrics import *


def parse_args():
    parse = argparse.ArgumentParser(description="New Network")
    parse.add_argument('--config', type=str, default='config/config.json')
    parse.add_argument('-p', '--phase', type=str, choices=['train', 'val'], default='train')
    parse.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parse.add_argument('-d', '--debug', action='store_true')
    parse.add_argument('-enable_wandb', default=True, action='store_true')
    parse.add_argument('-log_wandb_ckpt', action='store_true')
    parse.add_argument('-log_eval', action='store_true')

    args = parse.parse_args()
    return args


if __name__ == '__main__':
    # parse configs
    args = parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base') 
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])


    # Initialize WandbLogger
    if opt['enable_wandb']: 
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step') 
        wandb.define_metric('epoch') 
        wandb.define_metric("validation/*", step_metric="val_step") 
        val_step = 0 
    else:
        wandb_logger = None


    # dataset
    train_dataset= Data_set(opt['datasets']['train'], phase='train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt['datasets']['train']['batch_size'],
        shuffle=True
    )
    train_dataset_size = len(train_loader)
    print("train dataset size: ", train_dataset_size)

    val_dataset = Data_set(opt['datasets']['val'], phase='train')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt['datasets']['train']['batch_size'],
        shuffle=True
    )
    val_dataset_size = len(val_loader)
    print("test dataset size: ", val_dataset_size)

    test_dataset = Data_set(opt['datasets']['test'], phase='val')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt['datasets']['test']['batch_size'],
        shuffle=True
    )
    test_dataset_size = len(val_loader)
    print("test dataset size: ", test_dataset_size)

    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step   
    current_epoch = diffusion.begin_epoch  
    n_iter = opt['train']['n_iter']  
    n_epoch = opt['train']['n_epoch']

    
    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))
        
    diffusion.set_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for epoch_i, data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(data)
                diffusion.optimize_parameters() 
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)
            
                # save ckpt    
                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    idx = 0
                    image_num_idx = 0
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)
                    for _, val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continuous=True)
                        visuals = diffusion.get_current_visuals()
                        img_num = visuals['GT'].shape[0]
                        for i in range(img_num):
                            image_num_idx += 1
                            esti_img = Metrics.tensor2img(visuals['result'])
                            gt_img = Metrics.tensor2img(visuals['GT'])
                            cond_img = Metrics.tensor2img(visuals['condition'])

                            Metrics.save_img(esti_img, '{}/{}_{}_esti.png'.format(result_path, current_step, idx))
                            Metrics.save_img(gt_img, '{}/{}_{}_gt.png'.format(result_path, current_step, idx))
                            Metrics.save_img(cond_img, '{}/{}_{}_cond.png'.format(result_path, current_step, idx))

                            avg_psnr += Metrics.calculate_psnr(esti_img, gt_img)
                            avg_ssim += Metrics.calculate_ssim(esti_img, gt_img)


                    avg_psnr = avg_psnr / image_num_idx
                    avg_ssim = avg_ssim / image_num_idx

                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('ssim', avg_ssim, current_step)
                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_ssim': avg_ssim, 
                            'validation/val_step': val_step
                        })
                        val_step += 1

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')

    else:
        logger.info('Begin Model Evaluation.')

        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        image_num_idx=0
        result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
        os.makedirs(result_path, exist_ok=True)
        for _, test_data in enumerate(test_loader):
            idx += 1
            diffusion.feed_data(test_data)
            diffusion.test(continuous=True)
            visuals = diffusion.get_current_visuals()
            img_num = visuals['GT'].shape[0]
            for i in range(img_num):
                image_num_idx += 1
                esti_img = Metrics.tensor2img(visuals['result'][i])
                gt_img = Metrics.tensor2img(visuals['GT'][i])
                cond_img = Metrics.tensor2img(visuals['condition'][i])
                Metrics.save_img(esti_img, '{}/{}_{}_{}_esti.png'.format(result_path, current_step, idx, i))
                Metrics.save_img(gt_img, '{}/{}_{}_{}_gt.png'.format(result_path, current_step, idx, i))
                Metrics.save_img(cond_img, '{}/{}_{}_{}_cond.png'.format(result_path, current_step, idx, i))

                eval_psnr = Metrics.calculate_psnr(esti_img, gt_img)
                avg_psnr += Metrics.calculate_psnr(esti_img, gt_img)
                avg_ssim += Metrics.calculate_ssim(esti_img, gt_img)
    
        avg_psnr = avg_psnr / image_num_idx
        avg_ssim = avg_ssim / image_num_idx

        logger.info('# Test # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Test # SSIM: {:.4e}'.format(avg_ssim))
        tb_logger.add_scalar('psnr', avg_psnr, current_step)
        tb_logger.add_scalar('ssim', avg_ssim, current_step)
        if wandb_logger:
            wandb_logger.log_metrics({
                'test/val_psnr': avg_psnr,
                'test/val_ssim': avg_ssim
            })