import argparse
import os
import sys
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from torchvision.utils import save_image
from utils.make_deterministic import make_deterministic
from utils.metrics_plots import calculate_metrics
from PIL import Image
from typing import *
from matplotlib import pyplot as plt
from pathlib import Path
import click
from torch.cuda.amp import autocast as autocast
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
import cv2
from torchvision.utils import save_image
from dualbranch_dataset import h5_dataset, split_list_ratio
from train_options import train_Options
from model_dual_branch import FusionNet

def build_DataLoader(opt) -> Tuple[DataLoader, DataLoader]:

    if opt.root_split:
        train_dataset = h5_dataset(root=opt.train_root_dir)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.batch_size, shuffle=True,
                                                   num_workers=opt.num_workers)
        train_num = len(train_dataset)

        val_dataset = h5_dataset(root=opt.val_root_dir)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=opt.batch_size, shuffle=False,
                                                 num_workers=opt.num_workers)
        val_num = len(val_dataset)

    else:
        sample_list = sorted(os.listdir(opt.train_root_dir))
        train_list, val_list = split_list_ratio(sample_list, split_ratio = 0.8)
        train_dataset = h5_dataset(root=opt.train_root_dir, sample_list=train_list)
        val_dataset = h5_dataset(root=opt.train_root_dir, sample_list=val_list)

        train_num, val_num = len(train_dataset), len(val_dataset)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.batch_size, shuffle=True,
                                                   num_workers=opt.num_workers)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=opt.batch_size, shuffle=False,
                                                 num_workers=opt.num_workers)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))
    print('Using {} dataloader workers every process'.format(opt.num_workers))

    return train_loader, val_loader

def build_network(checkpoint, opt):

    model = FusionNet(num_classes=8,
                 use_arc_encoding=True,
                 use_attention=True,
                 spatial_base_channels=32,
                 spatial_num_blocks=4,
                 fusion_method='cross_attention', 
                 fusion_dim=512,
                 dropout_rate=0.3)

    model.to(opt.device)

    if opt.pretrained_weights_path != "":
        model_weight_path = opt.pretrained_weights_path
        assert os.path.exists(model_weight_path), "weights file: {} does not exist.".format(model_weight_path)
        weights_dict = torch.load(model_weight_path)
        # if the pretrained weights don't match the model, del some bias and weight
        print(model.load_state_dict(weights_dict, strict=False))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=opt.lr)
    if checkpoint['model_state_dict'] is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    if checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

def adjust_learning_rate(optimizer, iteration_count, lr_decay):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(optimizer, iteration_count, lr):
    """Imitating the original implementation"""
    lr = lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def initialize_tensorboard_writer(opt):
    if opt.log_dir is not None:
        if opt.log_dir == 'resume':
            try:
                log_dir = Path(opt.checkpoint_path).parent / 'logs'
            except Exception as e:
                print("When log-dir == resume, checkpoint-path is must be given")
        else:
            log_dir = str(Path(opt.log_dir))

    else:
        log_dir = str(Path(opt.checkpoint_dir) / f'logs')
    print('Save tensorboard log dir:', log_dir)
    writer = SummaryWriter(log_dir)
    return writer

def initialize_checkpoint():
    checkpoint = {
        'epoch_num': 0,
        'model_state_dict': None,
        'optimizer_state_dict': None,
        'loss': 0,
    }
    return checkpoint

def iter_one_epoch(dataLoader: DataLoader,
                   model: torch.nn.Module,
                   optimizer=None,
                   header: str = '',
                   epoch: int = 0,
                   istrain: bool = True,
                   opt=None):

    device = opt.device
    if istrain:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    bar = tqdm(dataLoader, file=sys.stdout)
    train_labels_lst, train_predicted_lst = [], []
    
    for step, datas in enumerate(bar):
        if step < 1e4:
            warmup_learning_rate(optimizer, iteration_count=step, lr=opt.lr)
        else:
            adjust_learning_rate(optimizer, iteration_count=step, lr_decay=opt.lr_decay)

        fibermap, point_set, label = datas

        if label.dim() == 2:
            label = label[:, 0]  # [B,1] -> [B]
            
        fibermap, point_set, label = fibermap.to(device), point_set.to(device), label.to(device)
        
        pred = model(point_set, fibermap)  # Output shape: [B, num_classes]
        loss = F.cross_entropy(pred, label)
        
        label = label.cpu().detach().numpy().tolist()
        train_labels_lst.extend(label)
        
        _, pred_idx = torch.max(pred, dim=1)
        pred_idx = pred_idx.cpu().detach().numpy().tolist()
        train_predicted_lst.extend(pred_idx)

        if istrain:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        bar.desc = f"{{{header}}} loss:{running_loss / (step+1):.3f}"

    mac_precision, mac_recall, mac_f1, accuracy = calculate_metrics(train_labels_lst, train_predicted_lst)
    loss_mean = running_loss / len(dataLoader)

    print(f'{header} loss:{loss_mean:.4f}')

    return model, loss_mean, mac_precision, mac_recall, mac_f1, accuracy


def train(opt):
    # set seed
    make_deterministic(opt.random_seed)

    if opt.checkpoint_path is not None:
        checkpoints = torch.load(opt.checkpoint_path,
                                 map_location=opt.device,
                                 weights_only=True)
        print('load parameters from:', opt.checkpoint_path)
    else:
        checkpoints = initialize_checkpoint()

    writer = initialize_tensorboard_writer(opt)
    train_dl, valid_dl = build_DataLoader(opt)
    model,optimizer = build_network(checkpoints, opt)

    start_epoch = checkpoints['epoch_num']
    print('start_epoch:', start_epoch)

    for epoch in range(start_epoch, start_epoch + opt.epochs):
        # train one epoch
        model, loss_mean, mac_precision, mac_recall, mac_f1, accuracy = iter_one_epoch(
            dataLoader=train_dl,
            model=model,
            optimizer=optimizer,
            header=f"Train Epoch [{epoch}/{start_epoch + opt.epochs - 1}]",
            epoch=epoch,
            istrain=True,
            opt=opt)

        writer.add_scalar(f'Training loss', loss_mean, epoch)
        writer.add_scalar(f'Training recision', mac_precision, epoch)
        writer.add_scalar(f'Training recall', mac_recall, epoch)
        writer.add_scalar(f'Training f1', mac_f1, epoch)
        writer.add_scalar(f'Training accuracy', accuracy, epoch)

        # update_checkpoint
        checkpoints['epoch_num'] += 1
        checkpoints['model_state_dict'] = model.state_dict()
        checkpoints['optimizer_state_dict'] = optimizer.state_dict()
        checkpoints['loss'] = loss_mean

        #  save the checkpoint
        if opt.save_latest_checkpoint:
            torch.save(checkpoints, str(Path(opt.checkpoint_dir) / f'latest.pt'))

        if (epoch + 1) % opt.save_checkpoint_freq == 0:
            torch.save(checkpoints, str(Path(opt.checkpoint_dir) / f'epoch{epoch}.pt'))

        #  validate one epoch
        if (epoch + 1) % opt.val_freq == 0:
            with torch.no_grad():
                model, loss_mean, mac_precision, mac_recall, mac_f1, accuracy = iter_one_epoch(
                    dataLoader=valid_dl,
                    model=model,
                    optimizer=optimizer,
                    header=f"Val Epoch [{epoch}/{start_epoch + opt.epochs - 1}]",
                    epoch=epoch,
                    istrain=False,
                    opt=opt)

                writer.add_scalar(f'Val loss', loss_mean, epoch)
                writer.add_scalar(f'Val recision', mac_precision, epoch)
                writer.add_scalar(f'Val recall', mac_recall, epoch)
                writer.add_scalar(f'Val f1', mac_f1, epoch)
                writer.add_scalar(f'Val accuracy', accuracy, epoch)

            writer.add_scalar(f'Val loss', loss_mean, epoch)

        writer.flush()

    print('Finished Training')


def main():
    opt = train_Options().get_opt()
    print('device:',opt.device)
    # update checkpoint dir with time suffix
    if opt.checkpoint_path is not None and opt.log_dir == 'resume':
        opt.checkpoint_dir= Path(opt.checkpoint_path).parent
    else:
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S")  # Generate current timestamp (formatted as YYYYMMDD_HHMMSS)
        opt.checkpoint_dir = opt.checkpoint_dir + '/' + opt.experiment_name + '/' + timestamp
        # create dir to save weights
        Path(opt.checkpoint_dir).mkdir(parents=True)
    print(f"Save dir: {opt.checkpoint_dir}")

    # execute training progress
    train(opt)

if __name__ == '__main__':
    main()
