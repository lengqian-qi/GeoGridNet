import argparse
import os
import click

class train_Options():
    def __init__(self):

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--lr', type=float, default=3e-3, help='learning rate')
        self.parser.add_argument("--batch_size", default=16, type=int)
        self.parser.add_argument("--epochs", default=200, type=int)
        self.parser.add_argument("--decay_epoch", default=50, type=float)
        self.parser.add_argument("--decay_rate", default=0.1, type=float)
        self.parser.add_argument("--augment", default=False) #preform data augmentation
        self.parser.add_argument("--num_workers", default=0, type=int)
        self.parser.add_argument("--num_classes", default=8, type=int)
        self.parser.add_argument("--pretrained_weights_path", default="")
        self.parser.add_argument('--lr_decay', type=float, default=1e-3, help='learning rate decay')

        self.parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

        # self.parser.add_argument("--root_dir", default="/data/cyzhuo/Scientific_Research/DATASETS/BRATS Dataset/BraTS2020_debug_data")
        
        self.parser.add_argument("--train_root_dir", default="")
        self.parser.add_argument("--val_root_dir", default="")

        self.parser.add_argument('--root_split', type=bool, default=True)
        self.parser.add_argument('--trainset_ratio', type=float, default=0.8)

        self.parser.add_argument('--checkpoint_dir', type=click.Path(file_okay=False), default='checkpoints')
        self.parser.add_argument('--experiment_name', type=str, default='')
        self.parser.add_argument('--checkpoint_path', default=None)
        self.parser.add_argument('--log_dir', default=None, help="If you want to continue, please set 'resume'")
        self.parser.add_argument('--save_latest_checkpoint', type=bool, default=True)

        self.parser.add_argument('--random_seed', type=int, default=2025)
        self.parser.add_argument('--val_interval', default=5)
        self.parser.add_argument('--save_checkpoint_freq', type=int, default=5)
        self.parser.add_argument('--val_freq', type=int, default=5)
        self.parser.add_argument('--save_pic_freq', type=int, default=100)

    def get_opt(self):
        self.opt = self.parser.parse_args()
        return self.opt

        

