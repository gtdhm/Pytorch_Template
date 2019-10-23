# ===================model.base_model.py======================
# This module implements a base model for the project.
#
# Version: 1.0.0
# Date: 2019.05.20
# ============================================================

import os
import time
import torch
from torch import nn
from torch.nn import init
from torch import optim
from model import BaseNetwork
from util import cal_equal, list_sort


###############################################################
# BaseModel Class
###############################################################
class BaseModel(object):
    """This class includes base processing for the model.
    Inputs:
        cfg: the total options

    Examples:
        <<< model = BaseModel(cfg)
            model.input(data)
            model.train()
            model.update_lr()
            model.save_model()
    """

    def __init__(self, cfg):
        # Init training variable
        self.cfg = cfg
        self.opts = cfg.opts
        self.best_cache = 'none'
        if cfg.mode == 'Train':
            self.BEST_METRIC = cfg.BEST_METRIC
            self.metric = 0  # Define metric, which used for learning rate policy 'plateau'
            self.loss = 1.0
            self.lr = cfg.opts.lr
            self.start_epoch = cfg.opts.start_epoch - 1
            self.result_path = None
            self.batch_x = self. batch_y = self.out = None

        # Init torch device
        torch.manual_seed(cfg.opts.seed)
        self.gpu_ids = list(map(int, self.opts.gpu_ids.split(',')))
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) \
            if -1 not in self.gpu_ids and torch.cuda.is_available() else torch.device('cpu')
        if self.opts.benchmark:
            torch.backends.cudnn.benchmark = True

        # TODO(User): redefine the following >>> self.network, self.optimizer, self.criterion, self.metric
        # 1. Define network
        self.network = BaseNetwork(len(cfg.class_name))
        if cfg.mode == 'Train':
            # 2. Define optimizer
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.opts.lr, betas=(0.9, 0.999), eps=1e-8)
            # 3. Create scheduler
            self.scheduler = self._create_scheduler()
            # 4. Define loss
            self.criterion = nn.CrossEntropyLoss()  # FocalLoss()
        # 5. Init weight
        self._weights_init()
        # TODO(User): End

        self._display_network(verbose=self.opts.display_net)
        # 6. Load model checkpoint
        if cfg.opts.load_checkpoint != 'scratch':
            self.load_model()

    def input(self, images, labels):
        """Unpack input data from the dataloader and perform
        necessary pre-processing steps.
        Inputs:
            images: the batch images itself
            labels: the batch labels itself
        """
        self.batch_x = images.to(self.device)
        self.batch_y = labels.to(self.device)

    def forward(self):
        """Run forward pass for the network."""
        self.out = self.network(self.batch_x)

    def backward(self):
        """Calculate the Total Loss and the Backward Gradients for
        the network.
        """
        self.loss = self.criterion(self.out, self.batch_y)
        self.loss.backward()

    def train(self):
        """Training flow of the whole model."""
        # Enable the Backward
        self.set_requires_grad([self.network], True)
        # Run the Forward
        self.forward()
        # Clean up Gradients
        self.optimizer.zero_grad()
        # Run the Backward
        self.backward()
        # Update the weights
        self.optimizer.step()

    def test(self):
        """val or test the whole model."""
        with torch.no_grad():
            # Disable the Backward
            self.set_requires_grad([self.network], False)
            self.forward()

    def update_lr(self):
        """Update learning rates for all the networks; called at the
        end of every epoch.
        """
        if self.opts.lr_scheduler == 'plateau':
            self.scheduler.step(self.metric)
        else:
            self.scheduler.step()
        self.lr = self.optimizer.param_groups[0]['lr']

    def save_model(self, current_epoch, metrics=None):
        """Save all the networks to the disk.
        Inputs:
            current_epoch: current epoch of the total epoch
            metrics: a current list like [metric name, metric value]
        """
        current_time = time.strftime("%m%d%H%M%S", time.localtime())
        save_filename = '[{}]_epoch:{}_{}:{:.3f}%_batch:{}_lr:{}_mode:{}_time:{}.pth' \
            .format(self.opts.net_name, current_epoch, metrics[0], metrics[1] * 100,
                    self.opts.batch, self.lr, self.opts.save_mode, current_time)
        if "Best" in metrics[0]:
            if os.path.isfile(os.path.join(self.cfg.CHECKPOINT_DIR, self.best_cache)):
                os.remove(os.path.join(self.cfg.CHECKPOINT_DIR, self.best_cache))
                save_filename = '[{}]_epoch:{}_{}:{:.3f}%_batch:{}_lr:{}_mode:{}_time:{}.pth' \
                    .format(self.opts.net_name, current_epoch, metrics[0], metrics[1] * 100,
                            self.opts.batch, self.lr, self.opts.save_mode, current_time)
            self.best_cache = save_filename
        save_path = os.path.join(self.cfg.CHECKPOINT_DIR, save_filename)

        if 'param' in self.opts.save_mode:
            save_store = {'network_state_dict': self.network.state_dict(),
                          'epoch': current_epoch, 'metric': metrics[1]}
            if 'optim' in self.opts.save_mode:
                save_store.update({'optimizer_state_dict': self.optimizer.state_dict()})
            torch.save(save_store, save_path)
        else:
            raise IOError("[Error] Save mode in options --> '{:s}' was not found...".format(self.opts.save_mode))
        print(">>> Saving model->%s" % save_filename)

    def load_model(self):
        """Load all the networks from the disk."""
        save_path = self._find_checkpoint_path()
        if 'param' in self.opts.save_mode:
            checkpoint = torch.load(save_path)
            if 'network_state_dict' in checkpoint.keys():
                self.network.load_state_dict(checkpoint['network_state_dict'])
                self.BEST_METRIC = float(checkpoint['metric'])
                self.start_epoch = int(checkpoint['epoch'])
                if 'optim' in self.opts.save_mode:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            else:
                network_dict = self.network.state_dict()
                checkpoint = {k: v for k, v in checkpoint.items() if k in network_dict}
                network_dict.update(checkpoint)
                self.network.load_state_dict(network_dict)
        else:
            raise IOError("[Error] Save mode in options --> '{:s}' was not found!".format(self.opts.save_mode))
        checkpoint_name = save_path.split(self.cfg.CHECKPOINT_DIR)[-1][1:-4]
        print(">>> Loading model->%s" % (checkpoint_name + '.pth'))
        if self.cfg.mode == 'Train':
            if self.start_epoch >= self.opts.epoch:
                print("\n[Warning] Epoch:{} in options should be larger than Epoch:{} in checkpoints!"
                      .format(self.start_epoch, self.opts.epoch))
        elif self.cfg.mode == 'Test' and self.opts.test_label == 'None':
            self.result_path = os.path.join(self.cfg.CHECKPOINT_DIR, checkpoint_name + '.csv')
            open(self.result_path, 'w', newline='', encoding="utf-8-sig")

    def _find_checkpoint_path(self):
        """Find the right checkpoint name."""
        if '.pth' not in self.opts.load_checkpoint:
            dir_names = []
            if 'best' in self.opts.load_checkpoint:
                for name in os.listdir(self.cfg.CHECKPOINT_DIR):
                    if name.startswith('[' + self.opts.net_name + ']') and 'Best' in name and name.endswith('.pth'):
                        dir_names.append(name)
                if len(dir_names) != 0:
                    save_path = os.path.join(self.cfg.CHECKPOINT_DIR, dir_names[0])
                    self.best_cache = dir_names[0]
                else:
                    raise IOError("[Error] No checkpoint file in {} ...".format(self.cfg.CHECKPOINT_DIR))
            else:
                for name in os.listdir(self.cfg.CHECKPOINT_DIR):
                    if name.startswith('['+self.opts.net_name+']') and name.endswith('.pth'):
                        dir_names.append(name)
                if len(dir_names) != 0:
                    dir_names = list_sort(dir_names, index=-1, mode="chars")
                    save_path = os.path.join(self.cfg.CHECKPOINT_DIR, dir_names[int(self.opts.load_checkpoint)])
                else:
                    raise IOError("[Error] No checkpoint file in {} ...".format(self.cfg.CHECKPOINT_DIR))
        else:
            save_path = os.path.join(self.cfg.CHECKPOINT_DIR, self.opts.load_checkpoint)
        return save_path

    def _weights_init(self):
        """Register CPU/GPU device (with multi-GPU support) and Initialize the network weights.
        """
        def common_init(m):
            class_name = m.__class__.__name__
            if hasattr(m, 'weight') and (class_name.find('Conv') != -1 or class_name.find('Linear') != -1):
                if self.opts.init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, 0.02)
                elif self.opts.init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=0.02)
                elif self.opts.init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif self.opts.init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=0.02)
                else:
                    raise IOError("[Error] Weight init type --> '{:s}' was not found..."
                                  .format(self.opts.init_type))
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

            elif class_name.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

        self.network.to(self.device)
        # multi-GPUs
        if len(self.gpu_ids) > 1:
            self.network = torch.nn.DataParallel(self.network, self.gpu_ids)
        self.network.apply(common_init)

    def _create_scheduler(self):
        """Create a learning rate scheduler."""
        if self.opts.lr_scheduler == "linear":
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 - (self.opts.lr_linear_fix - self.start_epoch)) \
                       / float((self.opts.epoch-self.opts.lr_linear_fix)+1)
                return lr_l
            scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        elif self.opts.lr_scheduler == 'step':
            self.opts.lr_step_decay = list(map(int, self.opts.lr_step_decay.split(',')))
            lr_step_decay = torch.tensor(self.opts.lr_step_decay, requires_grad=False)
            lr_step_decay = lr_step_decay - torch.full_like(lr_step_decay, self.start_epoch+1)
            scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                       milestones=lr_step_decay.numpy().tolist(),
                                                       gamma=0.1)
        elif self.opts.lr_scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2,
                                                             threshold=0.01, patience=5)
        elif self.opts.lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLr(self.optimizer,
                                                             T_max=self.opts.lr_linear_fix-self.start_epoch,
                                                             eta_min=0)
        else:
            raise IOError("[Error] LR scheduler --> '{:s}' was not found...".format(self.opts.lr_scheduler))
        return scheduler

    @ staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set parameter().requies_grad for all the networks to avoid
        unnecessary computations.
        Inputs:
            nets: a list of networks
            requires_grad: whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def _display_network(self, verbose=False):
        """Print the total number of parameters and architecture
        in the network.
        Inputs:
            verbose: print the network architecture or not(bool)
        """
        equal_left, equal_right = cal_equal(22)
        print("\n"+"="*equal_left+" Networks Initialized "+"="*equal_right)
        num_params = 0
        for param in self.network.parameters():
            num_params += param.numel()
        if verbose:
            print(self.network)
        print('>>> [%s] Total size of parameters : %.3f M' % (self.opts.net_name, num_params / 1e6))
        print('>>> [%s] Weights initialize with : %s ' % (self.opts.net_name, self.opts.init_type))
        if self.cfg.mode == 'Train':
            print('>>> [%s] Learning rate scheduler : %s ' % (self.opts.net_name, self.opts.lr_scheduler))
        equal_left, equal_right = cal_equal(6)
        print('%s Done %s' % ('=' * equal_left, '=' * equal_right))
        print('>>> [%s] was created ...' % type(self).__name__)
