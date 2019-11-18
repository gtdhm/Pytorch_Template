# ==================option.train_options.py=====================
# This module adds some train options for the total project.
#
# Version: 1.0.0
# Date: 2019.05.12
# =============================================================

import os
from option import BaseOptions
from util import mkdirs


###############################################################
# Train Options Class
###############################################################
class TrainOptions(BaseOptions):
    """This class includes training options. It also includes
    shared options defined in BaseOptions.

    Examples:
        use path:
        <<< cfg = option.TrainOptions()
            cfg.TRAIN_DATA_DIR
        use parameter:
        <<< cfg = option.TrainOptions()
            cfg.opts.model_name
    """

    def __init__(self):
        super(TrainOptions, self).__init__()
        self.mode = "Train"
        self.opts = self.add_parser()

        self.BEST_METRIC = 0.0 if self.opts.save_metric != 'FPR' else 1.0
        # Redefined the path
        self.CHECKPOINT_DIR = os.path.join(self.CHECKPOINT_DIR, self.opts.model_name + '_' + self.opts.dataroot)
        self.DATA_DIR = os.path.join(self.DATA_DIR, self.opts.dataroot)
        self.TRAIN_DATA_DIR = os.path.join(self.DATA_DIR, 'train')
        self.VAL_DATA_DIR = os.path.join(self.DATA_DIR, 'val')
        self.TRAIN_LABEL_DIR = os.path.join(self.DATA_DIR, self.opts.train_label)
        self.VAL_LABEL_DIR = os.path.join(self.DATA_DIR, self.opts.val_label)

        mkdirs(self.CHECKPOINT_DIR)
        # Display and save the path or parameter of paths
        self.display_options(self.opts, self.opts.display_path, self.opts.display_param)

    def add_parser(self):
        # Dataset base options
        self.parser.add_argument('--dataroot', type=str, default='multi_class_demo',
                                 help="your dataset file name")
        self.parser.add_argument('--train_label', type=str, default='train_split.csv',
                                 help="(.csv or .json) train label file name")
        self.parser.add_argument('--val_label', type=str, default='val_split.csv',
                                 help="(.csv or .json) val label file name")

        # Training parameters options
        self.parser.add_argument('--is_val', type=bool, default=True,
                                 help='use val or not when training')
        self.parser.add_argument('--epoch', type=int, default=20,
                                 help="epochs for train")
        self.parser.add_argument('--start_epoch', type=int, default=1,
                                 help="the starting epoch count")

        # Load checkpoints
        self.parser.add_argument('--load_checkpoint', type=str, default='scratch',
                                 help=">>> training from :[scratch]; or load [index | best | checkpoint_name.pth]")

        # Learning rate options
        self.parser.add_argument('--lr', type=float, default=1e-3,
                                 help="learning rate")
        self.parser.add_argument('--lr_scheduler', type=str, default='step',
                                 help="a learning rate scheduler >>> [linear | step | plateau | cosine]")
        self.parser.add_argument('--lr_gamma', type=float, default=0.1,
                                 help="the learning rate decay parameter: gamma")
        self.parser.add_argument('--lr_linear_fix', type=int, default=3,
                                 help="linear scheduler: learning rate in this stage is fixed")
        self.parser.add_argument('--lr_step_decay', type=str, default="188,397",
                                 help="step scheduler: Decays the learning rate of each parameter group by gamma")

        # Checkpoints save options
        self.parser.add_argument('--save_epoch', type=str, default="-1",
                                 help=">>> save last epoch:-1; save frequency of epoch:int")
        self.parser.add_argument('--save_metric', type=str, default="ACC",
                                 help="choose a metric to save model when it is best >>> [ACC | F1 | P | R | FPR]")
        self.parser.add_argument('--save_mode', type=str, default='param',
                                 help=">>> save parameters[param]; save parameters and optimizers[param,optim]")
        self.parser.add_argument('--save_train_log', type=bool, default=True,
                                 help="whether save the training loss and acc log to the disk")

        # Display relevant options
        self.parser.add_argument('--print_epoch', type=int, default=1,
                                 help="frequency of epoch to show training results on screen")
        self.parser.add_argument('--print_step', type=int, default=2,
                                 help="frequency of step to show training results on screen")
        # TODO(User) >>> add your own base parse
        #  parser.add_argument()
        # TODO(User): End
        opt = self.parser.parse_args()
        return opt
