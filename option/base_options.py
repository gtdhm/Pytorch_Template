# ==================option.base_options.py=====================
# This module contains some base options for the total project.
#
# Version: 1.0.0
# Date: 2019.05.12
# =============================================================

import os
import argparse
from util import cal_equal, wrote_txt_file, TOTAL_LENGTH


###############################################################
# Base Options Class
###############################################################
class BaseOptions(object):
    """This class defines base options, likes common path and
    useful parameter, which can be used during both training
    and testing time.
    """
    # TODO(User) >>> add your own path
    # Root directory of the project, based on the position of train.py
    ROOT_DIR = ".\\" if os.name == "nt" else "./"
    # data set path
    DATA_DIR = os.path.join(ROOT_DIR, 'dataset')
    # checkpoint path
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'work_dir')
    # train data path
    TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
    # val_data path
    VAL_DATA_DIR = os.path.join(DATA_DIR, 'val')
    # test data path
    TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')

    def __init__(self):
        # define the mode is train or not
        self.mode = "Base"
        self.parser = self.get_parser()
        # TODO(User) >>> modify the class name of your labels in Order!
        self.class_name = ['zero', 'one', 'two', 'three', 'four', 'five']
        # TODO(User): End

    @staticmethod
    def get_parser():
        """get the parser object"""
        parser = argparse.ArgumentParser(description='Customer Model Template based on Pytorch.',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # model base options
        parser.add_argument('--model_name', type=str, default='DemoModel',
                            help="a recognizable model name")
        parser.add_argument('--net_name', type=str, default='DemoNet',
                            help="a recognizable network name")
        parser.add_argument('--seed', type=int, default=3,
                            help="set a manual seed for random, np.random and torch.manual_seed")

        # Network base options
        parser.add_argument('--input_size', type=tuple, default=(224, 224, 3),
                            help='the input image dim (w * h * c)')
        parser.add_argument('--batch', type=int, default=32,
                            help='set a batch size inputs')
        parser.add_argument('--workers', type=int, default=2,
                            help='number of the workers')
        parser.add_argument('--gpu_ids', type=str, default='0',
                            help=">>> use multi GPU:[0,1,2]; use CPU:[-1]")
        parser.add_argument('--benchmark', type=bool, default=True,
                            help="whether use cudnn benchmark or not")

        parser.add_argument('--init_type', type=str, default='normal',
                            help="network initialization >>> [normal | xavier | kaiming | orthogonal]")

        # Data augmentation options
        parser.add_argument('--flip', type=str, default=None,
                            help='random flip the images for data augmentation >>> [horizontal | vertical]')
        parser.add_argument('--rotate', type=str, default=None,
                            help="random rotate the images for data augmentation >>> [0,360]")

        # Display relevant options
        parser.add_argument('--display_path', type=bool, default=True,
                            help="whether show model paths on screen")
        parser.add_argument('--display_param', type=bool, default=True,
                            help="whether show model parameters on screen")
        parser.add_argument('--display_net', type=bool, default=True,
                            help="whether show network architecture on screen")

        # TODO(User) >>> add your own base parse
        #  parser.add_argument()
        # TODO(User): End
        return parser

    def param_message(self, args):
        """Parameter options values."""
        message = ''
        equal_left, equal_right = cal_equal(len(self.mode)+10)
        message += '\n'+('=' * equal_left)+' '+self.mode+' Options '+('=' * equal_right)+'\n'
        for k, v in sorted(vars(args).items()):
            if 'checkpoint' in k:
                continue
            left = " " * (round(TOTAL_LENGTH/2) - len(str(k)+" :"))
            right = " " + str(v) + "\n"
            if TOTAL_LENGTH >= (len(left + str(k) + " :" + right) + 6):
                left = ">>>" + " " * (round(TOTAL_LENGTH/2) - len(str(k)+" :") - 3)
                right = " " + str(v) + " " * (round(TOTAL_LENGTH - 3 - len(left + str(k) + " : " + str(v)))) + "<<<\n"
            message += left + str(k) + " :" + right
        equal_left, equal_right = cal_equal(5)
        message += '=' * equal_left + ' End ' + '=' * equal_right
        return message

    def path_message(self):
        """Path options values"""
        def add_message(part1, part2):
            left = " " * (round(TOTAL_LENGTH / 3) - len(part2 + ':'))
            right = getattr(part1, part2) + '\n'
            if TOTAL_LENGTH >= (len(left + part2 + ': ' + right) + 6):
                left = ">>>" + " " * (round(TOTAL_LENGTH / 3) - len(part2 + ':') - 3)
                right = getattr(part1, part2) + " " * (
                    round(TOTAL_LENGTH - 3 - len(left + part2 + ': ' + getattr(part1, part2)))) + "<<<\n"
            return left + part2 + ': ' + right

        message = ''
        equal_left, equal_right = cal_equal(len(self.mode) + 8)
        message += '\n'+('=' * equal_left)+' '+self.mode+' Paths '+('=' * equal_right)+'\n'
        for a in dir(self):
            if self.mode == 'Train':
                if not a.startswith("__") and not callable(getattr(self, a)) and \
                        a.split('_')[-1] == 'DIR' and 'RESULT' not in a and 'TEST' not in a:
                    message += add_message(self, a)
            elif self.mode == 'Test':
                if not a.startswith("__") and not callable(getattr(self, a)) and \
                        a.split('_')[-1] == 'DIR' and 'TRAIN' not in a and 'VAL' not in a:
                    message += add_message(self, a)
        equal_left, equal_right = cal_equal(5)
        message += '=' * equal_left + ' End ' + '=' * equal_right
        return message

    def display_options(self, args, display_path, display_param):
        """Save and Display parameter options or path options"""
        path_message = self.path_message()
        param_message = self.param_message(args)
        if display_path:
            print(path_message)
        if display_param:
            print(param_message)
        file_name = self.mode + '_Options.txt'
        wrote_txt_file(os.path.join(self.CHECKPOINT_DIR, file_name), path_message+'\n'+param_message)
