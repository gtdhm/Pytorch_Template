# ==================option.test_options.py=====================
# This module adds some test options for the total project.

# Version: 1.0.0
# Date: 2019.05.12
# =============================================================

import os
from option import BaseOptions


###############################################################
# Test Options Class
###############################################################
class TestOptions(BaseOptions):
    """This class includes testing options. It also includes
    shared options defined in BaseOptions.
    Inputs:
        cfg:
    Examples:
        use path:
        <<< cfg = option.TestOptions()
            cfg.Test_DATA_DIR
        use parameter:
        <<< cfg = option.TestOptions()
            cfg.opts.model_name
    """

    def __init__(self):
        super(TestOptions, self).__init__()
        self.mode = "Test"
        self.opts = self.add_parser()

        # redefined the path
        self.CHECKPOINT_DIR = os.path.join(self.CHECKPOINT_DIR, self.opts.model_name + '_' + self.opts.dataroot)
        self.DATA_DIR = os.path.join(self.DATA_DIR, self.opts.dataroot)
        self.TEST_DATA_DIR = os.path.join(self.DATA_DIR, 'test')
        self.TEST_LABEL_DIR = os.path.join(self.DATA_DIR, self.opts.test_label) \
            if self.opts.test_label != 'None' else 'None'

        assert self.opts.load_checkpoint != 'scratch', "[Error] You must load checkpoint file in Test mode!"
        # Display and save the path or parameter of paths
        self.display_options(self.opts, self.opts.display_path, self.opts.display_param)

    def add_parser(self):
        # Dataset  base options
        self.parser.add_argument('--dataroot', type=str, default='multi_class_demo',
                                 help="your dataset file name")
        self.parser.add_argument('--test_label', type=str, default='test_split.csv',
                                 help="(.csv or .json)test file name if it has a test label")
        # Test base options
        self.parser.add_argument('--num_test', type=int, default=120,
                                 help='how many test images to run')

        # load checkpoints
        self.parser.add_argument('--load_checkpoint', type=str, default='best',
                                 help=">>> testing from load [index | best | checkpoint_name.pth]")

        # Checkpoints save options
        self.parser.add_argument('--save_mode', type=str, default='param',
                                 help=">>> test only support parameters mode[param]")
        self.parser.add_argument('--save_test_log', type=bool, default=True,
                                 help="whether save the training loss and acc log to the disk")
        # TODO(User) >>> add your own base parse
        #  parser.add_argument()
        # TODO(User): End
        opt = self.parser.parse_args()
        return opt

