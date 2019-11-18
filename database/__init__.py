# ================database.__init__.py=======================
# This package includes some database modules: base dataset.

# Written by: GT
# Date: 2019.05.20
# ============================================================
__version__ = '1.0.0'

"""Database Modules"""
from .base_dataset import *
from torch.utils.data import DataLoader


def load_database(cfg):
    """Create a dataset given the options.This is the main interface
    between this package and 'train.py'/'test.py'.
    Inputs:
        cfg: the total options
    Returns:
        train_loader: a train object of DataLoader
        val_loader: a val object of DataLoader
        test_loader: a test object of DataLoader
    """
    if cfg.mode == 'Train':
        train_db = BaseDataset(cfg, use_trans=True)
        train_db.load_data(mode='Train')
        train_loader = DataLoader(dataset=train_db,
                                  batch_size=cfg.opts.batch,
                                  shuffle=True,
                                  num_workers=cfg.opts.workers)

        if cfg.opts.is_val:
            val_db = BaseDataset(cfg)
            val_db.load_data(mode='Val')
            val_loader = DataLoader(dataset=val_db,
                                    batch_size=cfg.opts.batch,
                                    shuffle=True,
                                    num_workers=cfg.opts.workers)
        else:
            val_loader = None

        if '-1' in cfg.opts.save_epoch:
            cfg.opts.save_list = int(cfg.opts.save_epoch.split(',')[0].replace('-1', str(cfg.opts.epoch)))
        else:
            cfg.opts.save_list = int(cfg.opts.save_epoch.split(',')[0].replace(' ', ''))
        print(">>> [%s] was created ..." % type(train_db).__name__)
        return train_loader, val_loader

    elif cfg.mode == 'Test':
        test_db = BaseDataset(cfg)
        test_db.load_data(mode='Test')
        if cfg.opts.num_test < cfg.opts.batch:
            cfg.opts.batch = cfg.opts.num_test
        test_loader = DataLoader(dataset=test_db,
                                 batch_size=cfg.opts.batch,
                                 shuffle=False,
                                 num_workers=cfg.opts.workers)
        print(">>> [%s] was created ..." % type(test_db).__name__)
        return test_loader

    else:
        raise IOError("[Error] opts.mode --> '{:s}' in options should be 'Train' or 'Test'...".format(cfg.mode))

