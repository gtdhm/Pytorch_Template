# ========================train.py============================
# This module used to train the total project.
#
# Version: 1.0.0
# Date: 2019.05.20
# ============================================================
"""
======================Train the model=========================
python train.py --model_name DemoModel --net_name DemoNet --dataroot multi_class_demo
                --batch 32 --epoch 20 --lr 1e-3 --gpu_ids 0
                --load_checkpoint scratch --flip horizontal
==============================================================
"""

from option import TrainOptions
from database import load_database
from model import BaseModel
from util import MultiClassMetrics, print_train_info, print_val_info


def val_model(cfg, model, val_loader, val_flag, metrics, per_epoch):
    """val the model"""
    # inner loop for one batch
    for per_step, data in enumerate(val_loader):
        model.input(data[0], data[1])
        model.test()

        metrics.eval(data[1], model.out.cpu(), indicators="ACC, F1, FPR", step=len(val_loader))
        print_val_info(val_flag, cfg, [per_step + 1, len(val_loader)], metrics.metrics)

    if cfg.opts.save_metric == "FPR":
        if metrics.metrics[cfg.opts.save_metric] < model.BEST_METRIC:
            model.BEST_METRIC = metrics.metrics[cfg.opts.save_metric]
            model.save_model(per_epoch, ["Bestval"+cfg.opts.save_metric, metrics.metrics[cfg.opts.save_metric]])
    else:
        if metrics.metrics[cfg.opts.save_metric] > model.BEST_METRIC:
            model.BEST_METRIC = metrics.metrics[cfg.opts.save_metric]
            model.save_model(per_epoch, ["Bestval"+cfg.opts.save_metric, metrics.metrics[cfg.opts.save_metric]])


def train_model():
    """Train the model"""
    # 1. Get Training Options
    cfg = TrainOptions()

    # 2. Load train and val Dataset
    train_loader, val_loader = load_database(cfg)

    # 3. Create a Model
    model = BaseModel(cfg)

    # 4, Create metrics and visual CNN
    metrics = MultiClassMetrics(cfg.class_name)

    # 5. Outer loop for one epoch
    for per_epoch in range(model.start_epoch+1, cfg.opts.epoch+1):
        val_flag = False
        save_metrics = 0.0
        # inner loop for one batch
        for per_step, (images, labels, _) in enumerate(train_loader):
            model.input(images=images, labels=labels)
            model.train()

            metrics.eval(labels, model.out.cpu(), indicators="ACC, F1, FPR")
            save_metrics += metrics.metrics[cfg.opts.save_metric]
            val_flag = print_train_info(val_flag, cfg, [model.start_epoch+1, per_epoch, cfg.opts.epoch],
                                        [per_step+1, len(train_loader)], model.loss, model.lr, metrics.metrics)

        if cfg.opts.is_val:
            val_model(cfg, model, val_loader, val_flag, metrics, per_epoch)
        if per_epoch % cfg.opts.save_list == 0:
            model.save_model(per_epoch, ["train"+cfg.opts.save_metric, save_metrics/len(train_loader)])
        model.update_lr()


if __name__ == "__main__":
    train_model()
