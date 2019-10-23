set -ex
python train.py --model_name DemoModel --net_name DemoNet --dataroot multi_class_demo --batch 32 --epoch 20 --lr 1e-3 --gpu_ids 0 --load_checkpoint scratch --flip horizontal
