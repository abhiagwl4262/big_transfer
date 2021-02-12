### Train Resnet BiT model with Big Transfer based training strategy #########
CUDA_VISIBLE_DEVICES=0 python3 train.py --batch_size 32 --model BiT-M-R50x1 --name leaf_disease_R50 --logdir tmp/bit_logs --dataset imagenet2012 --datadir /mc2/datasets/leaf_disease/

### Train Resnet BiT model  with new OnecycleLR base LR scheduler #########
CUDA_VISIBLE_DEVICES=0 python3 train_new.py --batch_size 32 --model BiT-M-R50x1 --name leaf_disease_R50_new --logdir tmp/bit_logs --dataset imagenet2012 --datadir /mc2/datasets/leaf_disease/



### Train efficientnet model with BIg Transfer based training strategy #########
CUDA_VISIBLE_DEVICES=0 python3 train_efficientnet.py --batch_size 32 --model efficientnet-b2 --name leaf_disease_efficient_b2 --logdir tmp/bit_logs --dataset imagenet2012 --datadir /mc2/datasets/leaf_disease/

### Train efficientnet model with new OnecycleLR base LR scheduler #########
CUDA_VISIBLE_DEVICES=0 python3 train_new_efficientnet.py --batch_size 32 --model efficientnet-b2 --name leaf_disease_efficient_b2_new --logdir tmp/bit_logs --dataset imagenet2012 --datadir /mc2/datasets/leaf_disease/
