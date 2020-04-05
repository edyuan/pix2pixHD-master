python train.py --name LEv1hd --dataroot ./datasets/LEv1hd
python test.py --name LEv1hd --dataroot ./datasets/LEv1hd


python -m torch.distributed.launch train.py --name LEv0hdamp --dataroot ./datasets/LEv0hd --continue_train --fp16

