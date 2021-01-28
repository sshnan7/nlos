#python train.py --nlayer 18 --name 201230_newdata_mixup_lamb_prob1 --gammas 0.1 --schedule 10 20 --nepoch 30 --lr 0.001 --multi-gpu --augment mixup

#python train.py --nlayer 18 --name 210105_follow --gammas 0.1 --schedule 10 20 --nepoch 30 --lr 0.001 --multi-gpu --normalize --gaussian
#python valid.py --multi-gpu 0 --gpu-num 1 --nlayer 18



# Train
#python train.py --nlayer 18 --name 210117_aug --batch-size 64 --gammas 0.1 --schedule 10 20 --nepoch 30 --lr 0.001 --gpu-num 1 --augment all --flatten
python train.py --name 210120_hrnet --batch-size 16 --nepoch 20 --gpu-num 1 --augment cutmix --flatten --arch hrnet


# Valid
#python valid.py --gpu-num 1 --nlayer 18 --flatten --arch hrnet
