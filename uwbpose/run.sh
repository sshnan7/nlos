#python train.py --nlayer 18 --name 201230_newdata_mixup_lamb_prob1 --gammas 0.1 --schedule 10 20 --nepoch 30 --lr 0.001 --multi-gpu --augment mixup

#python train.py --nlayer 18 --name 210105_follow --gammas 0.1 --schedule 10 20 --nepoch 30 --lr 0.001 --multi-gpu --normalize --gaussian
#python valid.py --multi-gpu 0 --gpu-num 1 --nlayer 18



# Train
#python train.py --nlayer 18 --name 210117_aug --batch-size 64 --gammas 0.1 --schedule 10 20 --nepoch 30 --lr 0.001 --gpu-num 1 --augment all --flatten
#python train.py --name 210217_since --batch-size 32 --nepoch 20 --cutoff 256 --gpu-num 1 --augment cutmix --flatten #--arch hrnet
python train.py --name 210507_model_final007 --batch-size 64 --nepoch 60 --cutoff 256 --gpu-num 1 --flatten --normalize --gamma 0.5 #--arch hrnet


# Valid
#python valid.py --gpu-num 1 --batch-size 32 --nlayer 18 --flatten #--arch hrnet
python valid.py --gpu-num 0 --nlayer 18 --batch-size 64 --cutoff 256 --flatten --normalize #--arch hrnet