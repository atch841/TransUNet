set -x
folder=vit_pretrained_aug_glob_mlplr08
for ep in 20 40 60 100 200 400

do

    python train.py --dataset LiTS_tumor_1p_half --img_size 256 --batch_size 40 --max_epoch 80 --is_pretrain /home/viplab/data/moco/output/$folder/ckpt_epoch_$ep.pth --pretrain_epoch $ep --pretrain_folder $folder

done
set +x