set -x
folder=vit_pretrained_aug_glob
for ep in 100 150 200 250

do

    python train.py --dataset LiTS_tumor_1p_half --img_size 256 --batch_size 16 --max_epoch 80 --is_pretrain /home/viplab/nas/moco/output/$folder/ckpt_epoch_$ep.pth --pretrain_epoch $ep --pretrain_folder $folder

done
set +x