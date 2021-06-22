set -x
folder=output1
for ep in 20 40 60 100 150 200 

do

    python train.py --model deeplab_resnest --dataset LiTS_tumor_1p_half --img_size 256 --batch_size 10 --max_epoch 160 --is_pretrain /home/viplab/nas/moco_temp/$folder/ckpt_epoch_$ep.pth --pretrain_epoch $ep --pretrain_folder $folder

done
set +x