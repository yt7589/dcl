车辆型号细粒度识别

### 安装过程
```bash
git clone https://github.com/yt7589/dcl.git
cp -R /bak_dir/datasets/CUB_200_2011 ./datasets/.
cp -R /bak_dir/models/pretrained ./models/.
mkdir net_model
cp -R /bak_dir/opencv2 ./trt/.
cp -R /bak_dir/thirdparty ./trt/.
cp -R /bak_dir/trt/models ./trt/vehicle_fgvc/.
# run command
python train.py --data CUB --epoch 360 --backbone resnet50 --cp 100 --sp 100 --tb 32 --tnw 16 --vb 256 --vnw 16  --lr 0.0008 --lr_step 50 --cls_lr_ratio 10 --start_epoch 0 --detail training_descibe --size 224 --crop 224 --cls_2 --swap_num 3 3
```



参考原型：https://github.com/JDAI-CV/DCL

