```bash
curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth -o experiments/pretrained_models/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth

# tuning
## Get Required Pretrained model
curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -o experiments/pretrained_models/RealESRGAN_x4plus.pth
curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRNet_x4plus.pth -o experiments/pretrained_models/RealESRNet_x4plus.pth
curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth -o experiments/pretrained_models/RealESRGAN_x4plus_netD.pth

## Dataset prep

### Generate meta_info
python scripts/video_to_frame.py -i inputs/VID_20240813_073429.mp4 -o datasets/custom_tune/hr -n 50
python scripts/generate_meta_info.py --input datasets/custom_tune/hr --root datasets/custom_tune --meta_info datasets/custom_tune/meta_info/meta_info_custom_tune.txt

```
