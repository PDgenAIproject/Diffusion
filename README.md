# keratopathyAI_project

## Generation part

### Requirements

create environment

```bash
conda env create -f environment.yaml
conda activate ldm
```

GPU setting
```python
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

cache clear (optional)

```python
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
```

###### Compression using AutoencoderKL

slit image

```bash
CUDA_LAUNCH_BLOCKING=1 python main.py --base /camin1/chlee/latent-diffusion-keratopathy/configs/autoencoder/autoencoder_kl_L_slit.yaml -t --accelerator ddp --gpus 0,1
```

slit-beam image

```bash
CUDA_LAUNCH_BLOCKING=1 python main.py --base /camin1/chlee/latent-diffusion-keratopathy/configs/autoencoder/autoencoder_kl_L_slit.yaml -t --accelerator ddp --gpus 0,1
```

performance check

```bash
tensorboard --logdir /camin1/chlee/latent-diffusion-keratopathy/logs/L_slit/L_slit/ver1
```

###### LDM using AutoencoderKL

```bash
pip install -U huggingface_hub
huggingface-cli login   # 웹에서 발급받은 토큰 입력(모델 약관 동의 필요)
```

```bash
CUDA_LAUNCH_BLOCKING=1 python main.py --base /camin1/chlee/latent-diffusion-keratopathy/configs/latent-diffusion/L_slit-to-beam-ldm.yaml -t --accelerator ddp --gpus 0,1
```

###### image-to-image translation using DDIMSampler

```bash
python /camin1/chlee/latent-diffusion-keratopathy/scripts/img2img.py --prompt "slit-beam" --init_img /camin1/chlee/latent-diffusion-keratopathy/outputs/in/17172038_20220530_Lt_SLIT-1_64crop.jpg --outdir /camin1/chlee/latent-diffusion-keratopathy/outputs --config /camin1/chlee/latent-diffusion-keratopathy/configs/latent-diffusion/L_slit-to-beam-ldm.yaml --ckpt /camin1/chlee/latent-diffusion-keratopathy/logs/ldm_L_slit/ver1/epoch=99.ckpt --n_samples 1 --ddim_steps 100 --strength 0.5 --scale 9.0
```

###### Foundation model

```python
model = YOLO("yolov8x-cls.pt")
model.info()
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
results = model("/path/to/bus.jpg")
```
