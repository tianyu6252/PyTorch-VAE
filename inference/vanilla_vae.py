import os
import sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(path)
sys.path.append(path)


from models.vanilla_vae import VanillaVAE
from experiment import VAEXperiment
import pytorch_lightning as pl
import torch
import torchvision.utils as vutils

if __name__ == "__main__":

    save_img_path = "inference/result/vanilla_vae"
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('current_device:', current_device)


    # 加载模型
    model = VanillaVAE(in_channels=3, latent_dim=128)
    model = model.to(current_device)

    # 加载检查点
    checkpoint_path = "logs/VanillaVAE/version_8/checkpoints/epoch=1-step=5087.ckpt"
    # checkpoint_path = "logs/VanillaVAE/version_8/checkpoints/last.ckpt"
    experiment = VAEXperiment.load_from_checkpoint(checkpoint_path, vae_model=model, params={})

    # 采样
    num_samples = 16  # 生成 16 个样本
    samples = experiment.model.sample(num_samples, current_device)

    # 保存或显示生成的图像
    vutils.save_image(samples, os.path.join(save_img_path, "samples.png"), normalize=True, nrow=4)


    exit()
    # 加载输入图像
    from torchvision import transforms
    from PIL import Image

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载图像
    input_image = Image.open("input_image.png").convert("RGB")
    input_tensor = transform(input_image).unsqueeze(0).to(current_device)

    # 重建
    reconstructed_image = checkpoint.generate(input_tensor)

    # 保存或显示重建的图像
    vutils.save_image(reconstructed_image, "reconstructed.png", normalize=True)