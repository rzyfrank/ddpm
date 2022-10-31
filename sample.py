import numpy as np
from diffusion_lib import Diffusion
from diffusion_lib.beta_schedule import linear_beta_schedule
import torch
from diffusion_lib import Unet
from torchvision import transforms


def main(image_size, channels, timesteps, pretrain):
    diffusion = Diffusion(timesteps, linear_beta_schedule)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Unet(
        dim=64,
        channels=3,
        dim_mults=(1, 2, 4, 8)
    )
    model.to(device)
    last_info = torch.load(pretrain)
    model.load_state_dict(last_info['ema_model'])

    print('load pretrain diffusion_lib from {:}'.format(pretrain))

    # print(i for i in range(0,5,200))


    samples = diffusion.sample(model, image_size, batch_size=1, channels=channels)
    random_index = 5

    reverse_transform = transforms.Compose([
        transforms.Lambda(lambda t: (t+1)/2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),

    ])
    img = reverse_transform(samples[-1][0])
    img.show()


    # plt.imshow(samples[-1][0].reshape(image_size, image_size, channels))
    # plt.show()

    # samples = samples[0]
    # samples = ((samples + 1) / 2).clip(0, 1) * 255
    # save_image(samples, './image.jpg')

if __name__ == '__main__':
    pretrain = 'D:\coding\diffusion\scripts\ddpm\cifar10-b64-dim64\checkpoint\seed-1821-30-Oct-at-17-35-49.pth'
    main(32, 3, 1000, pretrain)
