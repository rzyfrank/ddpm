import copy

from torch.optim import Adam
import torch
from diffusion_lib import Unet
from torchvision.utils import save_image
from diffusion_lib import Diffusion
from diffusion_lib import choose_beta_schedule
from diffusion_lib import obtain_args
from diffusion_lib import prepare_logger
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from diffusion_lib import EMA
from diffusion_lib import cycle




def main(args):
    logger = prepare_logger(args)

    assert torch.cuda.is_available(), 'CUDA is not available!'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log('device:{:}'.format(device))

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    train_dataset = datasets.CIFAR10(
        root='./cifar10/cifar10_train',
        train=True,
        download=True,
        transform=train_transform
    )
    train_loader = cycle(DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    ))

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    test_dataset = datasets.CIFAR10(
        root='./cifar10/cifar10_test',
        train=False,
        download=True,
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    reverse_transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        # transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        # transforms.Lambda(lambda t: t.numpy().astype(np.uint8))
    ])

    model = Unet(
        dim=args.dim,
        channels=args.channels,
        dim_mults=(1, 2, 4, 8)
    )
    model.to(device)

    ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    if args.checkpoint is not None:
        last_info = torch.load(args.checkpoint)
        model.load_state_dict(last_info['diffusion_lib'])
        ema_model.load_state_dict(last_info['ema_model'])
        optimizer.load_state_dict(last_info['optimizer'])
        steps = last_info['step'] + 1
        logger.log('load pretrain diffusion_lib from {:}'.format(args.checkpoint))
    else:
        steps = 0

    ema = EMA(0.9999, steps)

    logger.log('optimizer:{:}'.format(optimizer))

    beta_schedule = choose_beta_schedule(args.beta_schedule)

    diffusion = Diffusion(args.timesteps, beta_schedule)

    assert steps is not args.steps, 'step is the same {:} vs {:}'.format(steps, args.steps)
    for step in range(steps, args.steps):
        img, _ = next(train_loader)
        model.train()
        optimizer.zero_grad()

        batch_size = img.shape[0]
        img = img.to(device)

        t = torch.randint(0, args.timesteps - 1, (batch_size,), device=device).long()

        loss = diffusion.p_losses(model, img, t, loss_type='l2')

        if step % args.print_freq == 0:
            logger.log('Step-{:}:Train Loss:{:}'.format(step, loss.item()))

        loss.backward()
        optimizer.step()
        ema.step_ema(ema_model, model)

        if step != 0 and step % args.eval_freq == 0:
            test_loss = 0
            with torch.no_grad():
                model.eval()
                for img,_ in test_loader:
                    img = img.to(device)
                    t = torch.randint(0, args.timesteps - 1, (batch_size,), device=device).long()
                    loss = diffusion.p_losses(model, img, t, loss_type='l2')
                    test_loss += loss.item()

                test_loss /= len(test_loader)
                logger.log('Test Loss:{:}'.format(test_loss))

            images = diffusion.sample(model, image_size=args.image_size, batch_size=1, channels=args.channels)[-1]
            images = ((images + 1) / 2).clip(0, 1)
            images_ema = diffusion.sample(ema_model, image_size=args.image_size, batch_size=1, channels=args.channels)[-1]
            images_ema = ((images_ema + 1) / 2).clip(0, 1)
            all_images = torch.cat((images, images_ema), dim=0)
            save_image(all_images, logger.path('sample') / 'epoch-{:}.png'.format(step), nrow=1)

        if step % args.save_freq == 0:
            save_info = {
                'diffusion_lib': model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }
            torch.save(save_info, logger.path('diffusion_lib') / 'seed-{:}-{:}.pth'.format(args.rand_seed, logger.get_time()))



if __name__ == '__main__':
    args = obtain_args()
    main(args)
