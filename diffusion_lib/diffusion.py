import torch
from .beta_schedule import linear_beta_schedule
import torch.nn.functional as F
from tqdm.auto import tqdm

class Diffusion():
    def __init__(self,timesteps, beta_schedule):
        self.timesteps = timesteps
        self.betas = beta_schedule(timesteps=timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.sqrt_one_minus_alphas = torch.sqrt(1 - self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_alphas_minus_alphas_cumprod = torch.sqrt(self.alphas - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def extract(self,a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, model, x_start, t, noise=None, loss_type='l2'):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    def p_losses_plus(self, model, x_start, t, loss_type='l2'):
        #随机设定噪声
        noise = torch.randn_like(x_start)


        x_t = self.q_sample(x_start, t, noise)
        x_t_plus_one = self.q_sample(x_start, t+1, noise)

        betas_t_plus_one = self.extract(self.betas, t+1, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t_plus_one = self.extract(self.sqrt_one_minus_alphas_cumprod, t+1, x_start.shape)
        sqrt_recip_alphas_t_plus_one = self.extract(self.sqrt_recip_alphas, t+1, x_start.shape)
        model_mean_t_plus_one = sqrt_recip_alphas_t_plus_one * (
                x_t_plus_one - betas_t_plus_one * model(x_t_plus_one, t+1) / sqrt_one_minus_alphas_cumprod_t_plus_one
        )
        posterior_variance_t_plus_one = self.extract(self.posterior_variance, t+1, x_start.shape)
        x_t_predict = model_mean_t_plus_one + torch.sqrt(posterior_variance_t_plus_one) * noise

        x_t_noise = model(x_t_predict, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, x_t_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, x_t_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, x_t_noise)
        else:
            raise NotImplementedError()

        return loss


    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index ==0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise


    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        device = next(model.parameters()).device

        b = shape[0]
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu())

        return imgs

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
