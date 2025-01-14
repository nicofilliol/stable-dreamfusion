from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

import time
import os

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusion(nn.Module):
    def __init__(self, device, visualize=True, out_folder="visualizations/"):
        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '') # remove the last \n!
                print(f'[INFO] loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            print(f'[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')
        
        self.device = device
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.visualize = visualize
        self.out_folder = out_folder
        self.dirs = ['front', 'left_side', 'back', 'right_side', 'overhead', 'bottom']
        self.last_update = {name : -1000 for name in self.dirs}

        if self.visualize:
            for d in self.dirs:
                if not os.path.exists(os.path.join(self.out_folder, f"{d}/nerf")): os.makedirs(os.path.join(self.out_folder, f"{d}/nerf"))
                if not os.path.exists(os.path.join(self.out_folder, f"{d}/noisy")): os.makedirs(os.path.join(self.out_folder, f"{d}/noisy"))
                if not os.path.exists(os.path.join(self.out_folder, f"{d}/noisy_pred")): os.makedirs(os.path.join(self.out_folder, f"{d}/noisy_pred"))
                if not os.path.exists(os.path.join(self.out_folder, f"{d}/final_denoised")): os.makedirs(os.path.join(self.out_folder, f"{d}/final_denoised"))
                if not os.path.exists(os.path.join(self.out_folder, f"{d}/denoised")): os.makedirs(os.path.join(self.out_folder, f"{d}/denoised"))
                if not os.path.exists(os.path.join(self.out_folder, f"{d}/residual")): os.makedirs(os.path.join(self.out_folder, f"{d}/residual"))
                if not os.path.exists(os.path.join(self.out_folder, f"{d}/noise")): os.makedirs(os.path.join(self.out_folder, f"{d}/noise"))
                if not os.path.exists(os.path.join(self.out_folder, f"{d}/pred_noise")): os.makedirs(os.path.join(self.out_folder, f"{d}/pred_noise"))
                if not os.path.exists(os.path.join(self.out_folder, f"{d}/residual_noise")): os.makedirs(os.path.join(self.out_folder, f"{d}/residual_noise"))

        print(f'[INFO] loading stable diffusion...')
                
        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", use_auth_token=self.token).to(self.device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", use_auth_token=self.token).to(self.device)

        # 4. Create a scheduler for inference
        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=self.num_train_timesteps)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    # TODO: Store visualizations of NeRF output, noise and residual
    def train_step(self, text_embeddings, pred_rgb, iteration, d, guidance_scale=100):
        # Convert d into text
        assert len(d) == 1, "Received more than one direction!"
        d = self.dirs[d]

        # Visualize step (want to approximately store every tenth image for each direction)
        if self.visualize and abs(iteration-self.last_update[d]) >= 10:
            visualize = True
            self.last_update[d] = iteration
        else:
            visualize = False

        # interp to 512x512 to be fed into vae.

        # _t = time.time()
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)

        # Store predicted (by NeRF) image
        if visualize:
            save_image(pred_rgb_512, os.path.join(self.out_folder, f"{d}/nerf/{iteration}.png"))

        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        # _t = time.time()
        latents = self.encode_imgs(pred_rgb_512)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # Store image corresponding to noisy latents
            if visualize:
                noisy_image = self.decode_latents(latents_noisy)
                save_image(noisy_image, os.path.join(self.out_folder, f"{d}/noisy/{iteration}.png"))

                noise_image = self.decode_latents(noise)
                save_image(noise_image, os.path.join(self.out_folder, f"{d}/noise/{iteration}.png"))

            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Compute previous noisy sample based on predicted noise by diffusion model
        with torch.no_grad():
            if visualize:
                # Denoised Image
                prev_latents = self.get_previous_sample(latents, t, noise_pred)
                prev_image = self.decode_latents(prev_latents)
                save_image(prev_image, os.path.join(self.out_folder, f"{d}/denoised/{iteration}.png"))

                # Completely Denoised Image
                num_inf_steps = self.scheduler.num_inference_steps
                final_latents = self.produce_latents(text_embeddings, t, num_inference_steps=25, guidance_scale=guidance_scale, latents=latents)
                self.scheduler.set_timesteps(num_inf_steps)
                final_image = self.decode_latents(final_latents)
                save_image(final_image, os.path.join(self.out_folder, f"{d}/final_denoised/{iteration}.png"))

                # Noisy Image using Predicted Noise
                pred_noisy_latents = self.scheduler.add_noise(latents, noise_pred, t)
                pred_noisy_image = self.decode_latents(pred_noisy_latents)
                save_image(pred_noisy_image, os.path.join(self.out_folder, f"{d}/noisy_pred/{iteration}.png"))

                # Image with residual noise applied
                residual_noise = noise_pred-noise
                res_latents = self.scheduler.add_noise(latents, residual_noise, t)
                residual_image = self.decode_latents(res_latents)
                save_image(residual_image, os.path.join(self.out_folder, f"{d}/residual/{iteration}.png"))

                # Predicted Noise
                pred_noise_image = self.decode_latents(noise_pred)
                save_image(pred_noise_image, os.path.join(self.out_folder, f"{d}/pred_noise/{iteration}.png"))

                # Residual Noise
                res_noise_image = self.decode_latents(residual_noise)
                save_image(res_noise_image, os.path.join(self.out_folder, f"{d}/residual_noise/{iteration}.png"))

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        grad = torch.nan_to_num(grad)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()
        latents.backward(gradient=grad, retain_graph=True)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        return 0 # dummy loss value

    def get_previous_sample(self, sample, timestep, noise_pred):
        return self.scheduler._get_prev_sample(sample, timestep, timestep-1, noise_pred)


    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                torch.cuda.empty_cache()
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cpu')

    sd = StableDiffusion(device)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()




