import argparse
import copy
import os
import shutil
import numpy as np
import PIL
from time import perf_counter
import imageio

import torch
import torch.nn.functional as F
from models.GAN import Generator

import click
import dnnlib


def parse_arguments():
    """
    default command line argument parser
    :return: args => parsed command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./configs/car_512.yaml')
    parser.add_argument("--generator_file", action="store", type=str,
                        help="pretrained weights file for generator", default='./checkpoints/ckp_car_512/model/GAN_GEN_7_120.pth')  # required=True

    args = parser.parse_args()

    return args

# w_plus
def project(
    gen,
    target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 2000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    device: torch.device
):
    from config import cfg as opt

    cfgfile = "./configs/car_512.yaml"
    opt.merge_from_file(cfgfile)
    opt.freeze()

    assert target.shape == (3, opt.dataset.resolution, opt.dataset.resolution)

    gen = copy.deepcopy(gen).eval().requires_grad_(False).to(device)

    # Compute w stats.
    print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    latent_size = gen.g_mapping.latent_size

    z_samples = np.stack(np.random.RandomState(123).randn(w_avg_samples, latent_size))
    z_samples = torch.from_numpy(z_samples.astype(np.float32)).to(device)
    w_samples = gen.g_mapping(z_samples).detach().cpu().numpy()
    w_avg = np.mean(w_samples, axis=0, keepdims=True)
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Load VGG16 feature detector.
    vgg16_dir = r'../stylegan2-ada-pytorch-main/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(vgg16_dir) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Setup noise inputs.
    noise_bufs = torch.load('./project_imgs/noise_bufs.pth')


    # Features for target image
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)  # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise)  #.repeat([1, 14, 1]), [1, gen.g_mapping.num_ws, 1]
        synth_images = gen.g_synthesis(ws, depth=6, alpha=1)   # const_input_layer=True

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum() # Perceptual loss with L2

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    del gen
    return w_out


@click.command()
@click.option('--network', 'network_pth', help='Network path filename', default='./checkpoints/ckp_thermos_256_resume/model/GAN_GEN_7_120.pth')
@click.option('--target', 'target_fname', help='Target image file to project to', default="../data/thermos/0020.jpg", metavar='FILE')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=2500, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', default="./project_imgs", metavar='DIR')
def run_projection(
    network_pth: str,
    target_fname: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    from config import cfg as opt

    cfgfile = "./configs/thermos_256.yaml"
    opt.merge_from_file(cfgfile)
    opt.freeze()

    # Load networks.
    print('Loading networks from "%s"...' % network_pth)
    device = torch.device('cuda')
    print("Creating generator object ...")
    # create the generator object
    gen = Generator(resolution=opt.dataset.resolution,
                    num_channels=opt.dataset.channels,
                    structure=opt.structure,
                    **opt.model.gen)

    print("Loading the generator weights from:", network_pth)
    # load the weights into it
    gen.load_state_dict(torch.load(network_pth))
    gen = gen.to(device)

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((opt.dataset.resolution, opt.dataset.resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        gen,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps:
            synth_image = gen.g_synthesis(projected_w.unsqueeze(0), depth=6, alpha=1)
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    synth_image = gen.g_synthesis(projected_w.unsqueeze(0), depth=6, alpha=1)
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())


#----------------------------------------------------------------------------

def projector(gen, target_fname, seed, num_steps):
    np.random.seed(seed)
    torch.manual_seed(seed)

    from config import cfg as opt

    cfgfile = "./configs/thermos_256.yaml"
    opt.merge_from_file(cfgfile)
    opt.freeze()

    device = torch.device('cuda')

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((opt.dataset.resolution, opt.dataset.resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        gen,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device),  # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

    projected_w = projected_w_steps[-1]
    synth_image = gen.g_synthesis(projected_w.unsqueeze(0), depth=6, alpha=1)
    synth_image = (synth_image + 1) * (255 / 2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

    return projected_w.unsqueeze(0).cpu().numpy(), synth_image


if __name__ == "__main__":
    run_projection()  # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
