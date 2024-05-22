import os
import argparse
import numpy as np
from PIL import Image

import torch
import matplotlib.pyplot as plt
from models.GAN import Generator
from generate_grid import adjust_dynamic_range
from projector import projector

device = torch.device('cuda')

def draw_style_mixing_figure(png, gen, out_depth, src_seeds, dst_seeds, style_ranges):
    n_col = len(src_seeds)
    n_row = len(dst_seeds)
    w = h = 2 ** (out_depth + 2)
    with torch.no_grad():
        latent_size = gen.g_mapping.latent_size
        src_latents_np = np.stack([np.random.RandomState(seed).randn(latent_size, ) for seed in src_seeds])
        dst_latents_np = np.stack([np.random.RandomState(seed).randn(latent_size, ) for seed in dst_seeds])
        src_latents = torch.from_numpy(src_latents_np.astype(np.float32)).to(device)
        dst_latents = torch.from_numpy(dst_latents_np.astype(np.float32)).to(device)
        src_dlatents = gen.g_mapping(src_latents)  # [seed, layer, component]
        dst_dlatents = gen.g_mapping(dst_latents)  # [seed, layer, component]
        src_images: object = gen.g_synthesis(src_dlatents, depth=out_depth, alpha=1)
        dst_images = gen.g_synthesis(dst_dlatents, depth=out_depth, alpha=1)

        src_dlatents_np = src_dlatents.cpu().numpy()
        dst_dlatents_np = dst_dlatents.cpu().numpy()
        canvas = Image.new('RGB', (w * (n_col + 1), h * (n_row + 1)), 'white')
        for col, src_image in enumerate(list(src_images)):
            src_image = adjust_dynamic_range(src_image)
            src_image = src_image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            canvas.paste(Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
        for row, dst_image in enumerate(list(dst_images)):
            dst_image = adjust_dynamic_range(dst_image)
            dst_image = dst_image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            canvas.paste(Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))

            row_dlatents = np.stack([dst_dlatents_np[row]] * n_col)
            row_dlatents[:, style_ranges[row]] = src_dlatents_np[:, style_ranges[row]]
            row_dlatents = torch.from_numpy(row_dlatents).to(device)

            row_images = gen.g_synthesis(row_dlatents, depth=out_depth, alpha=1)
            for col, image in enumerate(list(row_images)):
                image = adjust_dynamic_range(image)
                image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                canvas.paste(Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
        canvas.save(png)

def real_image_mixing(out_filename, gen, out_depth, src_imgs, dst_imgs, style_ranges):
    n_col = len(src_imgs)
    n_row = len(dst_imgs)
    w = h = 2 ** (out_depth + 2)
    latent_size = gen.g_mapping.latent_size

    src_data = [projector(gen, img, 20, num_steps=1500) for img in src_imgs]
    src_dlatents, src_images = zip(*src_data)

    # dst_data = [projector(gen, img, 20, num_steps=1000) for img in dst_imgs]
    dst_data = [projector(gen, dst_imgs[0], 20, num_steps=2000)] * 3
    dst_dlatents, dst_images = zip(*dst_data)

    with torch.no_grad():

        src_dlatents_np = np.stack(src_dlatents)
        src_dlatents_np = np.squeeze(src_dlatents_np, axis=1)
        dst_dlatents_np = np.stack(dst_dlatents)
        dst_dlatents_np = np.squeeze(dst_dlatents_np, axis=1)

        canvas = Image.new('RGB', (w * (n_col + 1), h * (n_row + 1)), 'white')
        for col, src_image in enumerate(list(src_images)):
            canvas.paste(Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
        for row, dst_image in enumerate(list(dst_images)):
            canvas.paste(Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))

            row_dlatents = np.stack([dst_dlatents_np[row]] * n_col)
            row_dlatents[:, style_ranges[row]] = src_dlatents_np[:, style_ranges[row]]
            row_dlatents = torch.from_numpy(row_dlatents).to(device)

            row_images = gen.g_synthesis(row_dlatents, depth=out_depth, alpha=1)
            for col, image in enumerate(list(row_images)):
                image = adjust_dynamic_range(image)
                image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                canvas.paste(Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
        canvas.save(out_filename)


def interpolation_mixing(out_filename, gen, out_depth, src_imgs, dst_imgs, style_ranges, psi):
    n_col = len(src_imgs)
    n_row = len(dst_imgs)
    w = h = 2 ** (out_depth + 2)
    latent_size = gen.g_mapping.latent_size

    src_data = [projector(gen, img, 20, num_steps=1800) for img in src_imgs]
    src_dlatents, src_images = zip(*src_data)

    # dst_data = [projector(gen, img, 20, num_steps=1000) for img in dst_imgs]
    dst_data = [projector(gen, dst_imgs[0], 20, num_steps=1800)] * n_row
    dst_dlatents, dst_images = zip(*dst_data)

    with torch.no_grad():

        src_dlatents_np = np.stack(src_dlatents)
        src_dlatents_np = np.squeeze(src_dlatents_np, axis=1)
        dst_dlatents_np = np.stack(dst_dlatents)
        dst_dlatents_np = np.squeeze(dst_dlatents_np, axis=1)

        canvas = Image.new('RGB', (w * (n_col + 1), h * (n_row + 1)), 'white')
        for col, src_image in enumerate(list(src_images)):
            canvas.paste(Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
        for row, dst_image in enumerate(list(dst_images)):
            canvas.paste(Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))

            row_dlatents = np.stack([dst_dlatents_np[row]] * n_col)
            # inter mixing
            row_dlatents[:, style_ranges[row]] = psi[row] * src_dlatents_np[:, style_ranges[row]] + \
                                                 (1 - psi[row]) * row_dlatents[:, style_ranges[row]]
            row_dlatents = torch.from_numpy(row_dlatents).to(device)

            row_images = gen.g_synthesis(row_dlatents, depth=out_depth, alpha=1)
            for col, image in enumerate(list(row_images)):
                image = adjust_dynamic_range(image)
                image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                canvas.paste(Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
        canvas.save(out_filename)


def interpolation_mixing_split(out_path, out_name, gen, out_depth, src_img, dst_img, style_range, psis):
    n_col = 3
    w = h = 2 ** (out_depth + 2)
    latent_size = gen.g_mapping.latent_size

    src_dlatents, src_images = projector(gen, src_img, 20, num_steps=1800)

    dst_data = [projector(gen, dst_img, 20, num_steps=1800)] * n_col
    dst_dlatents, dst_images = zip(*dst_data)

    with torch.no_grad():

        src_dlatents_np = np.stack(src_dlatents)
        # src_dlatents_np = np.squeeze(src_dlatents_np, axis=1)
        dst_dlatents_np = np.stack(dst_dlatents)
        dst_dlatents_np = np.squeeze(dst_dlatents_np, axis=1)

        for psi in psis:
            for row, dst_image in enumerate(list(dst_images)):
                row_dlatents = np.stack([dst_dlatents_np[row]])
                # inter mixing
                row_dlatents[:, style_range[row]] = psi * src_dlatents_np[:, style_range[row]] + \
                                                    (1 - psi) * row_dlatents[:, style_range[row]]
                row_dlatents = torch.from_numpy(row_dlatents).to(device)

                row_images = gen.g_synthesis(row_dlatents, depth=out_depth, alpha=1)
                for col, image in enumerate(list(row_images)):
                    image = adjust_dynamic_range(image)
                    image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                    image_pil = Image.fromarray(image, 'RGB')
                    outpath = f"{out_path}/row_{row}/psi_{psi:.1f}"
                    os.makedirs(outpath, exist_ok=True)
                    image_pil.save(f"{outpath}/{out_name}.png")


def main(args):
    from config import cfg as opt

    opt.merge_from_file(args.config)
    opt.freeze()

    print("Creating generator object ...")
    # create the generator object
    gen = Generator(resolution=opt.dataset.resolution,
                    num_channels=opt.dataset.channels,
                    structure=opt.structure,
                    **opt.model.gen)

    print("Loading the generator weights from:", args.generator_file)
    # load the weights into it
    gen.load_state_dict(torch.load(args.generator_file))

    src_imgs = ["./sharkcar/fish/dolp03-k3.jpg",
                "./sharkcar/fish/dolp05-k3.jpg",
                "./sharkcar/fish/whale01-k3.jpg"]

    dst_img = "../data/car_512/06b8153205f0c2.png"

    dst_imgs = [dst_img] * 3

    interpolation_mixing('./style_mixing/car/bio_06b8153205f0c2_(0713)_[461]_mix4.png', gen, out_depth=7,
                      src_imgs=src_imgs, dst_imgs=dst_imgs,
                      style_ranges=[range(0, 6)] * 1 + [range(6, 12)] * 1 + [range(12, 16)] * 1,
                      psi=[0.4, 0.5, 1])

    print('Done.')


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./configs/car_512.yaml')
    parser.add_argument("--generator_file", action="store", type=str,
                        help="pretrained weights file for generator",
                        default='./mix_car/model_few-car_multi_id.pt')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_arguments())
