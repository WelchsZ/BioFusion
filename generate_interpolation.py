import argparse
import numpy as np

import torch
import matplotlib.pyplot as plt

from generate_grid import adjust_dynamic_range
from models.GAN import Generator

from projector import projector

import os

device = torch.device('cuda')

def parse_arguments():
    """
    default command line argument parser
    :return: args => parsed command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./configs/thermos_256.yaml')
    parser.add_argument("--generator_file", action="store", type=str,
                        help="pretrained weights file for generator", default='./checkpoints/mix/model_test-mix-reg2_multi_id.pt')  # required=True

    args = parser.parse_args()

    return args


def random_Gimage_interpolate(out_filename, gen, out_depth, seed_A, seed_B, psi):
    w = h = 2 ** (out_depth + 2)
    latent_size = gen.g_mapping.latent_size

    with torch.no_grad():
        latent_np_A = np.stack(np.random.RandomState(seed_A).randn(latent_size))
        latent_np_B = np.stack(np.random.RandomState(seed_B).randn(latent_size))
        latent_A = torch.from_numpy(latent_np_A.astype(np.float32)).to(device)
        latent_B = torch.from_numpy(latent_np_B.astype(np.float32)).to(device)
        dlatent_A = gen.g_mapping(latent_A.unsqueeze(0)).detach().cpu().numpy()
        dlatent_B = gen.g_mapping(latent_B.unsqueeze(0)).detach().cpu().numpy()

        dlatent_interpolate = (dlatent_B.squeeze()[np.newaxis] - dlatent_A.squeeze()[np.newaxis]) * np.reshape(psi, [-1, 1, 1]) + dlatent_A.squeeze()[np.newaxis]
        dlatent_interpolate = torch.from_numpy(dlatent_interpolate.astype(np.float32)).to(device)
        imgs_interpolate = gen.g_synthesis(dlatent_interpolate, depth=out_depth, alpha=1)

        dpi = 100
        fig = plt.figure(figsize=(w * len(psi) / dpi, h / dpi))

        for col, image in enumerate(list(imgs_interpolate)):
            ax = plt.subplot(1, len(psi), col+1)
            # change dynamic range, convert to NumPy array
            image = adjust_dynamic_range(image)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()

            # show image in subplot
            ax.imshow(image)
            ax.axis('off')

        plt.savefig(out_filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

def real_image_interpolation(out_filename, gen, out_depth, img_A, img_B, psi):
    w = h = 2 ** (out_depth + 2)
    latent_size = gen.g_mapping.latent_size

    if img_A.endswith(("jpg", "png")):
        projected_w_A, projected_img_A = projector(gen, img_A, 20, num_steps=500)
    elif img_A.endswith("npz"):
        img_A = np.load(img_A)
        projected_w_A = img_A['w']
        projected_w_A = torch.from_numpy(projected_w_A)
    elif img_A.endswith("pt"):
        projected_w_A = torch.load(img_A)
        projected_w_A = projected_w_A.cpu().numpy()

    if img_B.endswith(("jpg", "png")):
        projected_w_B, projected_img_B = projector(gen, img_B, 20, num_steps=500)
    elif img_B.endswith("npz"):
        img_B = np.load(img_B)
        projected_w_B = img_B['w']
        projected_w_B = torch.from_numpy(projected_w_B)
    elif img_B.endswith("pt"):
        projected_w_B = torch.load(img_B)
        projected_w_B = projected_w_B.cpu().numpy()


    with (torch.no_grad()):
        dlatent_A = projected_w_A
        dlatent_B = projected_w_B

        dlatent_interpolate = (dlatent_B.squeeze()[np.newaxis] - dlatent_A.squeeze()[np.newaxis])\
                              * np.reshape(psi, [-1, 1, 1]) + dlatent_A.squeeze()[np.newaxis]
        dlatent_interpolate = torch.from_numpy(dlatent_interpolate.astype(np.float32)).to(device)
        imgs_interpolate = gen.g_synthesis(dlatent_interpolate, depth=out_depth, alpha=1)

        dpi = 100
        fig = plt.figure(figsize=(w * len(psi) / dpi, h / dpi))

        for col, image in enumerate(list(imgs_interpolate)):
            ax = plt.subplot(1, len(psi), col+1)

            image = adjust_dynamic_range(image)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()

            ax.imshow(image)
            ax.axis('off')

        plt.savefig(out_filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def real_image_interpolation_split(output_path, out_name, gen, out_depth, img_A, img_B, psi):
    w = h = 2 ** (out_depth + 2)
    latent_size = gen.g_mapping.latent_size

    if img_A.endswith(("jpg", "png")):
        projected_w_A, projected_img_A = projector(gen, img_A, 20, num_steps=1300)
    elif img_A.endswith("npz"):
        img_A = np.load(img_A)
        projected_w_A = img_A['w']
        projected_w_A = torch.from_numpy(projected_w_A)
    elif img_A.endswith("pt"):
        projected_w_A = torch.load(img_A)
        projected_w_A = projected_w_A.cpu().numpy()

    if img_B.endswith(("jpg", "png")):
        projected_w_B, projected_img_B = projector(gen, img_B, 20, num_steps=1100)
    elif img_B.endswith("npz"):
        img_B = np.load(img_B)
        projected_w_B = img_B['w']
        projected_w_B = torch.from_numpy(projected_w_B)
    elif img_B.endswith("pt"):
        projected_w_B = torch.load(img_B)
        projected_w_B = projected_w_B.cpu().numpy()

    with torch.no_grad():
        dlatent_A = projected_w_A
        dlatent_B = projected_w_B

        dlatent_interpolate = (dlatent_B.squeeze()[np.newaxis] - dlatent_A.squeeze()[np.newaxis]) \
                              * np.reshape(psi, [-1, 1, 1]) + dlatent_A.squeeze()[np.newaxis]
        dlatent_interpolate = torch.from_numpy(dlatent_interpolate.astype(np.float32)).to(device)
        imgs_interpolate = gen.g_synthesis(dlatent_interpolate, depth=out_depth, alpha=1)

        for idx, image in enumerate(imgs_interpolate):
            image = adjust_dynamic_range(image)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            outpath = f"{output_path}/psi_{psi[idx]:.1f}"
            os.makedirs(outpath, exist_ok=True)
            output_filename = f"{outpath}/{out_name}.png"
            plt.imsave(output_filename, image)



def main(args):
    device = torch.device('cuda')
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
    gen = gen.to(device)

    img_A = './data/0084.jpg'
    img_B = './data/2.jpg'

    psi = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    real_image_interpolation('./latent_interpolation/test121/orin_0084_bird2-5', gen, out_depth=6,
                              img_A=img_A, img_B=img_B, psi=psi)

    print('Done.')


if __name__ == '__main__':
    main(parse_arguments())