import pickle
import functools
import torch
from PTI.configs import paths_config, global_config
# from styleGAN2_model import Generator
from models.GAN import Generator


def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def load_tuned_G(run_id, type):
    new_G_path = f'{paths_config.checkpoints_dir}/model_{run_id}_{type}.pt'
    with open(new_G_path, 'rb') as f:
        new_G = torch.load(f).to(global_config.device).eval()
    new_G = new_G.float()
    toogle_grad(new_G, False)
    return new_G


def load_old_G():

    # # styleGAN2
    # old_G = Generator(256, 512, 8)

    # styleGAN
    from config import cfg as opt

    cfgfile = "./configs/car_512.yaml"
    opt.merge_from_file(cfgfile)
    opt.freeze()

    old_G = Generator(resolution=opt.dataset.resolution,
                    num_channels=opt.dataset.channels,
                    structure=opt.structure,
                    **opt.model.gen)

    with open(paths_config.stylegan_thermos, 'rb') as f:
        # old_G = pickle.load(f)['G_ema'].to(global_config.device).eval()
        # old_G.load_state_dict(torch.load(f)["g_ema"], strict=False)  # stylegan2
        old_G.load_state_dict(torch.load(f))  # stylegan
        old_G.eval()
        old_G = old_G.to(global_config.device)
        old_G = old_G.float()
    return old_G
