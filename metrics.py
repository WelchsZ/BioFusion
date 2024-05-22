import os
import PIL
from skimage import io, transform
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import numpy as np
import torch.nn.functional as F
import torch
import dnnlib
from pytorch_fid import fid_score
from utils import kid

device = torch.device('cuda')

def cal_lpips(original_path, generated_path):
    # Load VGG16 feature detector.
    vgg16_dir = r'../pretrained_models/vgg16.pt'
    with dnnlib.util.open_url(vgg16_dir) as f:
        vgg16 = torch.jit.load(f).eval().to(device)


    original_pil = PIL.Image.open(original_path).convert('RGB')
    w, h = original_pil.size
    s = min(w, h)
    original_pil = original_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    original_pil = original_pil.resize((256, 256), PIL.Image.LANCZOS)
    original_uint8 = np.array(original_pil, dtype=np.uint8)

    img1 = torch.tensor(original_uint8.transpose([2, 0, 1]), device=device)
    img1 = img1.unsqueeze(0).to(device).to(torch.float32)
    if img1.shape[2] > 256:
        img1 = F.interpolate(img1, size=(256, 256), mode='area')
    img1_feature = vgg16(img1, resize_images=False, return_lpips=True)

    generate_pil = PIL.Image.open(generated_path).convert('RGB')
    w, h = generate_pil.size
    s = min(w, h)
    generate_pil = generate_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    generate_pil = generate_pil.resize((256, 256), PIL.Image.LANCZOS)
    generate_uint8 = np.array(generate_pil, dtype=np.uint8)

    img2 = torch.tensor(generate_uint8.transpose([2, 0, 1]), device=device)
    img2 = img2.unsqueeze(0).to(device).to(torch.float32)
    if img2.shape[2] > 256:
        img2 = F.interpolate(img2, size=(256, 256), mode='area')
    img2_feature = vgg16(img2, resize_images=False, return_lpips=True)

    dist = (img1_feature - img2_feature).square().sum()

    dist = dist.cpu().numpy()

    return dist


def cal_fid(path_to_real_images, path_to_generated_images):
    fid_value = fid_score.calculate_fid_given_paths([path_to_real_images, path_to_generated_images], batch_size=50,
                                                    device='cuda', dims=2048, num_workers=0)
    return fid_value


def cal_kid(path_to_real_images, path_to_generated_images):
    kid_score = kid.calculate_kid_given_paths([path_to_real_images, path_to_generated_images], batch_size=50,
                                              cuda=0, dims=2048)

    return kid_score


def calculate_inversion_metrics(original_folder, generated_folder):
    metrics = []

    # scan all files in the original folder
    for filename in os.listdir(original_folder):
        original_path = os.path.join(original_folder, filename)
        generated_path = os.path.join(generated_folder, filename)

        # check if the corresponding generated image exists
        if os.path.exists(generated_path):
            # read the original and generated images
            original_image = io.imread(original_path)
            generated_image = io.imread(generated_path)

            original_image = transform.resize(original_image, (256, 256))
            generated_image = transform.resize(generated_image, (256, 256))

            # calculate PSNR, SSIM, and MSE
            psnr_value = psnr(original_image, generated_image, data_range=255)
            ssim_value = ssim(original_image, generated_image, multichannel=True)
            mse_value = mse(original_image, generated_image)
            lpips_value = cal_lpips(original_path, generated_path)
            metrics.append((filename, psnr_value, ssim_value, mse_value, lpips_value))

    return metrics

def calculate_average(metrics):

    # ensure that the list of metrics is not empty
    if not metrics:
        return 0, 0, 0, 0

    # convert the list of metrics to a NumPy array
    metrics_array = np.array([list(metric[1:]) for metric in metrics])
    # metrics_array = np.array([metric.cpu().numpy()[1:] for metric in metrics])

    # calculate the average PSNR, SSIM, and MSE
    average_psnr = np.mean(metrics_array[:, 0])
    average_ssim = np.mean(metrics_array[:, 1])
    average_mse = np.mean(metrics_array[:, 2])
    average_lpips = np.mean(metrics_array[:, 3])

    return average_psnr, average_ssim, average_mse, average_lpips


def cal_BIQI(real_car_path, real_fish_path, generated_path):

    # calculate FPI
    Iphi_list = load_images_from_folder(generated_path)

    psis = ['psi_0.0', 'psi_0.1', 'psi_0.2', 'psi_0.3', 'psi_0.4', 'psi_0.5', 'psi_0.6', 'psi_0.7', 'psi_0.8', 'psi_0.9', 'psi_1.0']
    biqis = []
    for psi in psis:
        num = 0
        fpi_sum = 0
        max_sum = 0
        src_sum = 0
        bio_sum = 0
        for Iphi_path in Iphi_list:
            Iphi_name = os.path.basename(Iphi_path)
            Iphi_dir_name = os.path.basename(os.path.dirname(Iphi_path))
            if Iphi_dir_name == psi and num <= 150:
                parts = Iphi_name.split('_')
                Ibio_name, Isrc_name = parts[0], parts[1]
                Ibio_img_path = os.path.join(real_fish_path, Ibio_name + '.jpg')
                Isrc_img_path = os.path.join(real_car_path, Isrc_name + '.png')

                try:
                    if os.path.exists(Ibio_img_path) and os.path.exists(Isrc_img_path):
                        lpips_Isrc_Iphi = cal_lpips(Isrc_img_path, Iphi_path)
                        lpips_Iphi_Ibio = cal_lpips(Iphi_path, Ibio_img_path)
                        lpips_Isrc_Ibio = cal_lpips(Isrc_img_path, Ibio_img_path)

                        fpi = 2*lpips_Isrc_Iphi*lpips_Iphi_Ibio/(lpips_Isrc_Iphi + lpips_Iphi_Ibio)  # 调和平均

                        fpi_sum += fpi
                        src_sum += lpips_Isrc_Iphi
                        bio_sum += lpips_Iphi_Ibio
                        max_sum += lpips_Isrc_Ibio

                        num += 1
                        src_avg = src_sum/num
                        bio_avg = bio_sum/num
                        max_avg = max_sum/num

                        print("src: "+str(src_avg)+"  bio: "+str(bio_avg)+"  max: "+str(max_avg))
                        print(str(num)+"  "+Iphi_dir_name+Iphi_name+": "+str(fpi)+"     fpi_avg: "+str(fpi_sum/num))
                except FileNotFoundError as e:
                    print(f"File not found: {e}")
                    continue

        fid_src = cal_fid(real_car_path, generated_path+"/"+psi)
        print("fid: "+str(fid_src))
        biqi = (1 / num) * fpi_sum + fid_src

        biqis.append(biqi)

    return biqis


def load_images_from_folder(folder):
    images = []
    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)
                images.append(img_path)
    return images

