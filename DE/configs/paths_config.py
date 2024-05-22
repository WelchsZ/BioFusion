## Pretrained model paths
e4e = './pretrained_models/e4e_ffhq_encode.pt'
stylegan2_ada_ffhq = '../pretrained_models/ffhq.pkl'
style_clip_pretrained_mappers = ''
ir_se50 = './pretrained_models/model_ir_se50.pth'
dlib = './pretrained_models/align.dat'
stylegan2_thermos = '/media/data/zym/bio/pretrained_models/stylegan2_30000.pt'
stylegan_thermos = '/media/data/zym/bio/StyleGAN.pytorch-master/checkpoints/ckp_car_512/model/GAN_GEN_7_40.pth'

## Dirs for output files
checkpoints_dir = '/media/data/zym/bio/StyleGAN.pytorch-master/PTI/checkpoints/mix_car'
embedding_base_dir = '/media/data/zym/bio/StyleGAN.pytorch-master/PTI/embeddings/mix_car'
styleclip_output_dir = '/media/data/zym/bio/StyleGAN.pytorch-master/PTI/StyleCLIP_results'
experiments_output_dir = '/media/data/zym/bio/StyleGAN.pytorch-master/PTI/output'

## Input info
### Input dir, where the images reside
input_data_path = '/media/data/zym/bio/StyleGAN.pytorch-master/PTI/mix_car'
### Inversion identifier, used to keeping track of the inversion results. Both the latent code and the generator
input_data_id = 'car'

## Keywords
pti_results_keyword = 'PTI'
e4e_results_keyword = 'e4e'
sg2_results_keyword = 'SG2'
sg2_plus_results_keyword = 'SG2_plus'
multi_id_model_type = 'multi_id'

## Edit directions
interfacegan_age = 'editings/interfacegan_directions/age.pt'
interfacegan_smile = 'editings/interfacegan_directions/smile.pt'
interfacegan_rotation = 'editings/interfacegan_directions/rotation.pt'
ffhq_pca = 'editings/ganspace_pca/ffhq_pca.pt'
