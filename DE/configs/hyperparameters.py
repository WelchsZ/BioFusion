## Architechture
lpips_type = 'vgg'
first_inv_type = 'w'
optim_type = 'adam'

## Locality regularization
latent_ball_num_of_samples = 10
locality_regularization_interval = 1
use_locality_regularization = True
regulizer_l2_lambda = 0.3
regulizer_lpips_lambda = 0.2
regulizer_alpha = 30

## Loss
pt_l2_lambda = 0.6
pt_lpips_lambda = 1
reg_l2_lambda = 1


## Steps
LPIPS_value_threshold = 0.06
max_pti_steps = 350
first_inv_steps = 450
max_images_to_invert = 30

## Optimization
pti_learning_rate = 3e-3
first_inv_lr = 5e-3
train_batch_size = 1
use_last_w_pivots = True
