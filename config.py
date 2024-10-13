
import torch

config = {
    'n_epoch': 100,
    'batch_size': 128,
    'lr': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'z_dim': 200,
    'crit_cycles': 5,
    'show_steps': 35,
    'save_steps': 35,
    'wandbact': 0,
    'data_path': '/content/img_align_celeba',
    'image_size': 128,
    'dataset_limit': 10000
}
