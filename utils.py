
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
import requests
import zipfile
from tqdm.auto import tqdm
import wandb

def show(tensor, num=25, wandbactive=0, name=''):
    data = tensor.detach().cpu()
    grid = make_grid(data[:num], nrow=5).permute(1,2,0)
    plt.figure(figsize=(16,16))
    if wandbactive == 1:
        wandb.log({name: wandb.Image(grid.numpy().clip(0,1))})
    plt.imshow(grid.clip(0,1))
    plt.axis('off')
    plt.show()

def gen_noise(num, z_dim, device):
    return torch.randn((num, z_dim, 1, 1), device=device)

def get_gp(crit, real, fake, alpha, gamma=10):
    mix_images = real * alpha + fake * (1 - alpha)
    mix_scores = crit(mix_images)

    grad = torch.autograd.grad(
        inputs=mix_images,
        outputs=mix_scores,
        grad_outputs=torch.ones_like(mix_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    grad = grad.view(grad.shape[0], -1)
    norm = torch.sqrt(torch.sum(grad**2, dim=1))
    gp = gamma * torch.mean((norm-1)**2)
    return gp

root_path = './checkpoints'
if not os.path.exists(root_path):
    os.mkdir(root_path)

def save_checkpoint(gen, crit, gen_opt, crit_opt, epoch):
    torch.save({
        'generator': gen.state_dict(),
        'critic': crit.state_dict(),
        'gen_optimizer': gen_opt.state_dict(),
        'crit_optimizer': crit_opt.state_dict(),
        'epoch': epoch
    }, os.path.join(root_path, f'checkpoint_epoch_{epoch}.pth'))

def load_checkpoint(gen, crit, gen_opt, crit_opt):
    checkpoint = torch.load(os.path.join(root_path, 'latest_checkpoint.pth'))
    gen.load_state_dict(checkpoint['generator'])
    crit.load_state_dict(checkpoint['critic'])
    gen_opt.load_state_dict(checkpoint['gen_optimizer'])
    crit_opt.load_state_dict(checkpoint['crit_optimizer'])
    return checkpoint['epoch']

def download_and_extract_celeba(url, target_path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Download the file
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(target_path, 'wb') as file, tqdm(
        desc=target_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)
    
    # Extract the file
    with zipfile.ZipFile(target_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(target_path))
