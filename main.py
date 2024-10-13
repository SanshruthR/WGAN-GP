
import torch
from torch.utils.data import DataLoader
import wandb
from model import Generator, Critic
from dataset import CelebaDataset
from engine import train
from config import config
from utils import load_checkpoint,download_and_extract_celeba
import os 
import requests
import zipfile

def main():
    ##############
    # Check if the dataset exists, if not, download it
    if not os.path.exists(config['data_path']):
        print("CelebA dataset not found. Downloading...")
        celeba_url = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"
        download_and_extract_celeba(celeba_url, "/content/celeba.zip")
        print("Download complete. Dataset extracted.")
  
    # Initialize wandb
    if config['wandbact'] == 1:
        wandb.login(key="<ENTER YOUR KEY HERE>")
        experiment_name = wandb.util.generate_id()
        wandb.init(project='wgan',
                   group=experiment_name,
                   config={
                       "optimizer": "adam",
                       "model": "wgan gp",
                       "epoch": config['n_epoch'],
                       "lr": config['lr'],
                       "batch_size": config['batch_size']
                   })

    # Initialize dataset and dataloader
    ds = CelebaDataset(config['data_path'], size=config['image_size'], lim=config['dataset_limit'])
    dataloader = DataLoader(ds, batch_size=config['batch_size'], shuffle=True)

    # Initialize models
    gen = Generator(config['z_dim']).to(config['device'])
    crit = Critic().to(config['device'])

    # Initialize optimizers
    gen_opt = torch.optim.Adam(gen.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    crit_opt = torch.optim.Adam(crit.parameters(), lr=config['lr'], betas=(0.5, 0.999))

    # Optionally load checkpoint
    # start_epoch = load_checkpoint(gen, crit, gen_opt, crit_opt)
    # if start_epoch > 0:
    #     print(f"Resuming training from epoch {start_epoch}")

    # Watch models with wandb
    if config['wandbact'] == 1:
        wandb.watch(gen, log_freq=100)
        wandb.watch(crit, log_freq=100)

    # Train the model
    train(dataloader, gen, crit, gen_opt, crit_opt, config)

if __name__ == "__main__":
    main()
