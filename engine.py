
import torch
from tqdm.auto import tqdm
import wandb
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from utils import show, gen_noise, get_gp, save_checkpoint

def train(dataloader, gen, crit, gen_opt, crit_opt, config):
    device = config['device']
    z_dim = config['z_dim']
    n_epoch = config['n_epoch']
    crit_cycles = config['crit_cycles']
    show_steps = config['show_steps']
    save_steps = config['save_steps']
    wandbact = config['wandbact']

    cur_step = 0
    gen_losses = []
    crit_losses = []

    for epoch in tqdm(range(n_epoch), desc="Epochs"):
        for real, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epoch}", leave=False):
            cur_batch_size = len(real)
            real = real.to(device)

            mean_crit_loss = 0
            for _ in range(crit_cycles):
                crit_opt.zero_grad()
                fake_noise = gen_noise(cur_batch_size, z_dim, device)
                fake = gen(fake_noise)
                crit_fake_pred = crit(fake.detach())
                crit_real_pred = crit(real)

                alpha = torch.rand((cur_batch_size,1,1,1), device=device, requires_grad=True)
                gp = get_gp(crit, real, fake.detach(), alpha)

                crit_loss = crit_fake_pred.mean() - crit_real_pred.mean() + gp

                mean_crit_loss += crit_loss.item() / crit_cycles
                crit_loss.backward(retain_graph=True)
                crit_opt.step()
            crit_losses.append(mean_crit_loss)

            gen_opt.zero_grad()
            fake_noise = gen_noise(cur_batch_size, z_dim, device)
            fake = gen(fake_noise)
            crit_fake_pred = crit(fake)
            gen_loss = -crit_fake_pred.mean()
            gen_loss.backward()
            gen_opt.step()
            gen_losses.append(gen_loss.item())

            if wandbact == 1:
                wandb.log({'gen_loss': gen_loss.item(), 'crit_loss': crit_loss.item()})

            if cur_step % save_steps == 0 and cur_step > 0:
                print(f"Saving checkpoint to wandb at step {cur_step}")
                save_checkpoint(gen, crit, gen_opt, crit_opt, epoch)

            if cur_step % show_steps == 0 and cur_step > 0:
                fake_grid = make_grid(fake[:25].detach().cpu(), nrow=5).permute(1, 2, 0)
                real_grid = make_grid(real[:25].detach().cpu(), nrow=5).permute(1, 2, 0)

                if wandbact == 1:
                    wandb.log({
                        "fake_images": wandb.Image(fake_grid.numpy()),
                        "real_images": wandb.Image(real_grid.numpy())
                    })

                show(fake, num=25, wandbactive=0, name='fake')
                show(real, num=25, wandbactive=0, name='real')

                gen_mean = sum(gen_losses[-show_steps:]) / show_steps
                crit_mean = sum(crit_losses[-show_steps:]) / show_steps
                print(f"epoch: {epoch}, step: {cur_step}, gen_loss: {gen_mean:.4f}, crit_loss: {crit_mean:.4f}")

                plt.figure(figsize=(10, 5))
                plt.plot(range(len(gen_losses)), gen_losses, label='gen')
                plt.plot(range(len(crit_losses)), crit_losses, label='crit')
                plt.legend()
                plt.title("Generator and Critic Losses")
                plt.xlabel("Steps")
                plt.ylabel("Loss")

                if wandbact == 1:
                    wandb.log({"loss_plot": wandb.Image(plt)})

                plt.show()

            cur_step += 1

        if wandbact == 1:
            wandb.log({"epoch": epoch})
