import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter

from dataset import get_data_loader
from model import Discriminator, Generator, initialize_weights
from tqdm import tqdm
# Hyper parameters etc.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 500
FEATURES_DISC = 64
FEATURES_GEN = 64


def train(dir_path: str, checkpoint_dir: str, load_path: str = None):
    # Get the dataloader
    loader = get_data_loader(batch_size=BATCH_SIZE, shuffle=True, num_workers=2, dir_path=dir_path)

    # Get the Generator and Discriminator and initialize the weights.
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
    initialize_weights(gen, load_path)
    initialize_weights(disc, load_path)

    # optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # criterion
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
    writer_real = SummaryWriter(f'logs/real')
    writer_fake = SummaryWriter(f'logs/fake')

    gen.train()
    disc.train()
    step = 0
    # The training loop!
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, _) in tqdm(enumerate(loader)):
            real = real.to(device)
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)

            # Train the discriminator
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

            # Train the generator
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Print the losses and print to tensorboard
            if batch_idx % 100 == 0:
                print(
                    f"EPOCH [{epoch}/{NUM_EPOCHS}] BATCH {batch_idx}/{len(loader)}/"
                    f" Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f} "
                )

                with torch.no_grad():
                    fake = gen(fixed_noise)
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True
                    )
                    writer_real.add_image('Real', img_grid_real, global_step=step)
                    writer_fake.add_image('Fake', img_grid_fake, global_step=step)

                    step += 1

        # Save a checkpoint
        if epoch % 10 == 0:
            torch.save(gen.state_dict(), f"{checkpoint_dir}/gen_{epoch}.pth")
            torch.save(disc.state_dict(), f"{checkpoint_dir}/disc_{epoch}.pth")
