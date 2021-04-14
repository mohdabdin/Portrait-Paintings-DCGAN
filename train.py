import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from networks.DCGAN_model_64 import Discriminator, Generator, initialize_weights
import random
import os
import natsort
from PIL import Image, ImageOps, ImageEnhance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D_LEARNING_RATE = 2e-4
G_LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NOISE_DIM = 128
NUM_EPOCHS = 100
FEATURES_DISC = 64
FEATURES_GEN = 64

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert('RGB')        
        tensor_image = self.transform(image)
        
        return tensor_image

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)), 
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)



dataset = CustomDataSet("./data/128_portraits/", transform=transforms)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weights(gen)
initialize_weights(disc)

### uncomment to work from saved models ###

#gen.load_state_dict(torch.load('saved_models/generator_model.pt'))
#disc.load_state_dict(torch.load('saved_models/discriminator_model.pt')) 

opt_gen = optim.Adam(gen.parameters(), lr=G_LEARNING_RATE, betas=(0.5, 0.99))
opt_disc = optim.Adam(disc.parameters(), lr=D_LEARNING_RATE, betas=(0.5, 0.99))
criterion = nn.MSELoss()

fixed_noise = torch.randn(16, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, real in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)

        ### Train Discriminator
        disc_real = disc(real).reshape(-1)

        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator using feature matching
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        

        #checkpoint to save models, print losses, and write to tensorboard 
        if batch_idx % 10 == 0:                
            torch.save(gen.state_dict(), 'generator_model.pt')
            torch.save(disc.state_dict(), 'discriminator_model.pt')
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_real = torchvision.utils.make_grid(
                    real[:16], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:16], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
