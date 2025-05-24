import os
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import torch.nn as nn
import torch.optim as optim


class AmbientDataset(Dataset):
    def __init__(self, folder, sample_rate=16000, duration=2):
        print(folder)

        print(glob(os.path.join(folder, "*.wav")))

        print(os.curdir)

        #    Check if directory exists
        print(f"Directory exists: {os.path.exists(folder)}")

        # List all files in the directory
        print("Files in directory:", os.listdir(folder))
        abs_path = os.path.abspath(folder)

        self.paths = glob(os.path.join(folder, "*.wav"))
        self.sample_rate = sample_rate
        self.samples = int(sample_rate * duration)
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.paths[idx])
        waveform = waveform.mean(dim=0)  # mono
        waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        waveform = waveform[:self.samples]  # truncate/pad
        if waveform.shape[0] < self.samples:
            padding = self.samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform

duration = 8  # seconds
sample_rate = 16000
output_size = int(sample_rate * duration)

dataset = AmbientDataset("training-samples/split", sample_rate=sample_rate, duration=duration)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_size=output_size):  # 2s at 16kHz
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            nn.Tanh()  # audio output range
        )
    
    def forward(self, z):
        return self.net(z)
    
class Discriminator(nn.Module):
    def __init__(self, input_size=output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)
    
G = Generator()
D = Discriminator()
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

for epoch in range(100):
    for real in loader:
        real = real.to(torch.float32)
        real = real.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        G.to(real.device)
        D.to(real.device)

        batch_size = real.size(0)
        z = torch.randn(batch_size, 100).to(real.device)
        fake = G(z)

        real_labels = torch.ones(batch_size, 1).to(real.device)
        fake_labels = torch.zeros(batch_size, 1).to(real.device)

        # Train Discriminator
        D_real = D(real)
        D_fake = D(fake.detach())
        loss_D = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        D_fake = D(fake)
        loss_G = criterion(D_fake, real_labels)
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1} - Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

G.eval()
z = torch.randn(1, 100)
sample = G(z).detach().cpu().numpy().squeeze()
torchaudio.save("generated.wav", torch.tensor(sample).unsqueeze(0), sample_rate)      