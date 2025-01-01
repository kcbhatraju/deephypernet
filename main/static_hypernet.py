import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm
import numpy as np
import torch.func
from math import ceil
from itertools import chain

from hypernet_lib import BaseNet, SharedEmbeddingHyperNet, StaticSharedEmbedding

class Lowest(BaseNet):
    def create_params(self):
        self.weight_generator = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(np.prod(self.input_dim), 10),
        )

class OneUp(SharedEmbeddingHyperNet):
    def create_params(self):
        class WeightGenerator(nn.Module):
            def __init__(self, num_params_to_estimate, num_embeddings, embedding_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(embedding_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, ceil(num_params_to_estimate / num_embeddings)),
                )
            
            def forward(self, x):
                return self.net(x)
        
        self.weight_generator = WeightGenerator(self.num_params_to_estimate, self.num_embeddings, *self.embedding_dim)

class TwoUp(SharedEmbeddingHyperNet):
    def create_params(self):
        class WeightGenerator(nn.Module):
            def __init__(self, num_params_to_estimate, num_embeddings, embedding_dim):
                super().__init__()

                self.net = nn.Sequential(
                    nn.Linear(embedding_dim, 16),
                    nn.ReLU(),
                    nn.Linear(16, ceil(num_params_to_estimate / num_embeddings)),
                )
            
            def forward(self, x):
                return self.net(x)
        
        self.weight_generator = WeightGenerator(self.num_params_to_estimate, self.num_embeddings, *self.embedding_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

base = Lowest(num_backward_connections=2).to(device)
one = OneUp(base, num_backward_connections=2).to(device)
two = TwoUp(one, num_backward_connections=2).to(device)

embed = StaticSharedEmbedding(two, 512, 4, (64, 1, 28, 28)).to(device)

morph = transforms.Compose([transforms.ToTensor()])

train_set = torchvision.datasets.MNIST(root="data/", train=True, transform=morph, download=True)
test_set = torchvision.datasets.MNIST(root="data/", train=False, transform=morph, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

print(f"Base network parameters: {base.num_weight_gen_params}")
print(f"One hypernetwork parameters: {one.num_weight_gen_params}")
print(f"Two hypernetwork parameters: {two.num_weight_gen_params}")
print(f"Embed hypernetwork parameters: {embed.num_weight_gen_params}")

num_epochs = 25
tq = tqdm(range(num_epochs))

# Train the hypernetwork
optimizer = optim.AdamW(embed.parameters(), lr=1e-2, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

for ep in tq:
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()

        logits = embed.embed_and_propagate(x_batch)
        
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()

        optimizer.step()

    scheduler.step()
    tq.set_postfix(loss=loss.item())

# Evaluate the hypernetwork
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        outputs = embed.embed_and_propagate(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

hypernet_acc = correct / total
print(f"Nested hypernet accuracy: {hypernet_acc:.4f}")

base = nn.Sequential(
    nn.Flatten(),
    nn.ReLU(),
    nn.Linear(784, 10),
).to(device)

# Train the base network for comparison
optimizer = optim.AdamW(base.parameters(), lr=1e-4, weight_decay=1e-3)

tq = tqdm(range(num_epochs))

for ep in tq:
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()

        logits = base(x_batch)
        
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()

        optimizer.step()

    tq.set_postfix(loss=loss.item())

# Evaluate the base network
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = base(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

base_acc = correct / total
print(f"Base network accuracy: {base_acc:.4f}")

print(f"Improvement: {(hypernet_acc - base_acc) * 100:.2f}%")
