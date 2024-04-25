import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
# from customDataset import SignLanguageDataset

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channel = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 32
num_epochs = 1

# Load Data
transform = transforms.ToTensor()
train_data = datasets.OxfordIIITPet(root='../Data', train=True, download=True,
                                    transform=transform)
test_data = datasets.OxfordIIITPet(root='../Data', train=False, download=True,
                                   transform=transform)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)
# dataset = SignLanguageDataset(csv_file='asl_alphabet.csv',
#                               root_dir='asl_alphabet',
#                               transform=transforms.ToTensor())

# train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000])
# train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
#                           shuffle=True)
# test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Model
model = torchvision.models.googlenet(pretrained=True)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backwards()

        # gradient descent or adam step
        optimizer.step()

    print(f'Cost at epoch {epoch} is {sum(losses)/len(losses)}')


# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}')


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuarcy on Test Set")
check_accuracy(test_loader, model)
