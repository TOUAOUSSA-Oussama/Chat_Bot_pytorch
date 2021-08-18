import json
import torch
import numpy as np
import torch.nn as nn

from model import NeuralNet
from torch.utils.data import DataLoader, Dataset
from nltk_utils import tokenize, stem, bag_of_words

# read the json file :
path = 'intents.json'
with open(path, 'r') as f:
    intents = json.load(f)

# preprocessing :
all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w) # extend et pas append pour ne pas avoir une liste de liste
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
Y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
X_train = torch.from_numpy(X_train)

Y_train = np.array(Y_train)
Y_train = torch.from_numpy(Y_train)
Y_train = Y_train.type(torch.LongTensor)

# Create a pytorch data :
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train

        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Hyperparameters :
batch_size = 8
learning_rate = 0.001
num_epochs = 1000
input_size = len(all_words)
hidden_size = 8
num_classes = len(tags)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# GPU device :
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model :
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# loss and optimizer :
criterion  = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# training loop :
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # forward pass:
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward pass :
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f"epoch : {epoch+1}/{num_epochs}, loss : {loss.item():.4f}")

# final loss :
print(f"Final loss : {loss.item():.4f}")

# save the model and data :
data = {
    "model_state": model.state_dict(),
    "input_size" : input_size,
    "hidden_size": hidden_size,
    "output_size": num_classes,
    "all_words": all_words,
    "tags": tags
}

path = 'data.pth'
torch.save(data, path)
print(f'End, the data saved in {path}')
