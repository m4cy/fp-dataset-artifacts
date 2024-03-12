import torch
import torch.nn as nn
import json
import pandas as pd
import datasets
import numpy as np
import tqdm

class BiasedModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.ff1 = nn.Linear(input_size, hidden_size)
        self.ff2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ff1(x)
        x = self.sigmoid(x)
        return self.ff2(x)

    

model = BiasedModel(input_size=1024, hidden_size=2048, output_size=3).to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def open_vocab():
    f = open("bag.txt", "r")
    vocab = f.read()
    vocab = vocab.split(' ')
    vocab = vocab[0:len(vocab) - 1]
    return vocab
    
def vectorify(vocab, example):
    example = example.lower().split(' ')
    input = [1 if word in example else 0 for word in vocab]
    return torch.tensor(input, dtype=torch.float64)

vocab = open_vocab()
indexed = {word: i for i, word in enumerate(vocab)}
def make_bow(example):
    bow = np.zeros(len(indexed))
    for word in example['hypothesis']:
        if word in indexed:
            bow[indexed[word]] = 1
    example['hypothesis'] = bow
    return example



class BatchDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        dataset = datasets.load_dataset("snli")
        dataset = dataset.map(make_bow)
        self.dataset = dataset
        self.split = split
    
    def __len__(self):
        return len(self.dataset[self.split])
    
    def __getitem__(self, idx: int):
        point = self.dataset[self.split][idx]
        target = [0] * 3
        target[point['label']] = 1
        return torch.tensor(point['hypothesis']).to('cuda'), torch.tensor(target, dtype=torch.float64).to('cuda')

dataset = BatchDataset('train')
train_loader =  torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
def train_biased(model, train_loader, criterion, optimizer):
    for epoch in range(3):
        epoch_loss = 0.0
        index = 0
        for X, y in tqdm.tqdm(train_loader):
            index += 1
            # input = torch.tensor(train_set[i]['hypothesis'])
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print('epoch: ', epoch, 'loss: ', epoch_loss)


# lines = []
# with open(r'./eval_output/eval_predictions.jsonl') as f:
#     lines = f.read().splitlines()
# line_dicts = [json.loads(line) for line in lines]
# train_set = pd.DataFrame(line_dicts)
# train_set = train_set.to_dict('records') 


# train_set = [{'hypothesis': 'The sisters are hugging goodbye while holding to go packages after just eating lunch.', 'label': 1}]
# train_biased(model, train_loader, criterion, optimizer)
# torch.jit.save(torch.jit.script(model), "biased_model.pt")
BiasedModel = torch.load('./biased_model.pt')
def predict(model, test_example):
    with torch.no_grad():
        input = torch.tensor(make_bow(test_example)['hypothesis']).to('cuda')
        output = model.forward(input)
        return torch.argmax(output)

# print(predict(model, {'hypothesis': 'Two woman are holding packages.'}))


test_set = BatchDataset('test')
test_loader =  torch.utils.data.DataLoader(test_set, batch_size=1024, shuffle=True)
correct = 0
total = 0
with torch.no_grad():   
    for X, y in tqdm.tqdm(test_loader):
        print(X.shape)
        print(y.shape)
        BiasedModel.eval()
        output = BiasedModel.forward(X)
        print(output.shape)
        if (torch.argmax(output) == torch.argmax(y)):
            correct += 1
        total += 1

print(correct/total)