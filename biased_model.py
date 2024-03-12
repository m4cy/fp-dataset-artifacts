import torch
import torch.nn as nn
import json
import pandas as pd

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

    

model = BiasedModel(input_size=32, hidden_size=64, output_size=3).double()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# I have 1024 words for vocab
# taking in hypothesis as data, I turn it into a vector. 1 if vocab word is present, 0 if not
# using that vector, I predict if it is entailment, neutral, or contradiction


def open_vocab():
    f = open("bag.txt", "r")
    vocab = f.read()
    vocab = vocab.split(' ')
    vocab = vocab[0:len(vocab) - 1]
    return vocab
    
def vectorify(vocab, example):
    example = example.lower().split(' ')
    input = [1 if word in vocab else 0 for word in example] + ([0] * 32)
    return torch.tensor(input[0:32], dtype=torch.float64)

def train_biased(model, train_set, criterion, optimizer):
    vocab = open_vocab()
    for epoch in range(100):
        epoch_loss = 0.0
        for i in range(len(train_set)):
            input = vectorify(vocab, train_set[i]['hypothesis'])
            optimizer.zero_grad()
            outputs = model(input)
            target = [0] * 2
            target.insert(train_set[i]['label'], 1)
            loss = criterion(outputs, torch.tensor(target, dtype=torch.float64))
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print('epoch: ', epoch, 'loss: ', epoch_loss)


lines = []
with open(r'./eval_output/eval_predictions.jsonl') as f:
    lines = f.read().splitlines()
line_dicts = [json.loads(line) for line in lines]
train_set = pd.DataFrame(line_dicts)
train_set = train_set.to_dict('records') 


# train_set = [{'hypothesis': 'The sisters are hugging goodbye while holding to go packages after just eating lunch.', 'label': 1}]
train_biased(model, train_set, criterion, optimizer)

def predict(model, test_example):
    vocab = open_vocab()
    with torch.no_grad():
        input = vectorify(vocab, test_example)
        print(input)
        output = model.forward(input)
        return torch.argmax(output)

print(predict(model, 'Two woman are holding packages.'))
torch.jit.save(torch.jit.script(model), "biased_model.pt")