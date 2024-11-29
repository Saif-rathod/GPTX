#Fully trained on Nvidia Cuda GPU.

import torch
import torch.nn as nn
from torch.nn import functional as F

train_test_split = 0.9 #huge hyperparameters
batch_size = 32
block_size = 8
max_iters  = 3000
eval_interval = 300
learning_rate = 1e-2

device = 'cuda' if torch.cuda.is_available() else 'cpu' #trained on Cuda at my uni lab
eval_iters = 200

torch.manual_seed(1337) #reproducibility

with open('tiny-shakespeare.txt', 'r') as f:
    text = f.read() 

chars = sorted(list(set(text))) #unique characters
vocab_size = len(chars)

#mapping char to idx , idx to char
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[ch] for ch in s] #takes string, return a list of indices     
decode = lambda l: ''.join([itos[i] for i in l]) #takes list of indices, return string

data = torch.tensor(encode(text), dtype=torch.long).to(device)
n = int(train_test_split * len(data)) 
train_data = data[:n] 
val_data = data[n:]

def get_batch(split, batch_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) #tensor of shape (batch_size), with random sequence start indices b/w 0 and len(data) - block_size
    x = torch.stack([data[i:i+block_size] for i in ix]) #stack all sequences of this batch row-wise oon top of each other to form a tensor
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #same as x but shifted by 1 token
    x, y = x.to(device), y.to(device)
    return x, y  

@torch.no_grad() 
def evaluate_loss():
    out = {}
    model.eval() 
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() 
    return out


class BigramLM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, vocab_size)  #embedding the vocab 

    def forward(self, idx, targets=None): #(B,T,C)
        logits = self.embed(idx)                               
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #transpose logits to B,C,T                  
            targets = targets.view(B*T)  #transpose targets to B,T               
            loss = F.cross_entropy(logits, targets) #loss calc             
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)                              
            logits = logits[:, -1, :]                          
            probs = F.softmax(logits, dim=-1)                  
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1)            
        return idx  

model = BigramLM(vocab_size) #Model
m = model.to(device)         


opt = torch.optim.AdamW(model.parameters(), lr=learning_rate) #Optimizer


for iter in range(max_iters): #Training
    xb, yb = get_batch('train', batch_size) #Get batch
    logits, loss = m(xb, yb) #forward pass          
    loss.backward() #backward pass                  
    opt.step() #update params                 
    opt.zero_grad(set_to_none=True) #gradients reset 

    if iter % eval_interval == 0:
        losses = evaluate_loss()
        print(f'Iter {iter:4d} | Train Loss {losses["train"]:6.4f} | Val Loss {losses["val"]:6.4f}')

#text gen thru model
context = torch.zeros((1, 1), dtype=torch.long, device=device) #
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))