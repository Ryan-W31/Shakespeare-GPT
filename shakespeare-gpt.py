# shakespeare-gpt.py -- Shakespeare-GPT -- Ryan-W31
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

print("\rHyperparameters: X", end="", flush=True)
# hyperparameters
# -----------------------------------------------------
batch_size = 64 # how many blocks to process in parallel
block_size = 256 # size of each block (context)
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
# -----------------------------------------------------
print("\rHyperparameters: OK")

print("Input Initialization: X", end="", flush=True)
torch.manual_seed(314159265)

# read in input
with open('./input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# create an alphabet of unique characters based on the input
chars = sorted(list(set(text)))
vocab_size = len(chars)

# tokenize the alphabet (character tokenizer)
# mapping for encoding (char to int) and decoding (int to char)
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s] # string to list of ints
decode = lambda l: ''.join([itos[i] for i in l]) # list of ints to string

# splitting the dataset into a train (90%) set and a validation (10%) set
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train = data[:n]
val = data[n:]

def get_batch(split):
    # create a batch of inputs (x) and targets (y)
    data = train if split == 'train' else val # choose split to use
    idx = torch.randint(len(data) - block_size, (batch_size,)) # get index of random token
    
    # get inputs and targets based in index
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])

    x, y = x.to(device), y.to(device)
    return x, y

# decorator to tell PyTorch to not keep track of these operations in its computational graph
@torch.no_grad() 
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
print("\rInput Initialization: OK")

print("CausalAttention Module: X", end="", flush=True)
# multiple heads of self-attention in parallel
class CausalAttention(nn.Module):
    # initialize module
    def __init__(self, num_heads, head_size, n_embed):
        super().__init__()
        assert n_embed % num_heads == 0
        self.attn = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size).view(1, 1, block_size, block_size)))

    # forward pass
    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.attn(x).split(n_embed, dim=2)

        k = k.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, T, C, H) where H is number of heads
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, T, C, H) where H is number of heads
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, T, C, H) where H is number of heads


        attention = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1))) 
        attention = attention.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf')) # (B, T, T)
        attention = F.softmax(attention, dim=-1) # (B, T, T)
        attention = self.attention_dropout(attention)

        out = attention @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.residual_dropout(self.proj(out))
        return out
print("\rCausalAttention Module: OK")

print("FeedForward Module: X", end="", flush=True)
# simple linear layer followed by non-linear activation function
class FeedForward(nn.Module):
    # initialize module
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), # projection layer
            nn.Dropout(dropout),
        )

    # forward pass
    def forward(self, x):
        return self.net(x)
print("\rFeedForward Module: OK")

print("Block Module: X", end="", flush=True)
class Block(nn.Module):
    # initialize module
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = CausalAttention(n_head, head_size, n_embed)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed) # layer norm
        self.ln2 = nn.LayerNorm(n_embed) # layer norm

    # forward pass
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
print("\rBlock Module: OK")

print("Bigram Module: X", end="", flush=True)
# simple bigram model
class Bigram(nn.Module):
    # initialize the module
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # create an 2D embedding table
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)]) # all Block modules
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
    # forward pass
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx and targets are a both (B, T) tensors of integers where B = batch and T = time (or block)
        token_emb = self.token_embedding_table(idx) # (B, T, C) | C = channels (or vocab_size)
        position_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_emb + position_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            # dimension manipulation
            B, T, C = logits.shape
            logits = logits.view((B*T, C))
            targets = targets.view((B*T))
            loss = F.cross_entropy(logits, targets) # get loss using cross_entropy
        return logits, loss
    
    # generate new tokens
    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) tensor
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # crop idx to the last block_size
            logits, loss = self(idx_cond) # get predictions
            logits = logits[:, -1, :] # focus on last time step (B, C)
            probs = F.softmax(logits, dim=1) # get probabilities over rows (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # sample from probs distribution (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # concatenate new token (B, T+1)
        
        return idx
print("\rBigram Module: OK")

print("Model Initialization: X", end="", flush=True)
model = Bigram()
m = model.to(device)
print("\rModel Initialization: OK")


print("Optimizer Initialization: X", end="", flush=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print("\rOptimizer Initialization: OK")

print(f"\n{sum(p.numel() for p in m.parameters()) / 1e6 : .2f} M parameters\n")

print("Iterations Starting now...")
for i in range(max_iters):
    if i % 10 == 0: # every 500 iterations record the mean loss
        print(f"\rIteration: {i:4d}")
    #     losses = estimate_loss()
    #     print(f"Step {i:4d}: TRAIN loss {losses['train']:.4f} VAL loss {losses['val']:.4f}")

    Xb, Yb = get_batch('train')

    # evaluate loss
    logits, loss = m(Xb, Yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# lets generate tokens from the trained Bigram
context = torch.zeros((1,1), dtype=torch.long, device=device) # get index (first token is index 0 or '/n')
out = m.generate(context, max_new_tokens=10000)[0].tolist() # generate the new tokens

# decode the tokens
output = open("output.txt", "w")
output.write(decode(out))
output.close()
    