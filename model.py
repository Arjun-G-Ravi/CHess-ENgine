import torch
import torch.nn as nn
from torch.nn import functional as F
# from g_mlp_pytorch import gMLP
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# device = torch.device('cpu')
class Head(nn.Module):

    def __init__(self, context_length, embed_size, head_dim):
        super().__init__()

        self.queries = nn.Linear(embed_size, head_dim, bias = False)
        self.keys = nn.Linear(embed_size, head_dim, bias=False)
        self.values = nn.Linear(embed_size, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

        # self.ln = nn.LayerNorm(head_dim)
        # self.ln2 = nn.LayerNorm(head_dim)
        self.head_dim = head_dim
    def forward(self, X):

        B,T,C = X.shape
        q = self.queries(X)
        k = self.keys(X)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        v = self.values(X)

        out = wei@v
        # print(out.shape)
        return out


class MultiHeadedAttention(nn.Module):

    def __init__(self, num_heads, context_length, embed_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(context_length, embed_size, embed_size//num_heads) for _ in range(num_heads)])
        self.fc = nn.Linear(embed_size,embed_size)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim =-1)
        out = self.fc(out)
        return out

class TLMBlock(nn.Module):

  def __init__(self, num_heads, context_length, embed_size):
    super().__init__()

    self.ln_1 = nn.LayerNorm(embed_size)
    self.sa_head = MultiHeadedAttention(num_heads, context_length,embed_size)
    self.dropout = nn.Dropout(p=0.2)
    self.ln_2 = nn.LayerNorm(embed_size)
    # self.silu = nn.SiLU()
    self.mlp = nn.Sequential(
       nn.Linear(embed_size, 2*embed_size),
       nn.Linear(2*embed_size,embed_size),
       nn.Linear(embed_size,embed_size),
       nn.GELU(),
       nn.Dropout(p=0.1)
    )
  def forward(self, x):

    # B,T = x.shape
    # print(B,T)
    x = x+self.sa_head(self.ln_1(x))
    # print(x.shape)
    x = x + self.mlp(self.ln_2(x))

    # print(x.shape)
    return x

class TLM(nn.Module):

  def __init__(self, vocab_size, context_length, embed_size, num_blocks = 5):
    super().__init__()
    self.token_embeddings = nn.Embedding(vocab_size, embed_size)
    self.positional_embeddings = nn.Embedding(context_length, embed_size)
    self.word_embeddings = nn.Embedding(context_length, embed_size)
    self.context_length = context_length
    self.block = nn.ModuleList()
    for _ in range(num_blocks):
      self.block.append(TLMBlock( 4, context_length,embed_size))

    self.lm_head = nn.Linear(embed_size,vocab_size)
    self.silu = nn.SiLU()

  def forward(self, idx, targets=None):
    B,T = idx.shape
    # print(idx.shape)
    # print(B,T)
    tok_emb = self.token_embeddings(idx)
    # print(tok_emb.shape)
    pos_emb = self.positional_embeddings(torch.arange(T, device=device))
    # word_emb = self.word_embeddings(idx)
    x = tok_emb + pos_emb
    # print(x.shape)
    for _,layer in enumerate(self.block):
      x = layer(x)

    logits = self.lm_head(x)
    if targets == None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T, C)
      # print(logits.shape)
      targets = targets.view(B*T)
      # print(targets.shape)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
      for _ in range(max_new_tokens):

        idx_cond = idx[:,-self.context_length:]

        logits, loss = self(idx_cond)

        logits = logits[:,-1,:]

        probs = F.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim = 1)
        # print(idx.shape)
      return idx

def generate(string):

  for s in string:
    print(s, end="")
    time.sleep(0.05)
if __name__ ==  "__main__":

    tlm = TLM(4,8,32)
    x = torch.ones((8,4),dtype=torch.long)
    out = tlm(x)
    print(out[0].shape)