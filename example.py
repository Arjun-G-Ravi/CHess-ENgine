from Data import Tokenizer, PrepData, MakeData
from Trainer import Trainer
from model import TLM
import torch
import json
#  print(dev,device)

with open("Chess_data.json",'r') as f:
    s = f.read()
    dataset = json.loads(s)

data = dataset["data"]
text = ""
for dat in data:
    s = "\n".join(dat)
    text+="\n"+s
    # print(s)

# print(text)

# print(len(data))
# exit(0)
context_length = 127
batch_size = 32
n_embs = 512
mkdata = MakeData(text, context_length, batch_size)
print(mkdata.vocab_size)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(dev)
# device = torch.device('cpu')
m = TLM(mkdata.vocab_size, context_length, n_embs)
m.to(device)

trainer = Trainer(10, 2, device=dev)
trainer.train(mkdata,m)

idxx = torch.zeros((1,1), dtype = torch.long).to(device=device)

# print(decoder(m.generate(idxx, 1000)[0].tolist()))
print(mkdata.decode(m.generate(idxx, 1000)[0].tolist()))