import torch
import torch.nn as nn

optimizers = {'Adam':torch.optim.Adam, 'SGD':torch.optim.SGD, 'AdamW':torch.optim.AdamW}

class Trainer():

  def __init__(self, num_epochs : int, save_step : int, learning_rate = 0.001, optimizer = 'Adam',device = 'cpu'):
    self.num_epochs = num_epochs
    self.optimizer = optimizer
    self.save_step = save_step
    self.device = torch.device(device)
    # print(self.device)
    self.learning_rate = learning_rate

  def train(self,mkdata,m):
    self.optimizer = optimizers[self.optimizer](m.parameters(), lr = self.learning_rate)
    # self.optimizer = torch.optim.Adam(m.parameters(), lr = self.learning_rate)
    c = 5
    for steps in range(self.num_epochs):
      
      xb, yb = mkdata.get_batch()
      xb = xb.to(self.device)
      yb = yb.to(self.device)
      logits, loss = m(xb,yb)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      torch.cuda.empty_cache()
      if steps%self.save_step == 0 and steps != 0:
        print(f"step:{steps} loss:{loss}")
      if c > loss.item():
        c = loss.item()
        torch.save(m.state_dict(), "./model.pt")

    return True
