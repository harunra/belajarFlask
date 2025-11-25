import torch
import torch.nn as nn
import torch.nn.functional as F
from demonn import net

model = torch.load('testnn.pt')
model.eval()