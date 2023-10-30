import torch
import numpy as np
from langdiff65 import loss_test, noise, train_net

positions = np.load('data_np.npy', allow_pickle = True)
generated_positions = positions[:,0][:]
bones_tensor =torch.tensor([[0,1],[1,2],[1,3],[3,4],[4,5],[5,6],[1,7],[7,8],[8,9],[9,10],[0,11],[11,12],[12,13],[13,14],[0,15],[15,16],[16,17],[17,18]])

dataset = torch.tensor(generated_positions, dtype = torch.float64)
random_ts = torch.rand(dataset.size()[0])

noised_positions = noise(dataset[0:2], random_ts[0:2],bones_tensor)