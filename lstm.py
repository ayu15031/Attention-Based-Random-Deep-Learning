import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import numpy as np

class LSTM(nn.Module):
    """ Class for LSTM module """
    def __init__(self, input_size, emb_dim, output_size, hidden_dim, n_layers, emb_weights=None):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, emb_dim)
        
        if emb_weights is not None:
          self.embedding.weight=nn.Parameter(torch.tensor(emb_weights,dtype=torch.float32))
          self.embedding.weight.requires_grad=False

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=0.5, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim*500, output_size)
        self.att = nn.Linear(hidden_dim*500, output_size)
    
    def forward(self, x):
        """ Function to implement forward pass """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        batch_size = x.size(0)
        x = self.embedding(x)

        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device)
        x = x.squeeze(2)
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)

        y = self.fc(out)
        a = self.att(out)
    
        return a, F.log_softmax(y)


def build_random_lstm(min_hidden_layer_lstm, max_hidden_layer_lstm, min_nodes_lstm, max_nodes_lstm, input_size, emb_dim, output_size, emb_weights):
    """ Function to build a random LSTM """"
    values = list(range(min_nodes_lstm,max_nodes_lstm))
    values_layer = list(range(min_hidden_layer_lstm,max_hidden_layer_lstm))

    hidden_dim = np.random.choice(values)
    hidden_layers = np.random.choice(values_layer)
    
    # Build model
    model = LSTM(input_size, emb_dim, output_size, hidden_dim, hidden_layers, emb_weights)


    return model
