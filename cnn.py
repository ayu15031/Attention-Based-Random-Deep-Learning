import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class CNN(nn.Module):
    """ Class for CNN """
    def __init__(self, input_size, emb_dim, output_size, filter_sizes, emb_weights=None):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(input_size, emb_dim)
        
        if emb_weights is not None:
          self.embedding.weight=nn.Parameter(torch.tensor(emb_weights,dtype=torch.float32))
          self.embedding.weight.requires_grad=False
        
        self.n_layers = len(filter_sizes)

        layers = []
        
        in_channels = emb_dim
        
        for i in range(self.n_layers):
          layers.append(nn.Conv1d(in_channels=in_channels,
                          out_channels=filter_sizes[i], kernel_size=5, padding=2))
          layers.append(nn.ReLU())
          layers.append(nn.BatchNorm1d(filter_sizes[i]))
          layers.append(nn.Dropout(0.5))
          if i < self.n_layers-1:
            layers.append(nn.MaxPool1d(3))
          
          in_channels = filter_sizes[i]
        
        layers.append(nn.AdaptiveMaxPool1d(20))

        self.conv_layers = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(20*filter_sizes[-1], 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)

        self.att = nn.Linear(256, output_size)
    
    def forward(self, x):
        """ Function to implement forward pass """
        batch_size = x.size(0)
        x = self.embedding(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)
        
        y = self.fc2(x)
        y = self.bn2(y)
        a = self.att(x)

        return a, F.log_softmax(y)


def build_random_cnn(min_hidden_layer_cnn, max_hidden_layer_cnn, min_nodes_cnn, max_nodes_cnn, input_size, emb_dim, output_size, emb_weights):
    """ Function to build a random CNN """
    values = list(range(min_nodes_cnn,max_nodes_cnn))
    values_layer = list(range(min_hidden_layer_cnn,max_hidden_layer_cnn))

    
    hidden_layers = np.random.choice(values_layer)

    filter_sizes = []
    for i in range(hidden_layers):
      filter_sizes.append(np.random.choice(values))

    # Build model
    model = CNN(input_size, emb_dim, output_size, filter_sizes, emb_weights)

    return model
