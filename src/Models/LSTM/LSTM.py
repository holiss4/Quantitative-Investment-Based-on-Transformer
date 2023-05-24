from torch import nn

class LSTM(nn.Module):
    """LSTM"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(LSTM, self).__init__()
        self.tslayer = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, X):
        y = self.tslayer(X)[0][:, -1, :]
        y = self.linear(y)
        return y