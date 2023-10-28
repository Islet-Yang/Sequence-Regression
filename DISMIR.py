import torch.nn as nn
import torch.nn.functional as F

class DISMIR(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sequence_length):
        '''
        Initialize the model
        Build the framework
        Define parameters
        '''
        super(DISMIR, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=10, padding=5)
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.2)
        self.lstm = nn.LSTM(sequence_length//2, hidden_size, bidirectional=True, batch_first=True)
        self.conv2 = nn.Conv1d(hidden_size*2, hidden_size, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size * hidden_size // 2, 750)
        self.drop3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(750, 300)
        self.fc3 = nn.Linear(300, output_size)

    def forward(self, x):
        '''
        Forward Output
        '''
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, input_size, sequence_length)        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)        
        x, _ = self.lstm(x)       
        x = x.permute(0, 2, 1)  # Change shape back to (batch_size, sequence_length, hidden_size*2)        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

