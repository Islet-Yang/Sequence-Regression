import torch.nn as nn
import torch.nn.functional as F

class DenseConv(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sequence_length):
        '''
        Initialize the model
        Build the framework
        Define parameters
        '''
        # Set the number of Linear layers
        self.h_1 = 256
        self.h_2 = 96
        
        super(DenseConv, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.2)
        self.conv5 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.2)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, self.h_1)
        self.fc2 = nn.Linear(self.h_1, self.h_2)
        self.fc3 = nn.Linear(self.h_2, output_size)

    def forward(self, x):
        '''
        Forward Output
        '''
        x = x.permute(0, 2, 1)  # Change the shape to (batch_size, input_size, sequence_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool1(x)
        x = self.drop1(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.pool2(x)
        x = self.drop2(x)
        x = x.permute(0, 2, 1)  # Change the shape back to (batch_size, sequence_length, hidden_size)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last output of the LSTM sequence
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
        