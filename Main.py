import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from scipy.stats import pearsonr
from CustomDataset import CustomDataset
from DenseConv import DenseConv
from DISMIR import DISMIR
from EarlyStopping import EarlyStopping


class ModelWorking:
    def __init__(self):
        '''
        Initialize all parameters, data sets, and model
        '''
        self.seed = 19  #seed number
        self.file_path = 'seq_and_value.tsv'  # file-path of the data
        self.save_path = 'best_model.pth'  # save-path for the model
        
        self.batch_size = 128  # batch_size
        self.sequence_length = 168  #length of the sequence
        self.input_size = 4  #kinds of bases
        self.hidden_size = 256  # number of hidden layers
        self.output_size = 1  # dim of output
        self.lr = 3e-4  # LEARNING RATE
        self.weight_decay = 1e-5  # control the degree of weight decay
        self.epoch = 500  # epochs to train
        self.train_patience = 50  # patience for early stopping
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check GPU and use it
        
        self.model = DISMIR(self.input_size, self.hidden_size, self.output_size, self.sequence_length).to(self.device)
        self.criterion = nn.MSELoss()
        # Using Adam as optimizer as an example. SGD is another common choice.
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.early_stopping = EarlyStopping(self.train_patience, checkpoint_path=self.save_path, mode='max')
        
        self.set_seed(self.seed)  # Fix the seed
        all_dataset = CustomDataset(self.file_path)  # read all data
        
        # shuffle all data
        indices = list(range(all_dataset.data_size()))
        random.shuffle(indices)

        # divide train/validation/test dataset
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1

        train_size = int(self.train_ratio * len(indices))
        val_size = int(self.val_ratio * len(indices))
        test_size = int(self.test_ratio * len(indices))

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        train_dataset = Subset(all_dataset, train_indices)
        val_dataset = Subset(all_dataset, val_indices)
        test_dataset = Subset(all_dataset, test_indices)
        
        # Create data_loader
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        
    def evaluate_model(self, loader, model, criterion, device):
        """
        This is a function to evaluate the model
        The function can return loss, all outputs and all targets
        The main reason is to make it easier for external code to take data and evaluate it in various ways appropriately
        For my program, I mainly choose pearson correlation to evaluate fitting degree
        """
        model.eval()  
        total_loss = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():  # No need to calculate grad
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(-1))  # calculate MSE
                total_loss += loss.item() * inputs.size(0)
                all_outputs.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        return total_loss / len(loader.dataset), all_outputs, all_targets
  
    def train(self):
        '''
        This is the main module for training
        At the end, a line chart is generated to show the entire training process
        '''
        corr = []  # Save the correlation for each epoch
        
        # Start to train the model
        for epoch in range(self.epoch):
            for inputs, targets in self.train_loader:
                self.model.train()
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.unsqueeze(-1))
                loss.backward()
                self.optimizer.step()
                
            print(f'Epoch [{epoch+1}/{self.epoch}], Loss: {loss.item()}')
            validation_mse, validation_outputs, validation_targets = self.evaluate_model(self.validation_loader, self.model, self.criterion, self.device)
            validation_outputs = np.array(validation_outputs).flatten()
            pearson_corr, _ = pearsonr(validation_outputs, validation_targets)  # Choose pearson correlation to evaluate the fitting degree
            corr.append(pearson_corr)
            print('Validation MSE:{}'.format(validation_mse))
            print('Validation Pearson correlation: {}'.format(pearson_corr))
           
            # Early Stopping
            self.early_stopping.step(pearson_corr, self.model)
            if(self.early_stopping.should_stop()):
                print('Early Stopping is Triggered.')
                self.epoch = epoch + 1
                break
            else:
              print('Early Stopping count: %s / %s'%(self.early_stopping.now_count(),self.train_patience))

        # Draw a line chart to show the training progress
        plt.plot(range(self.epoch), corr)
        plt.xlabel('Epoch')
        plt.ylabel('Correlation')
        plt.title('Correlation vs. Epoch')
        plt.savefig('training.png')
        
    def test(self):
        '''
        After training, using this function to test the model
        '''
        self.early_stopping.load_checkpoint(self.model)  # Load the model before early stopping
        test_mse, test_outputs, test_targets = self.evaluate_model(self.test_loader, self.model, self.criterion, self.device)
        test_outputs = np.array(test_outputs).flatten()
        pearson_corr, _ = pearsonr(test_outputs, test_targets)
        print('Test MSE:{}'.format(test_mse))
        print('Test Pearson correlation: {}'.format(pearson_corr))
        
        # Draw a scatter plot to show the effect on the test set
        plt.clf()  # Clear the canvas
        plt.scatter(test_outputs, test_targets, label=f'Pearson Correlation = {pearson_corr:.3f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Predicted vs Actual')
        plt.legend()
        plt.savefig('test.png')
        
    def set_seed(self, seed):
        '''
        Fix the seed for all random factors
        '''
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


# Main program
if __name__ == '__main__':
    analysis = ModelWorking()
    analysis.train()
    analysis.test()