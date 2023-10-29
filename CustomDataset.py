import torch
from torch.utils.data import Dataset
from scipy.stats import norm
import numpy as np
  
class CustomDataset(Dataset):
    def __init__(self, data_path):
        '''
        Load all data
        Save them and Transfer them into the proper format
        '''       
        self.data = []
        with open(data_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split('\t')
                seq = parts[0]
                value = float(parts[1])
                self.data.append((seq, value))
        self.data = np.array(self.data)
        self.normalization()
        self.length = len(self.data)
        
    def __len__(self):
        '''
        Return total number of samples for the dataset
        '''       
        return self.length

    def __getitem__(self, idx):
        '''
        Transfer the DNA(or RNA) sequence into one-hot encoded 
        '''
        seq, value = self.data[idx]        
        processed_seq = self.preprocess_sequence(seq)
        value = float(value)
        # Return inputs and labels
        return processed_seq, torch.tensor(value, dtype=torch.float)
      
    def preprocess_sequence(self, seq):
        '''
        Transfer the DNA(or RNA) sequence into one-hot encoded tensor
        '''        
        base_to_idx = {'A': [1,0,0,0], 'T': [0,1,0,0], 'G': [0,0,1,0], 'C': [0,0,0,1]}  # one-hot for four kind of deoxyribonucleotides and ribonucleotides
        
        seq_vector = [base_to_idx[base] for base in seq]
        seq_tensor = torch.tensor(seq_vector, dtype=torch.float)
        
        return seq_tensor
      
    def normalization(self):
        '''
        The output defaults to one-dimensional floating point
        If the whole data deviates from the normal distribution, it is proved to be effective to take logarithm, normalize and correct with cdf function
        You can check your data with provided 'Data_Distribution_Analysis.py'
        '''        
        values = self.data[:, 1]  # Extract all values
        log_values = np.log(values.astype(float))  # Calculate logarithm
        mean = np.mean(log_values)  # Calculate mean
        std = np.std(log_values)  # Calculate standard deviation
        normalized_values = norm.cdf((log_values - mean) / std)  # Normalize and correct
        self.data[:, 1] = normalized_values

        self.mean = mean  
        self.std = std  # save for inverse_transform

    def inverse_transform(self, normalized_value):
        '''
        Calculate the original data before normalization
        '''
        log_value = norm.ppf(normalized_value) * self.std + self.mean
        original_value = np.exp(log_value)
        return original_value
    
    def data_size(self):
        '''
        Return the size of dataset
        '''        
        return self.length
      
if __name__ == '__main__':
    # Here are few lines of code for testing
    file_path = 'seq_and_value.tsv' 
    test_dataset = CustomDataset(file_path)
    print(test_dataset.data_size())
    print(test_dataset.data[ :3])
    print(test_dataset.inverse_transform(test_dataset.data[0][1].astype(float)))

