# Sequence-Regression
This is basic framework for dealing with DNA(RNA) regression tasks with PyTorch. The highlight is that all modules are fully encapsulated and annotated in very detail，even beginners can complete some simple tasks with this framework quickily.  
    
Specific task implementation and modules explanations are as follows:  
## Data_Distribution_Analysis.py
This program can check the distribution in your dataset. And if they are not distributed evenly, you can use some method to improve（log + CDF function as example in my code)  
    
## CustomDataset.py
Class CustomDataset inherits torch.Dataset, You can use it to create your own datasets and dataloaders.  
Sepcific functions:    
  * a. Load： Save the data from the file in the dataset 
  * b. Preprocess：  In bioinformatics, it is possible to initialize the bases by encoding them with **one-hot**. Then embedding layers can be omitted.
  * c. Index：  Using `__getitem__()` to call by index

## Earlystopping.py
Class EarlyStopping is a tool to save the best model in the training process. If there is no performance improvement on several epochs at validation-set, the program can be terminated early.  
  
## DISMIR.py
This is the training model presented in a paper produced by my lab. It uses the Keras framework in the original paper, but I have reproduced it with PyTorch and changed the activation function of the final dense layers to fit the regression task now. I use it to compare to the current model.  
Here is the link of the paper: https://academic.oup.com/bib/article/22/6/bbab250/6318194  

## DenseConv.py
I design my own model to fit my task. I find that multilayer one-dimensional CNN with small convolution kernel + Bidirectional LSTM can accurately extract 
characteristics of short DNA(RNA) sequences with fixed length.
  
## Main.py
This is the main program. It contains training and testing process. Every parameter and action is clearly annotated, and you can check it in the code.  

## Supplementary instruction
My initial task is to build a mapping of a sequence of length 168 to a one-dimensional feature number. Since the data is measured by the laboratory, I have no right to disclose the original data. You can utilize and modify my structure to fit your own task. And I am very pleased to receive suggestions and feedbacks.


