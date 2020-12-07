# Cardiotocography_Classification_NN
This project aims to perform classification task on the [Cardiotocography dataset](https://archive.ics.uci.edu/ml/datasets/Cardiotocography). In this dataset, each cardiotocograms is labeled with a fetal state: N(_Normal_), S(_Suspect_), or P(_Pathologic_). A 3-layer feedforward neural network is designed for classification, and the parameters are refined with results from k-cross validation.  

## Initial Model
First of all, the data in ["ctg_data_cleaned.csv"](https://github.com/StephanieMussi/Cardiotocography_Classification_NN/blob/main/ctg_data_cleaned.csv) is read and split in to train data and test data (70:30).  

Then, a 3-layer neural network with batch size of 32, 10 neurons in the hidden layer, and decay parameter of 1e-6 is implemented.  
```python
batch_size = 32
beta = 0.000001
NO_INPUTS = 21
no_neurons = 10
NO_CLASSES = 3
```
```python
model = Sequential([Dense(no_neurons, activation='relu', 
                          kernel_initializer=RandomUniform(w_min_relu, w_max_relu), 
                          kernel_regularizer=l2(beta)),
                    Dense(NO_CLASSES, activation='softmax', 
                          kernel_initializer=RandomUniform(w_min_softmax, w_max_softmax),
                          kernel_regularizer=l2(beta))])
```
  
  
The model is trained for 200 epochs using [SGD optimizer](https://keras.io/api/optimizers/sgd/). The accuracies are as below:  
Train Accuracy | Test Accuracy
:-: | :-:
89.99% | 89.66%
  
  
Also, the graphs of accuracy and loss are plotted.  
<img src="https://github.com/StephanieMussi/Cardiotocography_Classification_NN/blob/main/Figures/1Acc.png" width="300" height="200">
<img src="https://github.com/StephanieMussi/Cardiotocography_Classification_NN/blob/main/Figures/1Loss.png" width="300" height="200">
  
    
    
## Find Optimal Batch Size
The performances of batch size  = 4, 8, 16, 32, 64 are compared to determine the optimal batch size.   
Here 5-cross validation is used, which slices the train data into 5 folders, and use one as test data in each iteration. The final accuracy is the mean accuracies obtained from the 5 iterations.  
To better compare the time to convergence and convegent accuracies, the number of epochs used is 1000. The results are summarized as below:  
  
  
__Accuracy:__
| Batch Size | 4 | 8 | 16 | 32 | 64 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| Accuracy | 90.24% | 90.03% | 89.16% | 89.70% | 89.63% |
<img src="https://github.com/StephanieMussi/Cardiotocography_Classification_NN/blob/main/Figures/2aAcc.png" width="300" height="200">  
  
  
__Time of 1 epoch:__ 
| Batch Size | 4 | 8 | 16 | 32 | 64 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| Epoch Time | 0.42s | 0.25s | 0.16s | 0.11s | 0.09s |
<img src="https://github.com/StephanieMussi/Cardiotocography_Classification_NN/blob/main/Figures/2aTime.png" width="300" height="200">
  
As can be seen, the accuracies using different batch size are similar, while the epoch time decreases a lot as batch size increases. In this way, 64 is determined as the optimal batch size.  

## Model With Optimal Batch Size
Using the batch size of 64, the model is trained for 200 epochs. The accuracies are shown:  
|Train Accuracy | Test Accuracy|
|:-: | :-:|
|88.91% | 89.97%  |
The accuracy and loss graphs are plotted:  
<img src="https://github.com/StephanieMussi/Cardiotocography_Classification_NN/blob/main/Figures/2cAcc.png" width="300" height="200">
<img src="https://github.com/StephanieMussi/Cardiotocography_Classification_NN/blob/main/Figures/2cLoss.png" width="300" height="200">
    
    
## Find Optimal Number of Hidden Neurons
The performances of number of hidden neurons  = 5, 10, 15, 20, 25 are compared to determine the optimal number of hidden neurons.   
__Accuracy:__
| Number of hidden neurons | 5 | 10 | 15 | 20 | 25 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| Accuracy | 89.29% | 89.36% | 89.09% | 89.36% | 88.69% |
<img src="https://github.com/StephanieMussi/Cardiotocography_Classification_NN/blob/main/Figures/3aAcc.png" width="300" height="200">  
  
  
__Time of 1 epoch:__ 
| Number of hidden neurons | 5 | 10 | 15 | 20 | 25 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| Epoch Time | 0.036s | 0.037s | 0.038s | 0.037s | 0.038s |
<img src="https://github.com/StephanieMussi/Cardiotocography_Classification_NN/blob/main/Figures/3aTime.png" width="300" height="200">

Although the epoch time with 5 hidden neurons is smallest, the time to convergence is larger than the others. The number of hidden neurons is determined to be 10 in balance of accuracy, epoch time and convergent time.

## Model With Optimal Number of Hidden Neurons
Using 10 hidden neurons, the model is trained for 200 epochs. The accuracies are shown:  
|Train Accuracy | Test Accuracy|
|:-: | :-:|
|88.91% | 89.97% |
The accuracy and loss graphs are plotted:  
<img src="https://github.com/StephanieMussi/Cardiotocography_Classification_NN/blob/main/Figures/3cAcc.png" width="300" height="200">
<img src="https://github.com/StephanieMussi/Cardiotocography_Classification_NN/blob/main/Figures/3cLoss.png" width="300" height="200">
  
    
## Find Optimal Weight Decay Parameter
The performances of weight decay parameter = 0, 1e-3, 1e-6, 1e-9, 1e-12 are compared to determine the optimal number of hidden neurons.   
__Accuracy:__
| Decay parameter | 0 | 1e-3 | 1e-6 | 1e-9 | 1e-12 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| Accuracy | 89.29% | 89.36% | 89.09% | 89.36% | 88.69% |
<img src="https://github.com/StephanieMussi/Cardiotocography_Classification_NN/blob/main/Figures/4aAcc.png" width="300" height="200">  
  
  
__Time of 1 epoch:__ 
| Decay parameter | 0 | 1e-3 | 1e-6 | 1e-9 | 1e-12 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| Epoch Time | 0.06s | 0.037s | 0.038s | 0.037s | 0.038s |
<img src="https://github.com/StephanieMussi/Cardiotocography_Classification_NN/blob/main/Figures/4aTime.png" width="300" height="200">

## Model With Optimal Weight Decay Parameter
  

  
    
    
## Comparison With 4-Layer Model
