# Cardiotocography_Classification_NN
This project aims to perform classification task on the [Cardiotocography dataset](https://archive.ics.uci.edu/ml/datasets/Cardiotocography). In this dataset, each cardiotocograms is labeled with a fetal state: N(_Normal_), S(_Suspect_), or P(_Pathologic_). A 3-layer feedforward neural network is designed for classification, and the parameters are refined with results from k-cross validation.  

## Initial Model
First of all, the data in 
A 3-layer neural network with batch size of 32, 10 neurons in the hidden layer, and decay parameter of 1e-6 is implemented.  
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

The model is trained for 200 epochs using [SGD optimizer](https://keras.io/api/optimizers/sgd/). 

## Find Optimal Batch Size

## Model With Optimal Batch Size

## Find Optimal Number of Hidden Neurons

## Model With Optimal Number of Hidden Neurons

## Find Optimal Weight Decay Parameter

## Model With Optimal Weight Decay Parameter

## Comparison With 4-Layer Model
