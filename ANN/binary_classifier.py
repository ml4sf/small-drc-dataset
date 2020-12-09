#https://medium.com/analytics-vidhya/pytorch-for-deep-learning-binary-classification-logistic-regression-382abd97fb43
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split


def get_data(csv_file, training_size_frac=0.75):
    df = pd.read_csv(csv_file)
    X = np.array([df[label] for label in df.columns[1:(len(df.columns)-2)]])
    X = X.transpose()
    Y = np.array(df[df.columns[-1]])
    Y = np.array([1 if _y==0 else 0 for _y in Y])
    print(Y)
   
    training_set_len = int(training_size_frac*X.shape[0])

    X_train = X[:training_set_len,:]
    Y_train = Y[:training_set_len]
    
    X_eval = X[training_set_len:,:]
    Y_eval = Y[training_set_len:]
    X_train, Y_train, X_eval, Y_eval = map(torch.tensor, (X_train, Y_train, X_eval, Y_eval))
    #return X_train.float(), Y_train.long(), X_eval.float(), Y_eval.long()
    return X_train.float(), Y_train.float(), X_eval.float(), Y_eval.float()

class dataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]  

    def __len__(self):
        return self.length

class Net(nn.Module):
    def __init__(self,input_shape):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_shape,400)
        self.fc2 = nn.Linear(400,200)
        self.fc3 = nn.Linear(200,1)  
        
        self.dropout = nn.Dropout(p=0.3)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_data('Descriptors1545Label.csv')

    LEARNING_RATE = 0.01
    BATCH_SIZE = 64
    L2_REGULARIZATION = 0.025
    EPOCHS = 1000
    
    x = X_train
    y = y_train
    
    sc = StandardScaler()
    x = sc.fit_transform(x)
    X_test = sc.fit_transform(X_test)
    

    trainset = dataset(x,y)#DataLoader
    trainloader = DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=False)

    model = Net(input_shape=x.shape[1])
    optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)
    loss_fn = nn.BCELoss()

    #forward loop
    losses = []
    accur = []
    print(x.shape)
    for i in range(EPOCHS):
        for j,(x_train,y_train) in enumerate(trainloader):

            #calculate output
            output = model(x_train)

            #calculate loss
            loss = loss_fn(output,y_train.reshape(-1,1))

            #accuracy
            predicted = model(torch.tensor(X_test, dtype=torch.float32))
            acc = (predicted.reshape(-1).detach().round() == y_test).float().mean()
     
            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses.append(loss)
        accur.append(acc)
        if i%50 == 0:
            print("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,acc))

    #plt.plot(accur, label = 'Accuracy over test set')
    plt.plot(accur, label = 'Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy over the test set')
    plt.ylim(.5, .8)
    #plt.legend(loc='center right')
    plt.savefig('TestCurve', dpi=300)
    plt.show()
