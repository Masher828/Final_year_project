import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
le = LabelEncoder()

data = pd.read_csv("Files/Titanic_Survivor/Dataset/train.csv")

#print(data.head(n=10))

#print(data.info())

columns_to_drop=["PassengerId","Name","Ticket","Cabin","Embarked"]
data_clean=data.drop(columns_to_drop,axis=1)
#print(data_clean.head(n=10))
data_clean["Sex"]=le.fit_transform(data_clean["Sex"])
#print(data_clean.head(n=10))

#print(data_clean.info())

data_clean=data_clean.fillna(data_clean["Age"].mean())
input_cols = ["Pclass","Sex","Age","SibSp","Parch","Fare"]
output_cols=["Survived"]

X=data_clean[input_cols]
Y=data_clean[output_cols]

#print(X.shape,Y.shape)
#print(type(X))


def entropy(col):
    counts = np.unique(col, return_counts=True)
    N = float(col.shape[0])

    entr = 0.0

    for ix in counts[1]:
        p = ix / N
        entr += (-1.0 * p * np.log2(p))
    return entr


cols=np.array([1,1,1,1,1,1,1,1,1])
print(entropy(cols))


def divide_data(x_data, fkey, fval):
    x_right = pd.DataFrame([], columns=x_data.columns)
    x_left = pd.DataFrame([], columns=x_data.columns)

    for ix in range(x_data.shape[0]):

        val = x_data[fkey].loc[ix]

        if val > fval:
            x_right = x_right.append(x_data.loc[ix])

        else:
            x_left = x_left.append(x_data.loc[ix])

    return x_left, x_right

x_left,x_right = divide_data(data_clean[:10],'Sex' , 0.5)
#print(x_left)
#print(x_right)


def information_gain(x_data, fkey, fval):
    left, right = divide_data(x_data, fkey, fval)

    l = float(left.shape[0]) / x_data.shape[0]

    r = float(right.shape[0]) / x_data.shape[0]

    if left.shape[0] == 0 or right.shape[0] == 0:
        return -10000
    i_gain = entropy(x_data.Survived) - (l * entropy(left.Survived) + r * entropy(right.Survived))

    return i_gain


for fx in X.columns:
    #print(fx)

    #print(information_gain(data_clean, fx, data_clean[fx].mean()))


 class DecisionTree:
    def __init__(self, depth=0, max_depth=4):
        self.left = None
        self.right = None
        self.fkey = None
        self.fval = None
        self.max_depth = max_depth
        self.depth = depth
        self.target = None

    def train(self, X_train):
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        info_gains = []

        for ix in features:
            i_gain = information_gain(X_train, ix, X_train[ix].mean())
            info_gains.append(i_gain)

        self.fkey = features[np.argmax(info_gains)]
        self.fval = X_train[self.fkey].mean()

        # print("Making tree features is",self.fkey)

        data_left, data_right = divide_data(X_train, self.fkey, self.fval)
        data_left = data_left.reset_index(drop=True)
        data_right = data_right.reset_index(drop=True)

        if data_left.shape[0] == 0 or data_right.shape[0] == 0:
            if X_train.Survived.mean() >= 0.5:
                self.target = "Survive"

            else:
                self.target = "Dead"

            return

        if (self.depth >= self.max_depth):
            if X_train.Survived.mean() >= 0.5:
                self.target = "Survive"

            else:
                self.target = "Dead"

            return
        self.left = DecisionTree(depth=self.depth + 1, max_depth=self.max_depth)

        self.left.train(data_left)

        self.right = DecisionTree(depth=self.depth + 1, max_depth=self.max_depth)

        self.right.train(data_right)

        if X_train.Survived.mean() >= 0.5:
            self.target = "Survive"

        else:
            self.target = "Dead"
        return

    def predict(self, test):

        if test[self.fkey] > self.fval:

            if self.right is None:
                return self.target

            return self.right.predict(test)

        else:

            if self.left is None:
                return self.target

            return self.left.predict(test)


d=DecisionTree()
d.train(data_clean)

y_pred = []
for ix in range(data_clean.shape[0]):
     y_pred.append(d.predict(data_clean.loc[ix]))

#print(y_pred)

#data4 = pd.read_csv("datasets/file_name3.csv")
#y_pred = []
#for ix in range(data4.shape[0]):
     #y_pred.append(d.predict(data4.loc[ix]))

#print(y_pred)
def survivor(sent1,sent2,sent3,sent4,sent5,sent6):
    name_dict = {
        'Survived': [1],
        'Pclass': [sent1],
        'Sex': [sent2],
        'Age': [sent3],
        'SibSp': [sent4],
        'Parch': [sent5],
        'Fare': [sent6]
    }

    df = pd.DataFrame(name_dict)
    df.to_csv('titanic.csv')
    data = pd.read_csv("titanic.csv")
    y_pred = []
    for ix in range(data.shape[0]):
        y_pred.append(d.predict(data.loc[ix]))

    os.remove('titanic.csv')

    return y_pred




