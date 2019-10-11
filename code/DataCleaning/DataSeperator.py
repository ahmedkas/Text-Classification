import numpy as np
from collections import Counter
from keras.utils import to_categorical

def Seperate(x,y,ratio=0.8,reshape=True):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    allY = []
    classes_count = []
    count = Counter(y)
    for element in count:
        allY.append(element)
    allY.sort()
    for element in allY:
        classes_count.append(count[element])

    taken_classes = [0]*len(allY)
    for i in range(len(x)):
        taken_classes[y[i]] += 1
        if taken_classes[y[i]] <= classes_count[y[i]]*ratio:
            x_train.append(x[i])
            y_train.append(y[i])
        else:
            x_test.append(x[i])
            y_test.append(y[i])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    # y_train = to_categorical(y_train,len(allY))
    # y_test = to_categorical(y_test,len(allY))
    if reshape:
        x_train = x_train.reshape((len(x_train),len(x_train[0]),1))
        x_test = x_test.reshape((len(x_test),len(x_test[0]),1))
    return x_train, y_train, x_test, y_test
