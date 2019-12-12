#!/usr/bin/env python
# coding: utf-8

import sys 
import numpy as np

oh1 = np.ones((101, 1))
ol = np.ones((10,1))

deltasol = np.ones((10, 1))
deltasoldir = np.ones((10, 1))
deltasow = np.ones((10, 101))

deltash1 = np.ones((100, 1))
deltash1dir = np.ones((100, 1))
deltash1w = np.ones((100, ((14*14)+1)))

mom = 0.9
prev_layer_1_weight_change = np.zeros((100, (14*14)+1))
prev_layer_2_weight_change = np.zeros((10, 101))

conf_m = np.zeros((10,10))

learning_rate = 0.05

num_images = 2000
num_images_test = 8000

def load_new_data():
    image_size = 28
    new_image_size = 14
    fac = 0.99 / 255
    
    f = open("t10k-images.idx3-ubyte", "rb")
    f.read(16 + (image_size * image_size * num_images))

    buf = f.read(image_size * image_size * num_images_test)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images_test, image_size, image_size, 1)

    data_new = np.zeros((num_images_test, new_image_size*new_image_size),dtype='float32')

    #resize to 14X14
    for Index in range(num_images_test):
        D = data[Index,:].reshape((image_size,image_size))
        D = D[0::2,0::2]
        D.reshape((new_image_size*new_image_size)).shape
        data_new[Index] = D.reshape(new_image_size*new_image_size)

    for i in range(num_images_test):
        for j in range(14*14):
            data_new[i][j] = (data_new[i][j] * fac) + .01

    return data_new
    
def load_new_labels():
    f = open("t10k-labels.idx1-ubyte", "rb")
    f.read(8 + (num_images))

    labels = np.zeros(num_images_test, dtype='int64')

    for i in range(num_images_test):
        buf = f.read(1)
        labels[i] = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    return labels

def write_weights_1(list):
    f = open("weights1.bin", 'wb')
    f.truncate(0) 
    np.save(f, list)
            
def write_weights_2(list):
    f = open("weights2.bin", 'wb')
    f.truncate(0) 
    np.save(f, list)
            
def read_weights_1(list):
    f = open("weights1.bin", 'rb')
    
    return np.load(f)

def read_weights_2(list):
    f = open("weights2.bin", 'rb')
    
    return np.load(f)

def loadLabels():
    f = open("t10k-labels.idx1-ubyte", "rb")
    f.read(8)

    labels = np.zeros(num_images, dtype='int64')

    for i in range(num_images):
        buf = f.read(1)
        labels[i] = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    return labels

def loadData():
    f = open("t10k-images.idx3-ubyte", "rb")
    f.read(16)
    
    image_size = 28
    new_image_size = 14
    fac = 0.99 / 255

    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)

    data_new = np.zeros((num_images, new_image_size*new_image_size),dtype='float32')

    #resize to 14X14
    for Index in range(num_images):
        D = data[Index,:].reshape((image_size,image_size))
        D = D[0::2,0::2]
        D.reshape((new_image_size*new_image_size)).shape
        data_new[Index] = D.reshape(new_image_size*new_image_size)

    for i in range(num_images):
        for j in range(14*14):
            data_new[i][j] = (data_new[i][j] * fac) + .01

    return data_new

#def printImage(image): 
#    image = image.reshape((14,14))
#    plt.imshow(image, cmap="Greys")
#    plt.show()

def init_layer1_weights():
    weights = np.random.uniform(low=0.0, high=0.01, size=(100, ((14*14)+1)))
    try:
        weights = read_weights_1(weights)
        return weights
    except:
        weights = np.random.uniform(low=0.0,high=0.01,size=(100, ((14*14)+1)))
        print("Could not read weights from file! Did you include weights1.bin")
        return weights
    
    return weights

def init_layer2_weights():
    weights = np.random.uniform(low=0.0,high=0.01,size=(10, 101))
    try:
        weights = read_weights_2(weights)
        return weights
    except:
        weights = np.random.uniform(low=0.0,high=0.01,size=(10, 101))
        print("Could not read weights from file! Did you include weights2.bin?")
        return weights
        
    return weights

def add_bias_to_image_data(image_data):	
    image_data = np.append(image_data, np.ones([(num_images),1]), 1)

    return image_data

def add_bias_to_image_data_new(image_data):	
    image_data = np.append(image_data, np.ones([(num_images_test),1]), 1)

    return image_data

def load_image_labels_one_hot(image_labels):
    image_labels_one_hot = np.ones((num_images, 10), dtype='int64')

    lr = np.arange(10)
    for i in range(num_images):
        image_labels_one_hot[i] = (lr==image_labels[i]).astype(np.int)

    return image_labels_one_hot

def load_image_labels_one_hot_new(image_labels):
    image_labels_one_hot = np.ones((num_images_test, 10), dtype='int64')

    lr = np.arange(10)
    for i in range(num_images_test):
        image_labels_one_hot[i] = (lr==image_labels[i]).astype(np.int)

    return image_labels_one_hot

def sigmoid(x, derive=False):
    if derive:
        return x * (1.0 - x) 

    if (x < 0):
        a = np.exp(x) 
        return (a / (1 + a))
        
    else:
        return ( 1.0 / (1.0 + np.exp(-x)) )

def forward_pass(layer_1_weights, layer_2_weights, input_layer_1, index):

    for i in range(100):
        oh1[i] = np.dot(input_layer_1[index,:], layer_1_weights[i])
        oh1[i] = sigmoid(oh1[i])
        
    for j in range(10):
        ol[j] = np.dot(np.transpose(oh1), layer_2_weights[j])
        ol[j] = sigmoid(ol[j])
    
def getError(labels, index):
    totalError = 0

    for i in range(10):
        totalError += ((1.0/2.0) * np.power(labels[index][i] - ol[i], 2.0))

    return totalError

def backprop_ol(labels, index):
    for i in range(10):
        deltasol[i] = (-1.0) * (labels[index][i] - ol[i])

    for o in range(10):
        deltasoldir[o] = sigmoid(ol[o], derive=True)

    for z in range(10):
        for j in range(101):
            deltasow[z][j] = oh1[j] * (deltasol[z] * deltasoldir[z])

def backprop_h1(input, layer_2_weights, index):
    for i in range(100):
        deltash1dir[i] += sigmoid(oh1[i], derive=True)

    sum = 0

    sums = np.ones((101, 1))
    p = 0

    for q in range(101):
        for l in range(10):
            sum += deltasol[l] * deltasoldir[l] * layer_2_weights[l][p]
        sums[q] = sum
        sum = 0
        p = p + 1


    for j in range(100):
        for k in range((14*14) +1):
            deltash1w[j][k] = input[index][k] * deltash1dir[j] * sums[j]

def update_weights_h1(layer_1_weights):
    for j in range(100):
        for i in range((14*14 )+1):
            v = (prev_layer_1_weight_change[j][i] * mom) - (learning_rate * deltash1w[j][i]) 
            prev_layer_1_weight_change[j][i] = v
            layer_1_weights[j][i] += v

    return layer_1_weights

def update_weights_ol(layer_2_weights):
    for j in range(10):
        for i in range(101):
            v = (prev_layer_2_weight_change[j][i] * mom) - (learning_rate * deltasow[j][i]) 
            prev_layer_2_weight_change[j][i] = v
            layer_2_weights[j][i] += v

    return layer_2_weights

def output_to_probability():
    sum = 0
    for i in range(10):
        sum+= ol[i]
        
    for j in range(10):
        ol[j] = ol[j]/sum
        
def print_top():
    inx = 0
    for i in range(10):
        if(ol[i] > ol[inx]):
            inx = i
            
    print("The network guessed: ", inx, " with a probability of: ", ol[inx])
    
def update_conf_m(expected, actual):
    conf_m[expected][actual] = conf_m[expected][actual] + 1
    
def print_conf_m():
    print(conf_m)
    
def get_best_guess():
    guess = ol[0]
    guessNum = 0
    for j in range(10):
        if(guess < ol[j]):
            guess = ol[j]
            guessNum = j
            
    return guessNum

def get_total_acc():
    right = 0
    wrong = 0
    
    for i in range(10):
        for j in range(10):
            if(j != i):
                wrong = wrong + conf_m[i][j]
            else:
                right = right + conf_m[i][j]
    print("Right: ", right)
    print("Wrong: ", wrong)
    print("Acc: ", right/(wrong+right))
            
def test_all():
    image_data = load_new_data()
    image_labels = load_new_labels()
    
    image_labels_one_hot = load_image_labels_one_hot(image_labels)
    final_input = add_bias_to_image_data_new(image_data)
    layer_1_weights = init_layer1_weights()
    layer_2_weights = init_layer2_weights()
    
    for i in range(num_images_test):
        forward_pass(layer_1_weights, layer_2_weights, final_input, i)
        output_to_probability()
        
        guess = get_best_guess()
        update_conf_m(image_labels[i], guess)
        
    print_conf_m()
    get_total_acc()
    
def learn():

    image_data = loadData()
    image_labels = loadLabels()

    image_labels_one_hot = load_image_labels_one_hot(image_labels)
    final_input = add_bias_to_image_data(image_data)
    layer_1_weights = init_layer1_weights()
    layer_2_weights = init_layer2_weights()

    epoch = 8
    err_break = 0.001
    err = np.zeros((epoch,1))
    end_index = epoch-1 
    inds = np.arange(num_images)

    for k in range(epoch):
        err[k] = 0 

        for i in range(num_images):
            inx = inds[i]

            forward_pass(layer_1_weights, layer_2_weights, final_input, inx)

            e = getError(image_labels_one_hot, inx)

            err[k] = err[k] + e

            backprop_ol(image_labels_one_hot, inx)
            backprop_h1(final_input, layer_2_weights, inx)

            layer_1_weights = update_weights_h1(layer_1_weights)
            layer_2_weights = update_weights_ol(layer_2_weights)

        if(err[k] < err_break):
            print("error passed on epoch ", k)
            end_index = k
            break

#    plt.plot(err[0:end_index])
#    plt.ylabel('error')
#    plt.xlabel('epochs')
#    plt.show()
    
    write_weights_1(layer_1_weights)
    write_weights_2(layer_2_weights)
    
    
def test(inx):
    image_data = load_new_data()
    image_labels = load_new_labels()
    
    image_labels_one_hot = load_image_labels_one_hot_new(image_labels)
    final_input = add_bias_to_image_data_new(image_data)
    layer_1_weights = init_layer1_weights()
    layer_2_weights = init_layer2_weights()
    
    forward_pass(layer_1_weights, layer_2_weights, final_input, inx)
    output_to_probability()
    print("Got: ", ol, "\nwanted: ", image_labels_one_hot[inx])
    print_top()
    
    printImage(image_data[inx])
    

i = input("train or test?: ")

if(i=="train" or i=="Train"):
    learn()
if(i=="test" or i=="Test"):
    test_all()
else:
    print("Input not recognized, run again. You may have to put the input in quotes if running from command line.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




