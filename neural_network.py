import numpy as np
import time
import csv
import random
import os

# Hidden layers
SECOND_LAYER_NEURONS = 16
THIRD_LAYER_NEURONS = 16

# .1 < x < 1
LEARNING_RATE = .19

# .15   -> 0.8745
# .175  -> 0.8828
# .19   -> 0.8781
# .2    -> 0.8865
# .21   -> 0.8848
# .2125 -> 0.8829
# .225  -> 0.8838
# .25   -> 0.8852
# .3    -> 0.8799
# 2     -> 0.7823

# takes 10 seconds to load whole dataset
def getData(num_sets=60000):
    with open('DATA/train_images.csv', 'r') as f:
        readCSV = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
        i = 0
        partial_image = []
        for row in readCSV:
            if(i == num_sets):
                break
            partial_image.append(np.array(row))
            i+=1

    with open('DATA/train_labels.csv', 'r') as l:
        readCSV = csv.reader(l,quoting=csv.QUOTE_NONNUMERIC)
        j = 0
        partial_label = []
        for row in readCSV:
            if(j == num_sets):
                break
            partial_label.append(np.array(row))
            j+=1


    return(partial_image, partial_label)


def getTestData(num_sets=10000):
    with open('DATA/test_images.csv', 'r') as f:
        readCSV = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
        i = 0
        partial_image = []
        for row in readCSV:
            if(i == num_sets):
                break
            partial_image.append(np.array(row))
            i+=1

    with open('DATA/test_labels.csv', 'r') as l:
        readCSV = csv.reader(l,quoting=csv.QUOTE_NONNUMERIC)
        j = 0
        partial_label = []
        for row in readCSV:
            if(j == num_sets):
                break
            partial_label.append(np.array(row))
            j+=1


    return(partial_image, partial_label)


def getWeights():
    second_layer = 256
    third_layer = 100

    np.random.seed(42)

    return([np.random.randn(784,16),np.random.randn(16,16),np.random.randn(16,10)])

def getBiases():
    first_layer = 16
    second_layer = 16
    third_layer = 10

    np.random.seed(11)

    return([np.random.randn(first_layer),np.random.randn(second_layer),np.random.randn(third_layer)])


def sigmoid(x):
    return(1.0/(1+np.exp(-x)))

def sig_prime(x):
    return(np.exp(-x)/((1+np.exp(-x))**2))

# first is 2d, second is 1d; numpy arrays; a x b * b
def mult(first, second):
    temp = np.zeros_like(first)
    for i in range(len(second)):
        temp[:,i] = first[:,i]*second[i]
    return temp

# first is 1d, length a; second is 1d, length b; -> a x b
def mul_a_delta(a, delta):
    temp = np.zeros((len(a),len(delta)))
    for i in range(len(delta)):
        temp[:,i] = a*delta[i]
    return temp


def feed_forward(image_data, weights, biases,prnt=False, tmp=None, test=False):
    z_data = []
    a_data = []

    a_data.append(image_data)

    neuron_count = [16, 16, 10]
    
    for l in range(len(neuron_count)):
        z = []
        temp = []
        for i in range(neuron_count[l]):
            z.append(sum(a_data[l]*weights[l][:,i]) + biases[l][i])
            #print("{} * {}".format(np.shape(a_data[l]), np.shape(weights[l][:,i])))

            #print("l: {} i: {}".format(l, i))
            #print("{} * {} \n\n".format(np.shape(weights[l][:,i]), 
            #    np.shape(a_data[l])))

        z = np.array(z, dtype=np.float128)

        z_data.append(z)
        a_data.append(sigmoid(z))


    if(test == True):
        max_val = -1
        temp_test = a_data[-1]
        index = -1

        for i in range(len(temp_test)):
            if(temp_test[i] > max_val):
                max_val = temp_test[i]
                index = i

        if(index <0):
            print('\n\n------------ERROR------------\n\n')
            return('error')

        return(z_data, a_data, index)
    


    return(z_data, a_data)


def back_prop(z_matrix, a_matrix, label, weights):
    del_w = [0,0,0]
    del_b = [0,0,0]

    delta_three = (a_matrix[3] - label) * sig_prime(z_matrix[2]) # size 10

    del_w[2] = mul_a_delta(a_matrix[2], delta_three)
    del_b[2] = delta_three

    delta_two = np.sum(mult(weights[2],delta_three),axis=1)*sig_prime(z_matrix[1]) # size 16

    
    del_w[1] = mul_a_delta(a_matrix[1], delta_two)
    del_b[1] = delta_two

    delta_one = np.sum(mult(weights[1],delta_two),axis=1)*sig_prime(z_matrix[0]) # size 16


    del_w[0] = mul_a_delta(a_matrix[0], delta_one)
    del_b[0] = delta_one


    '''
    for i in range(3):
        rev = 2-i
        if(i==0):
            delta_partial = (a_matrix[3] - label)
        else:
            delta_partial = np.sum(mult(weights[rev], delta),axis=1)

        print(i)
        delta = delta_partial* sig_prime(z_matrix[rev])

        del_w[rev] = mul_a_delta(a_matrix[rev], delta)
        #del_w[rev] = a_matrix[rev].reshape([len(a_matrix[rev]),1]).dot(delta.reshape([1,len(delta)]))
        del_b[rev] = delta
    '''
    

    return(del_w,del_b)


def test(dataset, labels, weights, biases):
    incorrect_labels = []
    num_correct = 0
    for i in range(len(dataset)):
        correct = False
        z_data, a_data, your_label = feed_forward(dataset[i], weights, biases, test=True)

        correct_label = -1
        for j in range(10):
            if(int(labels[i][j]) == 1):
                correct_label = j
                break
        #print("{} vs {}\n".format(labels[i], correct_label))

        if(correct_label == your_label):
            num_correct += 1

        accuracy = float(num_correct)/(i+1)

        # PRINTING FOR TESTING PURPOSES
        if(i%100 == 0):
            print("-- Image #{} --\nYour Label: {}\nCorrect Label: {}\nAccuracy: {}\n"
                .format(i+1,your_label, correct_label, round(accuracy,3)))
            incorrect_labels.append(correct_label)

    return(accuracy, incorrect_labels)


# Includes Stochastic Gradient Descent
def main():
    amt_training = 60000
    #amt_training = 4
    dataset, labels = getData(amt_training)

    weights = getWeights()
    biases = getBiases()

    n = 0
    x = 0

    ### training data
    print("Training...")
    while(n < amt_training):
        # printout for training
        if(n>x):
            print(n)
            x += 1000

        # randomizing batch size, m
        if(amt_training-n > 100):
            m = random.randint(1,100)
        else:
            m = amt_training-n

        del_w = [np.zeros((784,16)),np.zeros((16,16)),np.zeros((16,10))]
        del_b = [np.zeros((16)), np.zeros((16)), np.zeros((10))]

        for i in range(m):

            z_mat, a_mat = feed_forward(
                    dataset[n+i], weights, biases, prnt=True)

            add_del_w, add_del_b = back_prop(
                    z_mat, a_mat, labels[n+i], weights)

            # updating del_w and del_b lists
            for i in range(len(del_w)):
                del_w[i] += add_del_w[i]
                del_b[i] += add_del_b[i]


            '''
            print(add_del_b)
            print(np.shape(add_del_b))
            print("\n")
            print(del_b)
            print(np.shape(del_b))
            print("\n\n\n\n\n")
            '''

        n += m

        for i in range(len(weights)):
            weights_change = (LEARNING_RATE/m)*del_w[i]
            biases_change = (LEARNING_RATE/m)*del_b[i]

            weights[i] = weights[i] - weights_change
            biases[i] = biases[i] - biases_change


    ###
    print("Finished training...")

    for i in range(len(weights)):
        weight_filepath = "values/weights_" + str(i+1) + ".csv"
        biases_filepath = "values/biases_" + str(i+1) + ".csv"

        # logic for filepath
        if not os.path.exists(weight_filepath):
            if not os.path.exists("values"):
                os.mkdir("values")
            os.mknod(weight_filepath)
        if not os.path.exists(biases_filepath):
            if not os.path.exists("values"):
                os.mkdir("values")
            os.mknod(biases_filepath)

        np.savetxt(weight_filepath, weights[i], delimiter=",")
        np.savetxt(biases_filepath, biases[i], delimiter=",")
    

    test_images, test_labels = getTestData()
    final_accuracy, incorrect_labels = test(test_images, test_labels, weights, biases)

    print("Final accuracy: " + str(final_accuracy))

def main_exper():
    dataset, labels = getData(1)
    print(dataset[0])


def main_saved():
    weights = []
    biases = []
    for i in range(3):
        weights.append(np.loadtxt("values/weights_{}.csv".format(i+1), delimiter=","))
        biases.append(np.loadtxt("values/biases_{}.csv".format(i+1), delimiter=","))


    test_images, test_labels = getTestData()

    final_accuracy, incorrect_labels = test(test_images[:1000], test_labels, weights, biases)

    myDict = {}
    total = len(incorrect_labels)
    for i in range(10):
        myDict[i] = round(100*incorrect_labels.count(i)/float(total),3)

    print(myDict) # percentage incorrect per label


# Pure Gradient Descent
def main_pure(amt_training=60000):
    dataset, labels = getData(amt_training)

    weights = getWeights()
    biases = getBiases()

    x = 0

    for n in range(amt_training):
        if(n>x):
            print(n)
            x += 1000

        z_mat, a_mat = feed_forward(dataset[n], weights, biases)
        del_w, del_b = back_prop(z_mat,a_mat, labels[n], weights)


        for i in range(len(weights)):
            weights[i] = weights[i] - LEARNING_RATE*del_w[i]
            biases[i] = biases[i] - LEARNING_RATE*del_b[i]

    #for i in range(len(weights)):
    #    np.savetxt("values/weights_{}.csv".format(i+1), weights[i], delimiter=",")
    #    np.savetxt("values/biases_{}.csv".format(i+1), biases[i], delimiter=",")

    test_images, test_labels = getTestData()
    final_accuracy, incorrect_labels = test(test_images, test_labels, weights, biases)

    return(final_accuracy)


# no batches
def main_pure_test(dataset, labels, weights, biases, learning_rate):
    amt_training = 60000

    for n in range(amt_training):

        z_mat, a_mat = feed_forward(dataset[n], weights, biases)
        del_w, del_b = back_prop(z_mat,a_mat, labels[n], weights)


        for i in range(len(weights)):
            weights[i] = weights[i] - learning_rate*del_w[i]
            biases[i] = biases[i] - learning_rate*del_b[i]


    test_images, test_labels = getTestData()
    final_accuracy, incorrect_labels = test(test_images, test_labels, weights, biases)

    return(final_accuracy)


def find_learning_rate():
    dataset, labels = getData()
    initial_weights = getWeights()
    initial_biases = getBiases()

    start = .15
    end = .3

    learning_rate = start
    myDict = {}

    while(learning_rate <= end):
        accuracy = main_pure_test(dataset, labels, initial_weights, initial_biases, learning_rate)
        myDict[learning_rate] = accuracy
        print(myDict)
        print()

        learning_rate += 0.001
        learning_rate = round(learning_rate,4)

    return(myDict)


#print(find_learning_rate())
#main_saved()
main_pure()
#main_exper()
