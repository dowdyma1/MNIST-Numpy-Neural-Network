import numpy as np
import csv, random, os
import sys, getopt

# Hidden layers
SECOND_LAYER_NEURONS = 16
THIRD_LAYER_NEURONS = 16

LEARNING_RATE = .17


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
    second = SECOND_LAYER_NEURONS
    third = THIRD_LAYER_NEURONS

    return([np.random.randn(784, second),np.random.randn(second, third),
        np.random.randn(third,10)])


def getBiases():
    first_layer = SECOND_LAYER_NEURONS
    second_layer = THIRD_LAYER_NEURONS

    return([np.random.randn(first_layer),np.random.randn(second_layer),
        np.random.randn(10)])


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


def feed_forward(image_data, weights, biases, test=False):
    z_data = []
    a_data = []

    a_data.append(image_data)

    neuron_count = [SECOND_LAYER_NEURONS, THIRD_LAYER_NEURONS, 10]
    
    for l in range(len(neuron_count)):
        z = []
        temp = []
        for i in range(neuron_count[l]):
            z.append(sum(a_data[l]*weights[l][:,i]) + biases[l][i])

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

    delta_two = np.sum(mult(weights[2],delta_three),axis=1)*sig_prime(
            z_matrix[1]) 

    del_w[1] = mul_a_delta(a_matrix[1], delta_two)
    del_b[1] = delta_two

    delta_one = np.sum(mult(weights[1],delta_two),axis=1)*sig_prime(
            z_matrix[0]) 

    del_w[0] = mul_a_delta(a_matrix[0], delta_one)
    del_b[0] = delta_one

    return(del_w,del_b)


def incorrectPerLabel(incorrect_labels):
    myDict = {}
    total = len(incorrect_labels)
    for i in range(10):
        myDict[i] = round(100*incorrect_labels.count(i)/float(total),3)

    print("Percentage incorrect for each label:")
    
    for item in myDict:
        print("{}% of {}'s were incorrect.".format(myDict[item], item))


def test(dataset, labels, weights, biases):
    print("\nTESTING:\n")
    incorrect_labels = []
    num_correct = 0
    for i in range(len(dataset)):
        correct = False
        z_data, a_data, your_label = feed_forward(dataset[i], weights, 
                biases, test=True)

        correct_label = -1
        for j in range(10):
            if(int(labels[i][j]) == 1):
                correct_label = j
                break

        if(correct_label == your_label):
            num_correct += 1
            correct = True

        accuracy = float(num_correct)/(i+1)

        # PRINTING FOR TESTING PURPOSES
        if(i%100 == 0):
            print("-- Image #{} --\nYour Label: {}\nCorrect Label: {}\n\
Accuracy: {}\n".format(i+1,your_label, correct_label, round(accuracy,5)))
        
        if(not correct):
            incorrect_labels.append(correct_label)

    accuracy = num_correct/float(len(dataset))

    return(accuracy, incorrect_labels)


# runs the nn using saved weights and biases
def run_nn_saved():
    weights = []
    biases = []
    for i in range(3):
        weights.append(np.loadtxt("values/weights_{}.csv".format(i+1), 
            delimiter=","))
        biases.append(np.loadtxt("values/biases_{}.csv".format(i+1), 
            delimiter=","))

    test_images, test_labels = getTestData()

    final_accuracy, incorrect_labels = test(test_images, test_labels, 
            weights, biases)

    incorrectPerLabel(incorrect_labels)

    print("\nfinal accuracy is: {}%".format(100*final_accuracy))


# Pure Gradient Descent, no batches
def run_nn(no_save_parameters, amt_training=60000):
    dataset, labels = getData(amt_training)

    weights = getWeights()
    biases = getBiases()

    x = 0
    print("\nTRAINING:\n")

    for n in range(amt_training):
        if(n>=x):
            print("Current image number: " + str(n))
            x += 1000

        z_mat, a_mat = feed_forward(dataset[n], weights, biases)
        del_w, del_b = back_prop(z_mat,a_mat, labels[n], weights)

        for i in range(len(weights)):
            weights[i] = weights[i] - LEARNING_RATE*del_w[i]
            biases[i] = biases[i] - LEARNING_RATE*del_b[i]

    for i in range(len(weights)):
        print("\nsaving weights and biases to values directory")

        weight_filepath = "values/weights_" + str(i+1) + ".csv"
        biases_filepath = "values/biases_" + str(i+1) + ".csv"

        if(not no_save_parameters):
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
    final_accuracy, incorrect_labels = test(
            test_images, test_labels, weights, biases)

    return(final_accuracy, incorrect_labels)

def find_learning_rate(start=.15, end=.3, inc=0.001):
    dataset, labels = getData()
    initial_weights = getWeights()
    initial_biases = getBiases()

    learning_rate = start
    myDict = {}

    while(learning_rate <= end):
        print("\nCurrent learning rate: {}\n".format(learning_rate))
        accuracy, _ = run_nn(1)
        myDict[learning_rate] = accuracy

        learning_rate += inc
        learning_rate = round(learning_rate,4)

    return(myDict)


def main(argv):
    saved_parameters = 0
    no_save_parameters = 0
    lr_options = 0
    find_lr_options = []

    try:
        opts, args = getopt.getopt(argv,"hs:t:l:d:w:f:")
    except getopt.GetoptError:
        print('neural_network.py -s <second layer size> -t\
<third layer size> -l <learning rate> -d 1 -w 1 \
-f <start>,<end>,<increase>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("{:<26}\t{:}".format("-s <second layer size>",
                    "Change default size of the second layer"))
            print("{:<26}\t{:}".format("-t <third layer size>",
            "Change default size of the third layer"))
            print("{:<26}\t{:}".format("-l <learning rate>",
                "Change default learning rate"))
            print("{:<26}\t{:}".format("-d 1",
            "Use the saved parameters (weights & biases) \
instead of training them from scratch"))
            print("{:<26}\t{:}".format("-w 1",
"Does not save the parameters (weights & biases) \
to text files"))
            print("{:<26}\t{:}".format("-f <start>,<end>,<increase>",
            "Use the learning rate finder by specifying a starting \
learning rate, the end learning rate, and the amount \
you want to increase every iteration"))
            sys.exit()
        elif opt == "-s":
            SECOND_LAYER_NEURONS = int(arg)
        elif opt == "-t":
            THIRD_LAYER_NEURONS = int(arg)
        elif opt == "-l":
            LEARNING_RATE = float(arg)
        elif opt == "-d":
            saved_parameters = 1
        elif opt == "-w":
            no_save_parameters = 1
        elif opt == "-f":
            lr_options = 1
            find_lr_options = arg.split(",")
            for i in range(len(find_lr_options)):
                find_lr_options[i] = float(find_lr_options[i])

    if(saved_parameters):
        if(lr_options):
            print("Error: cannot use saved parameters with the learning \
rate finder.")
            sys.exit(2)
        elif(no_save_parameters):
            print("Error: cannot save the parameters since they're \
already saved.")
            sys.exit(2)
        else:
            run_nn_saved()
    elif(lr_options):
        if(len(find_lr_options) != 3):
            print("Error: please use all 3 parameters for the learning\
rate finder.")
            sys.exit(2)

        lr_dict = find_learning_rate(start=find_lr_options[0], 
            end=find_lr_options[1], inc=find_lr_options[2])
        print("Accuracy per learning rate:")
        print(lr_dict)
    else:
        final_accuracy, incorrect_labels = run_nn(no_save_parameters)

        incorrectPerLabel(incorrect_labels)

        print("final accuracy is: {}%".format(100*final_accuracy))


if __name__ == "__main__":
    main(sys.argv[1:])
