# Introduction
This project was created May 2019. It is a neural network with two hidden
layers, coded without any conventional neural network libraries with the
purpose to learn intimately how a basic neural network works. I was
introduced
to neural networks through one of [3Blue1Brown's YouTube videos](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&pbjreload=101) and I
was instantly hooked. In the video, he recommended [this online book](http://neuralnetworksanddeeplearning.com/) 
to get a deep understanding of the basics behind neural networks. I studied
the book and after some programming, managed to complete this project.

Although I completed this project May 2019, I have edited it quite
considerably to polish and update it, but the core of the code remains the same.

# Installation (linux and macOS)
You can download this project by doing:
```
git clone https://github.com/dowdyma1/MNIST-Numpy-Neural-Network
```
## Installing the image data
1. Download the data from [here](http://yann.lecun.com/exdb/mnist/) and place it in the DATA folder.
2. in the DATA folder, run command `gunzip *.gz`
3. still in the DATA folder, run `python convert_mnist.py`. This converts the images and labels to csv
4. All done! Ready to run program.

# Usage
Here are the commandline arguments:
```
-h                              prints this message

-s <second layer size>    	    Change default size of the second layer
-t <third layer size>     	    Change default size of the third layer
-l <learning rate>        	    Change default learning rate
-d 1                      	    Use the saved parameters (weights & biases) instead of training them from scratch
-w 1                      	    Does not save the parameters (weights & biases) to text files
-f <start>,<end>,<increase>	    Use the learning rate finder by specifying a starting learning rate, the end learning rate, and the amount you want to increase every iteration

```

# Layout of this neural network
This is a basic gradient descent neural network with two hidden layers
and using a logistic regression cost function.

The images are grayscale 28x28 pixels and they have been flattened to a matrix of
784 x 1. They have also been normalized, so since the value for black is
256, every pixel is divided by 256 making every value in the image matrix
between or equal to 0 or 1.

## Walkthrough of feed forward function
![Feed forward](/Tutorial_images/feedforward_vis-1.jpg)

## Walkthrough of backpropagation function
![Backpropagation](/Tutorial_images/backpropvis_full.png)

# Run sample
```
python neural_network.py
```

```

TRAINING:

Current image number: 0
Current image number: 1000
Current image number: 2000
Current image number: 3000
Current image number: 4000
Current image number: 5000
Current image number: 6000
Current image number: 7000
Current image number: 8000
Current image number: 9000
Current image number: 10000
Current image number: 11000
Current image number: 12000
Current image number: 13000
Current image number: 14000
Current image number: 15000
Current image number: 16000
Current image number: 17000
Current image number: 18000
Current image number: 19000
Current image number: 20000
Current image number: 21000
Current image number: 22000
Current image number: 23000
Current image number: 24000
Current image number: 25000
Current image number: 26000
Current image number: 27000
Current image number: 28000
Current image number: 29000
Current image number: 30000
Current image number: 31000
Current image number: 32000
Current image number: 33000
Current image number: 34000
Current image number: 35000
Current image number: 36000
Current image number: 37000
Current image number: 38000
Current image number: 39000
Current image number: 40000
Current image number: 41000
Current image number: 42000
Current image number: 43000
Current image number: 44000
Current image number: 45000
Current image number: 46000
Current image number: 47000
Current image number: 48000
Current image number: 49000
Current image number: 50000
Current image number: 51000
Current image number: 52000
Current image number: 53000
Current image number: 54000
Current image number: 55000
Current image number: 56000
Current image number: 57000
Current image number: 58000
Current image number: 59000

saving weights and biases to values directory

TESTING:

-- Image #1 --
Your Label: 7
Correct Label: 7
Accuracy: 1.0

-- Image #101 --
Your Label: 6
Correct Label: 6
Accuracy: 0.90099

-- Image #201 --
Your Label: 3
Correct Label: 3
Accuracy: 0.89552

-- Image #301 --
Your Label: 4
Correct Label: 4
Accuracy: 0.88372

-- Image #401 --
Your Label: 2
Correct Label: 2
Accuracy: 0.88279

-- Image #501 --
Your Label: 3
Correct Label: 3
Accuracy: 0.87226

-- Image #601 --
Your Label: 6
Correct Label: 6
Accuracy: 0.86522

-- Image #701 --
Your Label: 1
Correct Label: 1
Accuracy: 0.86163

-- Image #801 --
Your Label: 5
Correct Label: 8
Accuracy: 0.86267

-- Image #901 --
Your Label: 1
Correct Label: 1
Accuracy: 0.86681

-- Image #1001 --
Your Label: 9
Correct Label: 9
Accuracy: 0.86813

-- Image #1101 --
Your Label: 7
Correct Label: 7
Accuracy: 0.86558

-- Image #1201 --
Your Label: 3
Correct Label: 8
Accuracy: 0.85928

-- Image #1301 --
Your Label: 4
Correct Label: 4
Accuracy: 0.85165

-- Image #1401 --
Your Label: 6
Correct Label: 6
Accuracy: 0.85225

-- Image #1501 --
Your Label: 1
Correct Label: 7
Accuracy: 0.8501

-- Image #1601 --
Your Label: 3
Correct Label: 3
Accuracy: 0.84884

-- Image #1701 --
Your Label: 0
Correct Label: 0
Accuracy: 0.84774

-- Image #1801 --
Your Label: 4
Correct Label: 6
Accuracy: 0.84453

-- Image #1901 --
Your Label: 1
Correct Label: 1
Accuracy: 0.84692

-- Image #2001 --
Your Label: 6
Correct Label: 6
Accuracy: 0.84558

-- Image #2101 --
Your Label: 5
Correct Label: 5
Accuracy: 0.84388

-- Image #2201 --
Your Label: 2
Correct Label: 2
Accuracy: 0.84234

-- Image #2301 --
Your Label: 3
Correct Label: 3
Accuracy: 0.84398

-- Image #2401 --
Your Label: 5
Correct Label: 5
Accuracy: 0.84382

-- Image #2501 --
Your Label: 2
Correct Label: 2
Accuracy: 0.84526

-- Image #2601 --
Your Label: 8
Correct Label: 8
Accuracy: 0.84391

-- Image #2701 --
Your Label: 7
Correct Label: 7
Accuracy: 0.84487

-- Image #2801 --
Your Label: 8
Correct Label: 8
Accuracy: 0.84648

-- Image #2901 --
Your Label: 4
Correct Label: 4
Accuracy: 0.84729

-- Image #3001 --
Your Label: 6
Correct Label: 6
Accuracy: 0.84605

-- Image #3101 --
Your Label: 5
Correct Label: 5
Accuracy: 0.84811

-- Image #3201 --
Your Label: 9
Correct Label: 9
Accuracy: 0.84817

-- Image #3301 --
Your Label: 3
Correct Label: 3
Accuracy: 0.84914

-- Image #3401 --
Your Label: 7
Correct Label: 7
Accuracy: 0.84975

-- Image #3501 --
Your Label: 4
Correct Label: 4
Accuracy: 0.85119

-- Image #3601 --
Your Label: 3
Correct Label: 2
Accuracy: 0.85087

-- Image #3701 --
Your Label: 4
Correct Label: 4
Accuracy: 0.85166

-- Image #3801 --
Your Label: 6
Correct Label: 6
Accuracy: 0.85083

-- Image #3901 --
Your Label: 1
Correct Label: 1
Accuracy: 0.84953

-- Image #4001 --
Your Label: 2
Correct Label: 9
Accuracy: 0.84879

-- Image #4101 --
Your Label: 2
Correct Label: 2
Accuracy: 0.84931

-- Image #4201 --
Your Label: 7
Correct Label: 7
Accuracy: 0.85004

-- Image #4301 --
Your Label: 1
Correct Label: 5
Accuracy: 0.84957

-- Image #4401 --
Your Label: 9
Correct Label: 7
Accuracy: 0.84913

-- Image #4501 --
Your Label: 1
Correct Label: 9
Accuracy: 0.84981

-- Image #4601 --
Your Label: 3
Correct Label: 3
Accuracy: 0.85068

-- Image #4701 --
Your Label: 9
Correct Label: 9
Accuracy: 0.85173

-- Image #4801 --
Your Label: 7
Correct Label: 7
Accuracy: 0.85107

-- Image #4901 --
Your Label: 7
Correct Label: 7
Accuracy: 0.84942

-- Image #5001 --
Your Label: 3
Correct Label: 3
Accuracy: 0.84963

-- Image #5101 --
Your Label: 9
Correct Label: 9
Accuracy: 0.85081

-- Image #5201 --
Your Label: 4
Correct Label: 4
Accuracy: 0.85214

-- Image #5301 --
Your Label: 8
Correct Label: 8
Accuracy: 0.85399

-- Image #5401 --
Your Label: 5
Correct Label: 5
Accuracy: 0.85558

-- Image #5501 --
Your Label: 4
Correct Label: 4
Accuracy: 0.85784

-- Image #5601 --
Your Label: 9
Correct Label: 7
Accuracy: 0.85895

-- Image #5701 --
Your Label: 3
Correct Label: 3
Accuracy: 0.85967

-- Image #5801 --
Your Label: 9
Correct Label: 9
Accuracy: 0.86071

-- Image #5901 --
Your Label: 4
Correct Label: 4
Accuracy: 0.86053

-- Image #6001 --
Your Label: 9
Correct Label: 9
Accuracy: 0.86102

-- Image #6101 --
Your Label: 4
Correct Label: 4
Accuracy: 0.86084

-- Image #6201 --
Your Label: 9
Correct Label: 9
Accuracy: 0.86164

-- Image #6301 --
Your Label: 4
Correct Label: 4
Accuracy: 0.86367

-- Image #6401 --
Your Label: 0
Correct Label: 0
Accuracy: 0.86471

-- Image #6501 --
Your Label: 5
Correct Label: 5
Accuracy: 0.86556

-- Image #6601 --
Your Label: 5
Correct Label: 5
Accuracy: 0.86411

-- Image #6701 --
Your Label: 4
Correct Label: 4
Accuracy: 0.86435

-- Image #6801 --
Your Label: 2
Correct Label: 2
Accuracy: 0.86414

-- Image #6901 --
Your Label: 6
Correct Label: 6
Accuracy: 0.86553

-- Image #7001 --
Your Label: 1
Correct Label: 1
Accuracy: 0.86673

-- Image #7101 --
Your Label: 2
Correct Label: 2
Accuracy: 0.86777

-- Image #7201 --
Your Label: 8
Correct Label: 8
Accuracy: 0.86905

-- Image #7301 --
Your Label: 7
Correct Label: 7
Accuracy: 0.86974

-- Image #7401 --
Your Label: 2
Correct Label: 2
Accuracy: 0.87123

-- Image #7501 --
Your Label: 8
Correct Label: 8
Accuracy: 0.87162

-- Image #7601 --
Your Label: 1
Correct Label: 1
Accuracy: 0.87186

-- Image #7701 --
Your Label: 9
Correct Label: 9
Accuracy: 0.87261

-- Image #7801 --
Your Label: 0
Correct Label: 3
Accuracy: 0.87335

-- Image #7901 --
Your Label: 1
Correct Label: 1
Accuracy: 0.87229

-- Image #8001 --
Your Label: 4
Correct Label: 4
Accuracy: 0.87252

-- Image #8101 --
Your Label: 1
Correct Label: 1
Accuracy: 0.87286

-- Image #8201 --
Your Label: 6
Correct Label: 6
Accuracy: 0.87367

-- Image #8301 --
Your Label: 8
Correct Label: 8
Accuracy: 0.87363

-- Image #8401 --
Your Label: 4
Correct Label: 4
Accuracy: 0.8743

-- Image #8501 --
Your Label: 4
Correct Label: 4
Accuracy: 0.87507

-- Image #8601 --
Your Label: 1
Correct Label: 1
Accuracy: 0.87606

-- Image #8701 --
Your Label: 9
Correct Label: 9
Accuracy: 0.87737

-- Image #8801 --
Your Label: 2
Correct Label: 2
Accuracy: 0.87854

-- Image #8901 --
Your Label: 9
Correct Label: 9
Accuracy: 0.87979

-- Image #9001 --
Your Label: 7
Correct Label: 7
Accuracy: 0.88079

-- Image #9101 --
Your Label: 3
Correct Label: 3
Accuracy: 0.88056

-- Image #9201 --
Your Label: 1
Correct Label: 1
Accuracy: 0.88088

-- Image #9301 --
Your Label: 7
Correct Label: 7
Accuracy: 0.88173

-- Image #9401 --
Your Label: 5
Correct Label: 5
Accuracy: 0.88288

-- Image #9501 --
Your Label: 2
Correct Label: 2
Accuracy: 0.88317

-- Image #9601 --
Your Label: 5
Correct Label: 5
Accuracy: 0.88355

-- Image #9701 --
Your Label: 1
Correct Label: 2
Accuracy: 0.88352

-- Image #9801 --
Your Label: 0
Correct Label: 0
Accuracy: 0.88297

-- Image #9901 --
Your Label: 8
Correct Label: 8
Accuracy: 0.88213

Percentage incorrect for each label:
3.141% of 0's were incorrect.
2.462% of 1's were incorrect.
13.667% of 2's were incorrect.
12.054% of 3's were incorrect.
12.394% of 4's were incorrect.
16.638% of 5's were incorrect.
5.518% of 6's were incorrect.
12.649% of 7's were incorrect.
14.601% of 8's were incorrect.
6.876% of 9's were incorrect.
final accuracy is: 88.22%

```
