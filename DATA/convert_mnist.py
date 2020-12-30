# modified from https://pjreddie.com/projects/mnist-in-csv/

def convert(imgf, labelf, out1f, out2f, n):
    f = open(imgf, "rb")
    o1 = open(out1f, "w") # images
    o2 = open(out2f, "w") # labels
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):

        # one hot encoding labels
        num = ord(l.read(1))
        for i in range(10):
            if(i == num):
                o2.write("1")
            else:
                o2.write("0")
            if(i != 9):
                o2.write(",");
        o2.write("\n")

        image = []
        for j in range(28*28):
            image.append(ord(f.read(1))/float(256)) #normalize
        images.append(image)

    for image in images:
        o1.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o1.close()
    o2.close()
    l.close()

convert("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
        "train_images.csv", "train_labels.csv", 60000)
convert("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
        "test_images.csv", "test_labels.csv", 10000)

