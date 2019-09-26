# CIFAR-10 Classification using Support Vector Machines

## How to use
1. Download CIFAR-10 from https://www.cs.toronto.edu/~kriz/cifar.html
2. Unpack in the same folder as the repository
3. Use flags in cifar.py to train/test model

## Methodology
First the training set undergo the HoG feature detection. The output vector from the previous step is then applied to the SVM along with the labels.

Results are calculated based on the traning set that is also first transformed using HoG descriptor.

## Results

The results are not outstanding compared to the other algorithms mentioned in the benhmark page (http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html). 

Total error on the test set is 49.83%

* Category: automobile, TP: 685, FP: 315, Total: 1000, error: 31.5
* Category: bird, TP: 477, FP: 523, Total: 1000, error: 52.300000000000004
* Category: cat, TP: 305, FP: 695, Total: 1000, error: 69.5
* Category: deer, TP: 374, FP: 626, Total: 1000, error: 62.6
* Category: dog, TP: 302, FP: 698, Total: 1000, error: 69.8
* Category: frog, TP: 536, FP: 464, Total: 1000, error: 46.400000000000006
* Category: horse, TP: 426, FP: 574, Total: 1000, error: 57.4
* Category: ship, TP: 630, FP: 370, Total: 1000, error: 37.0
* Category: truck, TP: 700, FP: 300, Total: 1000, error: 30.0
* Category: All, TP: 5017, FP: 4983, Total: 10000, error: 49.830000000000005

# Credits
Load cifar routine taken from https://github.com/snatch59/load-cifar-10

Some of the SVM code for learning was taken from the OpenCV course
https://opencv.org/courses/
