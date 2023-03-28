# Age-Regression-from-Brain-MRI
## Overview
This project was completed as part of the Machine Learning for Imaging course at Imperial College London. The objective of the coursework was to implement two supervised learning approaches for age regression from brain MRI scans. The goal of age prediction from MRI scans is to detect structural changes in the brain that may indicate the presence of disease. We were provided with MRI scans of a total of 652 healthy subjects, split into different development sets and a hold-out test set on which we evaluated the final prediction accuracy.

The first approach involved regressing the age of a subject using the volumes of brain tissues as features. The second approach was to use a deep learning model to learn the features from the MRI scans directly. The two approaches required different processing pipelines with different components that we had to implement using methods that were discussed in the lectures and tutorials.

## Data
We were provided with MRI scans of a total of 652 healthy subjects, along with their age and gender information. The data was split into different subsets for different parts of the coursework. There was a set of 52 subjects to be used in part A to develop an image segmentation method, 500 subjects for training and two-fold cross-validation of age regression approaches in parts A and B, and a remaining set of 100 subjects to test the final age prediction accuracy.

## Part A: Age prediction using brain tissue segmentation and volume features
The first approach aimed to regress the age of a subject using the volumes of brain tissues as features. To this end, we implemented a four-class brain segmentation for grey matter (GM), white matter (WM), cerebrospinal fluid (CSF), and background. The segmentation method was trained using a total of 52 subjects and then applied to the remaining 600 brain scans, which were used to train and test the age regression.

We explored different regression techniques and investigated the best set of features for this task. We computed relative volumes as the ratios between each tissue volume and overall brain volume. We compared the performance of simple linear regression with a model with higher order polynomials and evaluated other regression methods such as regression trees, SVMs, and neural networks. We evaluated the accuracy of different methods using two-fold cross-validation on the set of 500 subjects and compared and reported the average age prediction accuracy appropriately.

### Task A-1: Brain tissue segmentation
The segmentation model takes the form of a 3D U-Net which follows an encoder-decoder scheme. The encoder component consists of four levels of blocks, each with two convolutional layers, batch normalization, leaky ReLU activation functions, and one max pooling layer. The decoder follows the same outline, with the exception of using transpose convolutional layers. From the encoding layers skip connections are used to bridge between the corresponding layers in the decoder.

### Task A-2: Feature Calculation
The segmentation model from A-1 is performed on a test set of 500 images, and thereafter features for each tissue class are computed. The engineered features include the absolute tissue volume and relative tissue volume per class. However, on account of being significantly more discriminative only the relative tissue volumes are used as input to the subsequent regression models.

### Task A-3: Age regression and cross-validation
The features are then fed into six regression methods: linear regression, ridge regression, lasso regression, k-NN, random forest regression, and support vector machines. Two fold-cross validation with random shuffling was used to train each regression model, with mean squared error, mean absolute error and R2 error metrics as the evaluation and performance criteria. Random forest regression yielded the best performance, and was thus selected as the final regression algorithm.

### Task A-4: Final Test on Hold-Out Data
The pipeline outlined in sections A-1, A-2, and A-3 was recomputed on the 500 element hold-out training set, and then tested on the 100 element hold-out test set with the same hyperparmeters. This resulted in a final mean square error score of 70.84, a mean absolute error of 6.59, and an R-2 score of 0.81.

## Part B: Age prediction using deep learning
The second approach was to use a deep learning model to learn the features from the MRI scans directly. We explored different architectures such as VGG-16, InceptionV3, and ResNet50, and compared their performance using two-fold cross-validation on the set of 500 subjects. The final model takes the form of a ResNet50, consisting of 50 convolutional layers distributed over 16 blocks, with ReLU activations. 

## Results
Our results showed that the deep learning approach outperformed the brain tissue segmentation approach. We achieved an average age prediction error of 6.21 years using the deep learning approach, compared to 6.59 years using the brain tissue segmentation approach. Our best-performing model was the ResNet50 architecture.




