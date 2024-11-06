# DL-Bench-Data

Here is the data and code for DLEval project.
You can find the data and types and classifications in the .zip files
DLEval
Categories of Code Stages and Tasks in Machine Learning
Pre/post Processing
Code in the pre or post-processing stage often manipulates data (input or output). For example, pre-processing code cleans or augments input data, whereas post-processing code augments output data for visualization. Due to the ambiguity at the function level, we have a combined category for pre and post-processing code [Wen et al., 2020].

Model Construction
This stage defines the network architecture and sets up the computational graph for deep learning models, including defining layers, activation functions, and layer connections. Examples include defining CNN architectures and forward pass logic. Loss functions are part of this stage, but optimization steps are in the training phase [Howard et al., 2019].

Training
The training stage optimizes the model's parameters using a loss function and optimization algorithm. This includes backpropagation and weight updates. Code for gradient descent using optimizers like Adam or SGD and looping over epochs and batches falls under this stage [Diederik et al., 2014].

Inference
Inference code is used to generate labels based on a trained model. It processes new input data and outputs results, such as classifications or detections, without changing model parameters. This stage emphasizes speed and efficiency for real-time deployment [Kirillov et al., 2019].

Evaluation & Metrics
Code in this stage assesses the performance of a trained model using various metrics. It involves running the model on a validation/test dataset and comparing predictions to ground truth labels to measure accuracy, precision, recall, F1-score, etc. [Wu et al., 2020].

Categories of Machine Learning Tasks
Classification
Classification tasks involve assigning input data to categories or classes. For example, models using softmax activation in the final layer for outputs like "dog" or "cat" fall under this category. Categorical cross-entropy loss is a common indicator.

Regression
Regression tasks predict continuous values. Code indicating regression tasks often has linear activation functions in the final layer.

Object Detection
Detection tasks identify objects and their locations within images. Code that outputs bounding boxes and class labels (e.g., YOLO, Faster R-CNN) and employs anchor boxes or non-maximum suppression is indicative of detection tasks.

Image Segmentation
Segmentation tasks assign labels to each pixel in an image. Code involving semantic or instance segmentation (e.g., U-Net, Mask R-CNN) where the output is a mask with pixel-level classifications is a common example.

Time Series Prediction
These tasks forecast future values using historical data. Code involving recurrent neural networks (RNNs), LSTM, GRU models, and loss functions like mean absolute error (MAE) or MSE is typical.

Recommendation
Recommendation tasks suggest items or actions based on user data. Code implementing collaborative or content-based filtering algorithms, matrix factorization, or deep learning-based models for recommendations falls into this category.

General
Code that is versatile and applicable to multiple ML tasks without being exclusive to a specific one is labeled as General.

Input Types in Machine Learning
Image
Processing for image data includes steps like resizing, normalization, and data augmentation. Code that resizes images (e.g., 224x224 for CNNs), normalizes pixel values, or applies augmentations (flipping, cropping, noise addition) typically signals image data [Krizhevsky et al., 2012].

Text
Text processing involves tokenization, n-gram generation, stemming, lemmatization, and embeddings. Code that handles these processes and converts text into vectors (e.g., using TF-IDF, Word2Vec, BERT) indicates text data [Liu et al., 2018].

Structured Array
Tabular data, where rows represent data points and columns represent features, is processed by normalization, one-hot encoding, or handling missing values. Code that reads CSVs into DataFrames and applies these techniques indicates structured array data, commonly used in regression or classification tasks [Chen et al., 2016].

Others
When input data does not match typical types (image, text, structured array), it is labeled as Others. This includes input such as model parameters or hyperparameters. For example, def __init__(self, weight, bias=None) initializing model components without direct input data processing falls under this label.
