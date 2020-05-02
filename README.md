# ContextBasedImageRetrieval
A context based image retrieval application using CNN features

## Overview
This application allows the user to search for images similar to a query image. The program uses Computer Vision techniques to ananlyze the contents of the image and retrieves similar images from a large image database based on its content.  

## Specifications
1. The program loads a pre-trained convolutional neural network trained on the 'Imagenet' dataset.   
2. The last layer from the neural network is separated to extract image features.   
3. Principle Componenet Analysis (PCA) is used to reduce the feature vector size.  
4. These features and the PCA object are then saved into a file.  
5. When the program is launched, the saved features are loaded along with the PCA object.   
6. When the user selects an image, the features for that query image are extracted.  
7. Euclidean distance between the query image features and the loaded features is calculated and stored in an array.  
8. Finally, the array is sorted and the images corresponding to the first 5 features are selected to show.  

## How to run:
1. Install the libraries and dependies given in requirements.txt.  
2. Open the program in your favorite Python IDE and run.  

## Controls:
Click on the 'select image' button to select an image.  
Click on the 'view matches' button to view the matching images.  