# Portrait Paintings - GANs
Implementing a DCGAN model with feature matching to generate 64x64 portrait paintings.

To run this project, simply install the code and follow the link provided in data/dataset.txt
to download the dataset and place it in your local data folder.

Use the method provided in the data/preprocessing.py file to process the images, these methods include:
 - resizing
 - cropping faces
 - renaming files

Lastly once you have a proper dataset of 64x64 images you can run the train.py file to train the model
and view the results live on your tensorboard.

Further details on this project are presented in the following article: https://levelup.gitconnected.com/painting-portraits-using-gans-with-pytorch-afeb69b1c5a1
