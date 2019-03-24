# car-detector
A convolutional neural network that detects the cars and their colors in aerial images. It uses a simple Python script with some Tensorflow functions. If you are learning Tensorflow, this can be your code!

# Content
Here you can find the following files and folders:
- CarDetector.py: Try to create, train and use the neural network!
- CarDetectorSolved.py: Here is a solution with good performance.
- Images: The dataset used to train the neural network.

# Datasets
The dataset consists of aerial images with cars of different colors obtained from the simulator of [SwarmCity Project]{http://www.swarmcityproject.com/}. Here you can find 95 images to try the neural network, but we have more than 10,000 to perform a better training. Contact us if you need these images!

# Scripts
You can try the code just doing the following things:

main: 
- Set the train variable to 'true' if you want to train the model and 'false' if you want to make predictions.
- Set the paths for the dataset and model.

main_train:
- Define the architecture of the convolutional neural network.
- Create the placeholders for the input and output variables.
- Initialize the parameters of the different layers.
- Define the cost function for training the neural network.
- Run the session to train the neural network, providing the inputs and desired outputs and requesting the training error.

main_predict:
- Run the session to make predictions with the neural network, providing the inputs and requesting the outputs.

# Contact
Juan Jesús Roldán Gómez - PhD on Automation and Robotics - www.jjrg.org - jj.roldan.gomez@gmail.com.
