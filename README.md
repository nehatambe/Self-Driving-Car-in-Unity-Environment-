# Self-driving-car
The project demonstrates how to train a virtual car to drive autonomously in Unity simulator using reinforcement learning and random forest model. 

# Installation 
Install Python, Unity, and Anaconda. 
Install the required libraries.

# Files 
1.	drive.py – This file implements socket communication and it is used to connect with Unity Environment. In autonomous mode the car predicts next action using q_learning_model.h5 and pos_detect_model.h5 file. 
2.	check_and_train.py – Code written in this file saves recorded images into an info.CSV file and trains pos_detect_model and q_learning_model. Trained models are saved into .h5 file. 
3.	action.py- This class defines several actions and calculate their values. 
4.	q_learning_model.py – In this file neural network is built and trained using reinforcement learning. States and actions of recorded images are stored using ReplayMemory class. 
5.	data_utils.py – This file contains utility methods such as loading data, image processing etc.
6.	pos_detect_model.py – In this file neural network is built and trained using recorded images to predict the position of a car.

# Model training and running 

1.	Install python, unity, and download Git lfs for the udacity environment.
2.	After installation, you need to enable it on your machine: git lfs install
3.	Next, download developers code from GitHub. 
https://github.com/suoeryu/CPSC587Project
4.	Then create a python environment used by the model using conda:  	
cd car-behavioral-cloning
conda env create -f environments.xml
5.	Activate car-behavioral-cloning using following commands:
a.	$ activate car-behavioral-cloning
6.	Start the simulator, and a startup screen pops up. 

To run the car in autonomous mode, run following command
drive.py -m drive 

To record and train model, follow below steps 

7.	Create a folder for storing images.
8.	Go to the terminal and run following command to record images in training mode. 
python drive.py -s “<Image folder path>”
9.	Update the image folder path in check_and_train.py line 24.
10.	Run check_and_train.py to train the models.
11.	Run the car in an autonomous mode using following command
drive.py -m drive 

# References
The code is derived from car-behavioral-cloning. Thank a lot Naoki Shibuya!

https://github.com/naokishibuya/car-behavioral-cloning

https://github.com/udacity/self-driving-car-sim





