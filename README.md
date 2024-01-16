One-Handed Sign Language Recognition Project

Description
The One-Handed Sign Language Recognition Project is a machine learning-based solution that allows users to train a model to recognize one-handed sign language gestures and convert them into text. 
It can be a valuable tool for individuals with disabilities who use sign language as an alternative to a keyboard for text input.

The project utilizes computer vision and machine learning techniques, including the MediaPipe library, to detect hand landmarks. 
Users should train a custom model to classify one-handed sign language gestures since MediaPipe currently recognizes only one hand.

The recognized gestures are then converted into text and displayed on the console.

Getting Started

Prerequisites

Before you begin, ensure you have met the following requirements

Python 3.x installed.
Required Python packages installed (you can install them using pip):
opencv-python: For computer vision tasks.
mediapipe: For hand landmark detection.
scikit-learn: For training and using the machine learning model.

Usage
Training the Model
Collect Training Data:

Collect one-handed sign language gesture images using your camera.
Organize the images into subdirectories by class in the ./data directory.
Adjust the number_of_classes and dataset_size in the train_model.py script to fit your needs.
Train the Model:

Run the training script:

bash
Copy code
python collect_imgs.py
The trained model will be saved as 'model.p'.

Recognizing One-Handed Sign Language Gestures
Run the Inference Script:

Start the one-handed sign language recognition script:

python inference_classifier.py
Recognize Gestures:

Sign with one hand in front of your camera, and the recognized gestures will be displayed on the console.
Customize Gesture Mapping:

Customize the labels_dict dictionary in the inference_classifier.py script to map gesture class indices to specific letters or actions.
