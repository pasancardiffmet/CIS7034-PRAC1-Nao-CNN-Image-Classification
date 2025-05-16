# This section outlines a conceptual prototype for integrating the trained CIFAR-10
# demonstrate how to interact with the robot's camera and speech modules.
# 1.  **Nao Captures Image:** Use the Nao's camera to capture an image of an object.
# 2.  **Transfer Image:** Transfer the captured image data from the Nao to the computer running the CNN model.
# 3.  **Preprocess Image:** On the computer, load the captured image, resize it to 32x32 pixels, and normalize the pixel values to match the training data format.
# 4.  **CNN Prediction:** Load the trained CNN model weights and use the model to predict the class of the preprocessed image.
# 5.  **Nao Reacts:** Send the prediction result back to the Nao robot. The Nao can then announce the predicted class name using its text-to-speech capabilities or perform a corresponding action.

import time
from PIL import Image
from naoqi import ALProxy
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
import os
import cv2
# Loading the model weights saved from the training step
model_weights_path = 'my_model_.weights.h5'
cifar10_class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Re-define the model architecture (same as in the trained model)
# Define the CNN model
model = Sequential()

# Convolutional Block 1q
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Convolutional Block 2
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Convolutional Block 3
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


# Flattening and Dense Layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# Load the trained weights into the model
model.load_weights(model_weights_path)
print "Successfully loaded model weights for Nao integration."



# Nao Robot Connection
nao_ip = "172.18.16.45"
nao_port = 9559

# Naoqi Service Proxies
video_service = None
tts_service = None
try:
    # Connect to Naoqi services
    video_service = ALProxy("ALVideoDevice", nao_ip, nao_port)
    tts_service = ALProxy("ALTextToSpeech", nao_ip, nao_port)
    print "Connected to Naoqi services."
    # --- Nao Introduction ---
    # Set volume to be loud (value between 0.0 and 1.0)
    initial_volume = tts_service.getVolume() # Save current volume
    tts_service.setVolume(2) # Set a high volume
    # Nao introduces itself and the demonstration
    intro_text = "Hello, I'm Nao. This is an image classification demonstration from the CIFAR 10 dataset. This uses a custom CNN Model"
    print "Nao is about to say: {}".format(intro_text)
    tts_service.say(intro_text)
    tts_service.say("Presented by Pasan Jayaweera Student ID-st20319008")
    tts_service.say("The image classes I can recognise are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck ")
    print "Nao finished introduction."

    # --- Step 1 & 2: Nao Captures and Transfers Image ---

    # Subscribe to the Nao's camera (Top camera, VGA resolution, RGB color space, 5 fps)
    camera_client_name = "cifar10_classifier_client"
    resolution = 2 # VGA (640x480)
    colorSpace = 11 # RGB
    fps = 5
    camera_index = 0 # Top camera

    print "Subscribing to Nao camera..."
    video_client = video_service.subscribeCamera(
        camera_client_name, camera_index, resolution, colorSpace, fps
    )
    print "Camera subscribed."

    # Get a single image frame
    print "Getting image from Nao camera..."
    nao_image = video_service.getImageRemote(video_client)
    print "Image captured."

    # # Unsubscribe from the camera
    # video_service.unsubscribe(video_client)
    # print "Camera unsubscribed."

    # --- Continuous Classification Loop ---
    print "\nStarting continuous image capture and classification. Press 'q' on the image window to exit."
    while True:
        # Get a single image frame
        nao_image = video_service.getImageRemote(video_client)
        #Check if image capture was successful
        if nao_image is None:
            print "Warning: Failed to capture image from Nao. Skipping this frame."
            time.sleep(0.1) # Add a small delay before trying again
            continue # Skip the rest of the loop and try to get the next frame
        image_width = nao_image[0]
        image_height = nao_image[1]
        image_array = nao_image[6] # Raw image data
        image_string = str(bytearray(image_array)) # Convert byte array to string for PIL

        # Create a PIL Image from the raw data

        pil_image = Image.frombytes("RGB", (image_width, image_height), image_string)

        #Display Captured Image using OpenCV
        if pil_image:
            # Convert PIL image to OpenCV format (BGR) for display
            # Naoqi colorSpace 11 is RGB, OpenCV expects BGR for imshow
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Display the image
            cv2.imshow("Nao Camera Feed", cv_image)

            # Preprocess Image for CNN
            # Resize the image to 32x32 pixels
            # Use LANCZOS resampling for high-quality downsampling
            try:
                resized_image = pil_image.resize((32, 32), Image.LANCZOS)
            except AttributeError:
                try:
                     resized_image = pil_image.resize((32, 32), Image.ANTIALIAS)
                except AttributeError:
                     resized_image = pil_image.resize((32, 32))


            # Convert PIL image to a NumPy array
            img_array = np.array(resized_image)

            # Normalize the pixel values
            img_array = img_array.astype('float32') / 255.0

            img_array = np.expand_dims(img_array, axis=0) # Shape becomes (1, 32, 32, 3)


            #Step 4: CNN Prediction
            # Make a prediction using the loaded model
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_name = cifar10_class_names[predicted_class_index]
            print "Predicted class name: {}".format(predicted_class_name)

            # Nao speaks the predicted class name
            nao_response = "I see a {}.".format(predicted_class_name)
            tts_service.say(nao_response)

        # Check for key press to exit the loop
        # cv2.waitKey(1) waits for a key event for 1ms
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): # Exit if 'q' is pressed
            print "Exiting continuous capture loop."
            break

        # Add a small delay to control the loop speed
        time.sleep(1.5)  # Adjust as needed


except RuntimeError as e:
    print "Could not connect to Naoqi. Please check the IP, port, and ensure Naoqi is running. Error: {}".format(e)
except Exception as e:
    print "An error occurred during Nao integration: {}".format(e)
