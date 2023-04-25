# import the opencv library
import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('keras_model.h5')
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    img = cv2.resize(frame, (224,224))
    #Converting the Image
    test_img = np.array(img, dtype = np.float32)
    test_img = np.expand_dims(test_img, axis = 0)
    normalised_img = test_img/255.0

    prediction = model.predict(normalised_img)
    print(prediction)
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()