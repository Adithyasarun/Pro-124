import cv2
import tensorflow as tf
import numpy as np


# import the tensorflow modules and load the model


model = tf.keras.models.load_model("keras_model.h5")

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = camera.read()
    
    img = cv2.resize(frame,(224,224))
    test_img = np.array(img,dtype=np.float32)
    test_img = np.expand_dims(test_img,axis=0)
    normal_img = test_img/255.0

    predction = model.predict(normal_img )
    print("predction: ",predction)
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
camera.release()

# Destroy all the windows
cv2.destroyAllWindows()