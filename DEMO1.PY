import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pneumonia detection model
model = load_model("/home/harikrishnan/VSCode/OpenCV/X_RAY/pneumonia.h5")

def predict_pneumonia(frame):
    input_shape = (224, 224)
    img = cv2.resize(frame, input_shape)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0

    prediction = model.predict(img)
    pneumonia_probability = prediction[0][0]

    return pneumonia_probability

video_capture = cv2.VideoCapture('/home/harikrishnan/VSCode/OpenCV/X_RAY/WhatsApp Video 2023-11-27 at 10.00.22 PM.mp4')

while True:
    success, frame = video_capture.read()

    pneumonia_probability = predict_pneumonia(frame)

    if pneumonia_probability > 0.5:
        prediction_text = 'Pneumonia Detected'
    else:
        prediction_text = 'Normal'

    final_frame = cv2.putText(frame, text=prediction_text, org=(100, 200),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0), fontScale=2)
    cv2.imshow("Pneumonia Detection", final_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
