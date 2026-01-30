from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
import sys


# Load trained model
model = tf.keras.models.load_model(
    r'C:\Users\KyawNyi\Desktop\ML\code\best_model.keras'
)
print("Model loaded!")


# Model configuration
class_names = ['lemon', 'orange']
THRESHOLD = 0.7
IMG_SIZE = (128, 128)


# Prediction function
def import_and_predict(image_data, model):
    # Resize
    image = image_data.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    image = image.convert('RGB')

    image = np.asarray(image).astype(np.float32) / 255.0
    image = image[np.newaxis, ...]

    prediction = model.predict(image, verbose=0)[0]
    return prediction


# Open webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Failed to open camera.")
    sys.exit()

print("Camera opened successfully.")
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH),
      cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    """        
    #Fix webcam color bias (CLAHE)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    """
    #Center square ROI (remove bg)
    h, w, _ = frame.shape
    s = min(h, w)
    y1 = (h - s) // 2
    x1 = (w - s) // 2
    roi = frame[y1:y1 + s, x1:x1 + s]

    #Convert to PIL
    rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    #Predict
    prediction = import_and_predict(pil_image, model)
    confidence = np.max(prediction)
    class_index = np.argmax(prediction)
    predicted_label = class_names[class_index]

    # Debug
    # print("Prediction:", prediction)

    #Confidence-based UNKNOWN
    if confidence < THRESHOLD:
        label_text = f"UNKNOWN ({confidence:.2f})"
        color = (0, 0, 255)
    else:
        label_text = f"{predicted_label.upper()} ({confidence:.2f})"
        color = (0, 255, 0)

    #Display
    cv2.putText(
        frame,
        label_text,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2
    )

    cv2.imshow("Lemon vs Orange (Live)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Camera closed.")
