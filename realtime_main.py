import cv2
import numpy as np
import tensorflow as tf

# Modeli yükleme
model = tf.keras.models.load_model('yuz_tanima_modeli.h5')

# Etiketler
class_labels = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def preprocess_image(img):
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_expression(frame, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        processed_face = preprocess_image(face)
        prediction = model.predict(processed_face)
        label = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return frame

# Yüz tanıma için Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kamera başlatma
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = predict_expression(frame, model)
    cv2.imshow('Face Expression Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
