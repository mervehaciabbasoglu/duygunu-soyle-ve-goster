import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Veri yolu
data_dir = 'konusmatanima'

# Etiketler
emotion_labels = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Ses dosyalarını ve etiketleri yükleme
def load_data(data_dir):
    x_data = []
    y_data = []
    for actor in os.listdir(data_dir):
        actor_dir = os.path.join(data_dir, actor)
        if os.path.isdir(actor_dir):
            for file in os.listdir(actor_dir):
                if file.endswith('.wav'):
                    file_path = os.path.join(actor_dir, file)
                    y, sr = librosa.load(file_path, sr=None)
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                    x_data.append(np.mean(mfccs.T, axis=0))
                    label = file.split('-')[2]  # Örnek olarak etiketin dosya adında üçüncü bölümde olduğunu varsayıyorum
                    y_data.append(emotion_labels[int(label) - 1])
    return np.array(x_data), np.array(y_data)

# Verileri yükleme
x_data, y_data = load_data(data_dir)

# Etiketleri sayısal değerlere dönüştürme
encoder = LabelEncoder()
y_data = to_categorical(encoder.fit_transform(y_data))

# Veriyi eğitim ve test setlerine ayırma
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(40, 1)),
    MaxPooling1D(2),
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(emotion_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=32)

# Modeli kaydetme
model.save('ses_tanima_modeli.h5')
