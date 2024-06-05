import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import tensorflow as tf
import librosa
import sounddevice as sd
import soundfile as sf
import os

# Modeli yükleme
model = tf.keras.models.load_model('ses_tanima_modeli.h5')

# Etiketler
emotion_labels = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def record_audio(duration, fs, filename):
    print("Recording...")
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    sf.write(filename, myrecording, fs)
    print("Recording complete")

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)[..., np.newaxis]

def predict_emotion(file_path, model):
    features = preprocess_audio(file_path)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    label = emotion_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return label, confidence

def start_recording():
    duration = int(duration_entry.get())
    filename = 'test.wav'
    record_audio(duration, 44100, filename)
    messagebox.showinfo("Info", "Recording complete")

def play_audio():
    filename = 'test.wav'
    if os.path.exists(filename):
        data, fs = sf.read(filename, dtype='float32')
        sd.play(data, fs)
        sd.wait()
    else:
        messagebox.showerror("Error", "No recording found")

def analyze_audio(filename='test.wav'):
    if os.path.exists(filename):
        label, confidence = predict_emotion(filename, model)
        result_label.config(text=f"Predicted Emotion: {label} ({confidence:.2f}%)")
    else:
        messagebox.showerror("Error", "No recording found")

def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        analyze_audio(file_path)

# GUI oluşturma
root = tk.Tk()
root.title("Emotion Recognition from Speech")

tk.Label(root, text="Duration (seconds):").grid(row=0, column=0)
duration_entry = tk.Entry(root)
duration_entry.grid(row=0, column=1)
duration_entry.insert(0, "5")

record_button = tk.Button(root, text="Record", command=start_recording)
record_button.grid(row=1, column=0, columnspan=2)

play_button = tk.Button(root, text="Play", command=play_audio)
play_button.grid(row=2, column=0, columnspan=2)

analyze_button = tk.Button(root, text="Analyze", command=analyze_audio)
analyze_button.grid(row=3, column=0, columnspan=2)

upload_button = tk.Button(root, text="Upload & Analyze", command=upload_file)
upload_button.grid(row=4, column=0, columnspan=2)

result_label = tk.Label(root, text="")
result_label.grid(row=5, column=0, columnspan=2)

root.mainloop()
