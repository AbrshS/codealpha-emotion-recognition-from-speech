

---

## 🎤 Emotion Recognition from Speech

A deep learning model that classifies human emotions (e.g., happy, sad, angry, fear, etc.) from speech audio using Convolutional Neural Networks (CNN).

---

### 📂 Dataset

This project uses **multiple open datasets**:

* RAVDESS
* CREMA-D
* TESS
* SAVEE

Each file is labeled with an emotion class such as:
`neutral`, `happy`, `sad`, `angry`, `fear`, `disgust`, `surprise`, `calm`.

---

### 🧠 Model

* **Input**: Extracted MFCC features from audio
* **Architecture**: 1D CNN with Conv → Pool → Dropout layers
* **Output**: Softmax layer with 8 emotion classes

---

### 🚀 How to Run

1. **Install requirements**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model**

   ```python
   python train_emotion_model.py
   ```

   > Includes callbacks to save the best model to `emotion_recognition_speech_model.h5`.

3. **Test the model**

   ```python
   python test_model.py
   ```

   This will:

   * Load `emotion_recognition_speech_model.h5`
   * Predict on test set
   * Print classification report
   * Show confusion matrix with predicted vs actual labels

---

### 🎧 Test with Custom Audio (optional)

Record your own `.wav` file and test it like this:

```python
data, sr = librosa.load("your_voice.wav")
features = extract_features(data, sr)
prediction = model.predict(features)
```
---

### 📁 Folder Structure

```
├── data/                 # Dataset folders (RAVDESS, CREMA, etc.)
├── model/                # Saved model .h5
├── train_emotion_model.py
├── test_model.py
├── utils.py              # Feature extraction, plotting
├── requirements.txt
└── README.md
```

---

### ✅ Future Improvements

* Real-time voice input using microphone
* Add more robust emotion classes
* Deploy with a web interface (e.g., Streamlit)

---
