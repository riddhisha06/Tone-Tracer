# 🎯 Tone Tracer — Sarcasm Detector for Text

**Tone Tracer** is an NLP-powered sarcasm detection system that classifies sentences and tweets as **sarcastic** or **regular** using **TF-IDF vectorization (n-grams)** and **Logistic Regression**.  
The model provides **real-time predictions with confidence scores**, and includes both **CLI** and **Streamlit Web App** support.

---

## 🧠 Model Details

| Component | Description |
|---------|-------------|
| Algorithm | Logistic Regression (with class balancing) |
| Vectorizer | TF-IDF (unigram + bigram) |
| Pre-processing | Lowercasing, removing URLs, mentions, hashtags, punctuation |
| Output Labels | `1 = Sarcastic`, `0 = Regular` |

---

## 🚀 Features

✔ Detect sarcasm in any text input  
✔ Real-time confidence scores for both sarcasm and non-sarcasm  
✔ Available via CLI and Web App (Streamlit)  
✔ Lightweight and fast — built with scikit-learn  

---

## 📂 Project Structure

```

Tone-Tracer/
│
├── train.csv                      # Dataset
├── requirements.txt               # Dependencies
│
├── model/
│   ├── sarcasm_model.pkl          # Trained model
│   ├── vectorizer.pkl             # TF-IDF vectorizer
│
├── src/
│   ├── train.py                   # Script to train the model
│   ├── predict.py                 # CLI prediction script
│
└── streamlit_app/
├── app.py                     # Streamlit web interface

````

---

## 🔧 Installation

Clone the repository:
```bash
git clone https://github.com/IshaanTripathi03/ToneTracer.git
cd ToneTracer
````

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 CLI Prediction

```bash
python src/predict.py "Oh, I absolutely love waking up at 5 AM for work..."
```

Sample Output:

```
Prediction: Sarcastic 😏
Confidence: 86.92%
```

---

### 🌐 Launch the Streamlit Web App

```
cd streamlit_app
streamlit run app.py
```

Features:

* Input any sentence or tweet
* Prediction result: **Sarcastic 😏 / Regular 🙂**
* Confidence percentage for both categories

---

## 📊 Dataset Overview

| Column   | Purpose        |
| -------- | -------------- |
| `tweets` | Text data      |
| `class`  | Category label |

Label mapping used in training:

```
sarcasm → 1
others  → 0
```

---

## 📌 Roadmap (Future Improvements)

🔹 Add deep-learning models (LSTM / BERT)
🔹 Deploy to Streamlit Cloud / Hugging Face Spaces
🔹 Display sarcasm rationale/explanation
🔹 Train on more diverse datasets

---

## 👨‍💻 Author

Developed by **Riddhisha Srivastava** along with my teammates - **Sachin Tripathi, Anshika Mishra, Ishaan Tripathi**

Contributions, forks, and improvements are welcome!

---

⭐ **If you like Tone Tracer, please consider giving the repository a star!**
