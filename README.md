# 🚀 AI Social Media Assistant

An AI-powered web application that helps users optimize their social media posts by predicting performance and generating engaging captions and hashtags.

---

## 📌 Project Overview

This project combines **Machine Learning** and **Generative AI** to assist users in improving their social media content.

The system provides:
- 📊 Post performance prediction (Low / Medium / High)
- 🤖 AI-generated captions
- 🏷 AI-generated hashtags

---

## 🎯 Problem Statement

Users struggle to predict how well their social media posts will perform before publishing. Creating engaging captions and selecting effective hashtags is also difficult.

This project solves these problems using AI.

---

## 💡 Proposed Solution

The system includes:

### 1. Machine Learning Model
- Predicts performance using:
  - Follower count
  - Caption length
  - Hashtags count
  - Post timing
  - Call-to-action (CTA)

### 2. Generative AI
- Generates captions and hashtags using OpenAI API

### 3. Web Application
- Built using Django for real-time interaction

---

## 🛠 Tech Stack

- Python
- Django
- HTML, CSS, JavaScript
- Scikit-learn (Random Forest)
- OpenAI API
- Pandas, NumPy

---

## 🧠 Model Details

- Algorithm: Random Forest Classifier
- Accuracy: ~55%–60%
- Balanced dataset using resampling
- Medium class has natural overlap (real-world scenario)

---

## ⚠️ IMPORTANT NOTE (MODEL FILE ISSUE)

The `.pkl` model files are NOT uploaded because GitHub does not allow large files (>100MB).

You must generate them manually using the notebook.

---

# 🔧 HOW TO RUN THIS PROJECT (FULL STEPS)

---

## 🔹 Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/ai-social-media-assistant.git
cd ai-social-media-assistant


Step 2: Install Required Libraries
pip install -r requirements.txt

🔹 Step 3: Run Jupyter Notebook
jupyter notebook

🔹 Step 4: Generate Model Files

After running notebook, these files will be created:

final_social_media_model.pkl
model_columns.pkl
🔹 Step 5: Move Model Files

Place them inside:

ai_social_media/core/
🔹 Step 6: Add OpenAI API Key

Open:

core/views.py

Replace:

client = OpenAI(api_key="YOUR_API_KEY")

🔹 Step 7: Run Django Server
python manage.py runserver

Open browser:

http://127.0.0.1:8000/
