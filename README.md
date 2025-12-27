# Personality-Aware Non-verbal Behavior Generation in Dyadic Interactions

**Master Thesis Project**
[Amr Amer](mailto:your.email@example.com) | [LinkedIn](https://linkedin.com/in/yourprofile) | [Portfolio](https://yourportfolio.com)

---

## 📌 Overview
This project presents a **transformer-based architecture** that generates non-verbal behavior for listener avatars in dyadic interactions, conditioned on personality traits. The model uses multimodal inputs (audio, body language, hand gestures) and achieves state-of-the-art results on the UDIVA dataset.

**Key Features:**
- Personality-aware listener avatar generation
- Multimodal input fusion (audio, motion, personality)
- State-of-the-art performance on UDIVA dataset
- Applications in virtual therapy, customer service, and personalized assistants

---

## 🚀 Demo
![Demo GIF](assets/Images/final-avatars.gif)

**Try the live demo:**
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/yourusername/your-thesis-repo/main)

---

## 📊 Results
| Model                          | Face FID ↓ | Body FID ↓ | P-FID ↓ | Variance ↑ |
|--------------------------------|------------|------------|---------|------------|
| Personality-agnostic Baseline  | 7.67       | 58.87      | 96.82   | 0.97       |
| **Ours**                      | **6.15**   | **43.16**  | **87.73**| 1.03       |

- **User Study:** 86% accuracy in distinguishing extroverted vs. introverted avatars
- **Qualitative Results:** Extroverted avatars show more smiling, leaning, and dynamic gestures

---

## 🛠️ Installation
```bash
git clone https://github.com/yourusername/your-thesis-repo.git
cd your-thesis-repo
pip install -r requirements.txt
streamlit run src/streamlit_app.py
