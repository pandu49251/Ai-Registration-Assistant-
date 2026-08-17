# 🤖 Smart AI Registration Assistant

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey)

An intelligent, fully automated conversational AI agent designed to streamline student internship applications. Built for the **FreeInternships.in** program, this project replaces tedious manual forms with a smart, context-aware chatbot.

It features a **Jupyter Lab-style auto-launcher**—simply run the script, and it automatically finds a free port and opens the beautiful web UI in your default browser!

---

## ✨ Features

### Core Requirements Satisfied:
- ✅ **NLP Preprocessing:** Uses NLTK for Tokenization and WordNet Lemmatization.
- ✅ **Intent Classification:** Powered by a Scikit-Learn `TfidfVectorizer` and `MultinomialNB` (Naive Bayes) model.
- ✅ **Dialog Management:** Context-aware Finite State Machine handles multi-step registration (Name → Email → Field).
- ✅ **Entity Extraction & Validation:** Uses RegEx to validate email formats mid-conversation.
- ✅ **Persistent Storage:** Saves validated applications to a local `registrations.json` database.

### Bonus Features Included:
- ⭐ **Sentiment Analysis:** Uses NLTK `VADER` to detect frustrated users and respond with empathy.
- ⭐ **FAQ Handling:** Answers questions regarding eligibility, duration, and bot identity.
- ⭐ **Admin Dashboard:** Built-in dashboard (accessible via the UI) to view all registered students.
- ⭐ **Analytics & Logging:** All chats, intents, and responses are safely logged to `chat_logs.txt`.
- ⭐ **Zero-Config Launch:** Automatically finds an open network port and launches your web browser.

---

## 🛠️ Technology Stack

* **Language:** Python
* **Natural Language Processing:** NLTK
* **Machine Learning:** Scikit-Learn (TF-IDF, Naive Bayes)
* **Web Server:** Flask, Werkzeug
* **Frontend:** HTML5, CSS3, JavaScript (Fetch API)

---

## 🚀 How to Run (Hassle-Free)

**1. Clone the repository:**
```bash
git clone https://github.com/YOUR-USERNAME/Smart-AI-Registration-Assistant.git
cd Smart-AI-Registration-Assistant
```

**2. Install the required dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the Assistant:**
```bash
python smart_assistant.py
```

*That's it! The terminal will safely load the AI models in the background and instantly open the chatbot in your web browser.*

---

## 🧠 How the AI Works

1. **Preprocessing:** When a user sends a message, it is converted to lowercase, tokenized into individual words, and lemmatized to its dictionary root.
2. **Sentiment Check:** The raw text is passed through VADER to check for negative compound polarity.
3. **State Machine Bypass:** If the bot is actively asking for registration details (e.g., waiting for an email), it bypasses general prediction and applies specific RegEx validation rules to the input.
4. **Intent Prediction:** If the user is just chatting, the text is vectorized and passed to the Naive Bayes ML model to predict what the user wants to know (e.g., `faq_duration`, `register`).

---

## 📂 Project Structure

```text
📁 Smart-AI-Registration-Assistant/
│
├── smart_assistant.py       # Main application (AI Engine + Web Server + UI)
├── requirements.txt         # Python dependencies
├── Final_Project_Report.pdf # Comprehensive Project Report
├── architecture.png         # Flowchart of the AI logic
└── README.md                # This file
```

*(Note: `registrations.json` and `chat_logs.txt` will be automatically generated upon first run).*

---
**Prepared for FreeInternships.in**
