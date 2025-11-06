<div align="center">

# 📘 StudyMate  
### 💡 An AI-Powered PDF-Based Q&A System for Students

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)]()
[![IBM Watsonx](https://img.shields.io/badge/IBM%20Watsonx-Mixtral--8x7B-blueviolet)]()
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)]()


</div>

---

## 🧠 Overview

*StudyMate* is an *AI-powered academic assistant* that allows students to *interact with their study materials* — textbooks, lecture notes, and research papers — in a *conversational Q&A format*.

Instead of scrolling through long PDFs or manually searching for key points, users can simply *upload their documents* and *ask natural-language questions*.  
StudyMate returns *direct, well-contextualized answers*, referenced from the actual PDF content.

> 🎯 *Mission:* To make learning smarter, faster, and interactive using AI-powered context-based understanding.

---

## 🧩 Core Features

| # | Feature | Description |
|---|----------|-------------|
| 🗣 1 | *Conversational Q&A from Academic PDFs* | Ask natural-language questions and receive contextual answers derived directly from your uploaded materials. |
| 🧾 2 | *Accurate Text Extraction & Preprocessing* | Uses PyMuPDF to extract, clean, and chunk text from multiple PDFs efficiently. |
| 🔍 3 | *Semantic Search (FAISS + Embeddings)* | Leverages SentenceTransformers embeddings and FAISS to fetch the most relevant text chunks. |
| 🧠 4 | *LLM-Based Answer Generation* | Employs *IBM Watsonx’s Mixtral-8x7B-Instruct* model for reliable, fact-grounded answer generation. |
| 🖥 5 | *Streamlit Interface* | User-friendly, local web interface for document upload, question input, and result visualization. |

---

## 🌟 Extended AI Features

We went beyond just Q&A — making StudyMate an *all-in-one learning ecosystem*:

### 🗣 *AI Voice Assistant*
- Reads out answers aloud with *Text-to-Speech*.  
- Enables *voice question input* for hands-free learning.  
- Great for accessibility and auditory learners.

### ⏳ *Pomodoro Timer*
- Built-in *focus timer* to help students manage study/break intervals.  
- Encourages productivity using the *Pomodoro Technique* (25/5 cycles).  
- Integrated session summaries for review after each timer.

### 📝 *Sticky Notes*
- Add, edit, and save personal notes while studying.  
- Notes can be linked to specific PDFs or answers and exported later.  
- Perfect for quick revision or flashcard creation.

### 🤖 *Smart Trained Chatbot*
- Context-aware chatbot for multi-turn conversations.  
- Remembers previous questions within the same session.  
- Acts like a personal *AI tutor* for deeper concept exploration.

---

## ⚙ Tech Stack

| Category | Tools & Libraries |
|-----------|-------------------|
| *Language* | Python |
| *Frontend* | Streamlit |
| *AI Model / LLM* | IBM Watsonx – Mixtral-8x7B-Instruct |
| *Vector Search* | FAISS |
| *Embeddings* | SentenceTransformers |
| *PDF Processing* | PyMuPDF |
| *Voice Assistant* | gTTS / SpeechRecognition |
| *Additional Libraries* | HuggingFace Transformers, NumPy, Pandas |

---

## 🧠 Architecture Flow


📂 PDF Upload (Streamlit UI)
        ↓
🧾 Text Extraction & Chunking (PyMuPDF)
        ↓
🔢 Embedding Generation (SentenceTransformers)
        ↓
🔍 Semantic Search & Indexing (FAISS)
        ↓
🤖 Answer Generation (IBM Watsonx Mixtral-8x7B)
        ↓
💬 Output on Streamlit → Voice, Notes, Chatbot

🔍 How It Works — Step by Step

Upload PDFs via the Streamlit web interface.

Extract and preprocess text from PDFs using PyMuPDF.

Convert chunks into embeddings using SentenceTransformers.

Retrieve top-K relevant passages using FAISS similarity search.

Feed retrieved text + question into the LLM (IBM Watsonx Mixtral-8x7B) for contextual answer generation.

Display and interact with the answer: listen via voice, take notes, or continue the chat.

**🌍 Why StudyMate is Unique**

🧭 Contextual & Grounded — Answers come only from uploaded PDFs, ensuring accuracy.

🧠 Integrated Productivity Suite — Includes timer, notes, and chatbot tools.

⚡ Lightweight and Local — No dependency on cloud storage for document data.

🔒 Privacy-first — Users’ study material never leaves the local environment.

💬 Human-like interaction — AI assistant that learns from the conversation flow.

**🔮 Future Enhancements**
Phase	Upcoming Features
Next Release	Cross-document reasoning, automatic quiz generation
Mid-Term	Personalized learning profiles & progress tracking
Future Scope	Integration with Learning Management Systems (LMS), AI-based tutoring dashboard, cloud sync & analytics


**💻 Installation & Setup**
# Clone the repository
git clone https://github.com/your-username/StudyMate.git
cd StudyMate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py


Once started, open the local URL (e.g., http://localhost:8501) in your browser.

## 👥 Team Members

| 👤 *Name* | 🧩 *Role* | 🛠 *Contribution* |
|-------------|--------------|----------------------|
| *KARRI UDAY* | 🧠 AI & Backend Integration | Built LLM pipeline, FAISS retrieval, and backend system |
| *JEEVAN W PRAKASH* | 🎨 Frontend Developer | Developed Streamlit UI and implemented extra AI features |
| *JAGADEESH C* | 🎤 Data & Voice Integration | Added AI voice assistant and Pomodoro timer |
| *(QA & Documentation)* | 🧾 Quality Assurance & Docs | Built Sticky Notes feature, tested modules, and handled documentation |

**Conclusion**

StudyMate isn’t just another chatbot — it’s a personalized AI study companion.
By combining context-grounded Q&A, voice interaction, and productivity tools, it makes studying more efficient, interactive, and fun.

📚 “Study smarter, not harder — with StudyMate.”
