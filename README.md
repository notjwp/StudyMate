📘 StudyMate: An AI-Powered PDF-Based Q&A System for Students

> 🚀 Your AI study companion that transforms PDFs into interactive, voice-enabled, and intelligent learning experiences.

---

## 🧠 Project Description

*StudyMate* is an *AI-powered academic assistant* that enables students to interact with their study materials — such as *textbooks, lecture notes, and research papers* — in a *conversational, question-answering format*.  

Instead of passively reading large PDFs or relying on manual searches for specific information, users can *upload one or more PDFs* and *ask natural-language questions*.  
StudyMate responds with *direct, well-contextualized answers*, grounded and referenced from the uploaded source content.

---

## 🎯 Key Objectives / Expected Solutions

1. *Conversational Q&A from Academic PDFs*  
   Enables students to ask natural-language questions and receive *contextual answers grounded in their own study materials*.

2. *Accurate Text Extraction and Preprocessing*  
   Efficiently extracts and chunks content from multiple PDFs using *PyMuPDF* for high-quality downstream processing.

3. *Semantic Search Using FAISS and Embeddings*  
   Retrieves the most relevant text chunks using *SentenceTransformers embeddings* and *FAISS vector search* for precise question matching.

4. *LLM-Based Answer Generation*  
   Uses *IBM Watsonx’s Mixtral-8x7B-Instruct* model to generate *informative, grounded answers* from retrieved content.

5. *User-Friendly Local Interface*  
   A clean *Streamlit-based frontend* allows seamless document upload, question input, and visualization of AI-generated results.

---

## ✨ Extra Features (Our Unique Additions)

To make StudyMate a *complete learning ecosystem*, we added the following innovative AI-powered tools:

### 🗣 1. AI Voice Assistant  
- Reads out answers using a Text-to-Speech engine.  
- Allows voice-based question input.  
- Increases accessibility and enables hands-free learning.  

### ⏳ 2. Pomodoro Timer Page  
- Built-in *Pomodoro productivity tool* to help students manage focused study sessions.  
- Optional study-break cycles (25/5, 45/10, etc.) integrated into the StudyMate interface.

### 📝 3. Sticky Notes Feature  
- Lets users *create, edit, and save quick notes* during their study sessions.  
- Notes are linked to PDF sections or AI answers and persist across sessions.  

### 🤖 4. Smart Trained Chatbot  
- A *context-aware chatbot* that remembers previous queries within a session.  
- Enables extended discussion with uploaded study materials, acting like a *personal tutor*.

---

## 🧩 Architecture Flow

```text
📂 PDF Upload (User)
       ↓
🧾 Text Extraction & Chunking (PyMuPDF)
       ↓
🔍 Embedding Generation (SentenceTransformers)
       ↓
🧠 Semantic Search & Indexing (FAISS)
       ↓
🤖 Answer Generation (IBM Watsonx Mixtral-8x7B)
       ↓
💬 Streamlit UI → Display Answers + Voice Output + Notes
