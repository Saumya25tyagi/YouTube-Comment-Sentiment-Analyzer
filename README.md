# 📊 YouTube Comment Sentiment Analyzer

A simple Python + Streamlit application that fetches comments from any YouTube video using the **YouTube Data API**, performs **sentiment analysis** using a **BERT-based multilingual model**, and displays the results in an interactive dashboard.

---

## 🚀 Features

- 🔗 Fetches comments using YouTube Data API  
- 🤖 Uses BERT (`nlptown/bert-base-multilingual-uncased-sentiment`) for accurate sentiment analysis  
- ✅ Filters out invalid / empty comments  
- 📥 Automatically generates:
  - `youtube_comments.csv` → raw comments  
  - `youtube_comments_with_sentiments.csv` → comments + sentiment  
- 📈 Displays results in a neat pie chart (Streamlit UI)

---

## 🛠 Tech Stack

| Tool | Use |
|------|-----------------------------|
| Python | Core language |
| Streamlit | Dashboard / UI |
| HuggingFace Transformers | Sentiment model |
| Google API Client | YouTube Data API |
| Matplotlib | Visualizations |

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
streamlit run streamlit_app.py
```

> 💡 Make sure you have a valid **YouTube Data API key** in your `streamlit_app.py` file.

---

## ❌ Files Excluded from Git

```
youtube_comments.csv
youtube_comments_with_sentiments.csv
```

These files are auto-generated on each run and have been intentionally ignored using `.gitignore`.

---

## ❤️ Author

Made with ❤️ by **Saumya**
