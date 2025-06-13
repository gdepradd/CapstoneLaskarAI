# Journalyze: NLP-powered Diary Analyzer

Journalyze adalah aplikasi prototipe berbasis Python untuk menganalisis entri jurnal harian menggunakan teknologi Natural Language Processing (NLP). Aplikasi ini menyatukan Named Entity Recognition (NER), klasifikasi emosi, dan rangkuman teks, serta menyediakan visualisasi hasil analisis.

## 📂 Struktur Proyek

```
├── 2nd_meet_brainstorming_ide_capstone_fariz_(ner_+text_summarization+text+_results_viz).ipynb
├── FINAL_of_text_summarization_fariz_laskar.ipynb
├── journalyze.py
├── ner_fariz_laskar.ipynb
├── text_classification_fariz_laskar.ipynb
├── requirements.txt
└── README.md

````

## 🧠 Deskripsi File

| File | Deskripsi |
|------|-----------|
| `2nd_meet_brainstorming_ide_capstone_...ipynb` | Notebook berisi eksplorasi awal ide proyek, termasuk visualisasi dummy seperti wordcloud, distribusi entitas dan emosi, serta contoh rangkuman jurnal. |
| `FINAL_of_text_summarization_fariz_laskar.ipynb` | Notebook untuk fine-tuning model text summarization berbasis model BART. |
| `ner_fariz_laskar.ipynb` | Notebook fine-tuning model Named Entity Recognition (NER) menggunakan BERT. |
| `text_classification_fariz_laskar.ipynb` | Notebook fine-tuning model klasifikasi emosi berdasarkan entri jurnal. |
| `journalyze_.py` | Script utama prototipe aplikasi analisis jurnal. |
| `requirements.txt` | Daftar dependensi Python yang diperlukan untuk menjalankan aplikasi. |
| `README.md` | Dokumentasi proyek. |

## 🔧 Fitur Utama

- 📝 **Analisis Jurnal**: Input berupa entri harian teks.
- 🧠 **Named Entity Recognition**: Ekstraksi nama orang, tempat, organisasi, dll.
- 💬 **Klasifikasi Emosi**: Deteksi emosi dari teks seperti joy, sadness, anger, dll.
- 📚 **Text Summarization**: Ringkasan otomatis dari jurnal menggunakan model BART.
- 🌥 **Visualisasi WordCloud**: Kata-kata penting yang sering muncul.
- 📊 **Distribusi Emosi & Entitas**: Grafik ringkasan berdasarkan entri jurnal.

## ▶️ Cara Menjalankan

1. Install dependencies:

```bash
pip install -r requirements.txt
````

2. Jalankan aplikasi prototipe:

```bash
streamlit run journalyze_no-db.py
```
