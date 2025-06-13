import os
import streamlit as st
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    TFAutoModelForSequenceClassification,
)

# === Konfigurasi Path CSV ===
CSV_PATH = "journal_entries.csv"

# === Atur direktori cache khusus Hugging Face ===
HF_CACHE_DIR = "./hf_cache"
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR

# === Inisialisasi session_state untuk menyimpan entri ===
if "entries" not in st.session_state:
    st.session_state.entries = []

# Load dari CSV jika ada
if os.path.exists(CSV_PATH) and not st.session_state.entries:
    df = pd.read_csv(CSV_PATH)
    for _, row in df.iterrows():
        st.session_state.entries.append({
            "date": pd.to_datetime(row["date"]).date(),
            "text": row["text"]
        })

# === Load pipelines dan cache ke disk ===
@st.cache_resource
def load_pipelines():
    cache_dir = "./hf_cache"

    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("farizkuy/bart-xsum-finetuned-fariz", cache_dir=cache_dir)
    summarizer_tokenizer = AutoTokenizer.from_pretrained("farizkuy/bart-xsum-finetuned-fariz", cache_dir=cache_dir)
    summarizer = pipeline("summarization", model=summarizer_model, tokenizer=summarizer_tokenizer)

    ner_model = AutoModelForTokenClassification.from_pretrained("farizkuy/bert-laskar-ner", cache_dir=cache_dir)
    ner_tokenizer = AutoTokenizer.from_pretrained("farizkuy/bert-laskar-ner", cache_dir=cache_dir)
    ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")

    emo_model = TFAutoModelForSequenceClassification.from_pretrained("farizkuy/emotion_tf", cache_dir=cache_dir)
    emo_tokenizer = AutoTokenizer.from_pretrained("farizkuy/emotion_tf", cache_dir=cache_dir)
    emo_pipeline = pipeline(
        "text-classification",
        model=emo_model,
        tokenizer=emo_tokenizer,
        top_k=1,
        framework="tf"
    )

    return summarizer, ner_pipeline, emo_pipeline

summarizer, ner_pipeline, emotion_classifier = load_pipelines()

# === UI: Judul & Form ===
st.title("📓 Aplikasi Jurnal Harian dengan Analisis Emosi & Entitas")

# Form input entri
with st.form("entry_form"):
    st.subheader("✍️ Tambahkan Entri Baru")
    entry_date = st.date_input("Tanggal Entri", datetime.today())
    entry_text = st.text_area("Isi Jurnal")
    submitted = st.form_submit_button("Simpan Entri")

    if submitted and entry_text.strip():
        new_entry = {
            "date": entry_date,
            "text": entry_text.strip()
        }

        # Simpan ke session_state
        st.session_state.entries.append(new_entry)

        # Simpan ke CSV
        new_df = pd.DataFrame([new_entry])
        if os.path.exists(CSV_PATH):
            new_df.to_csv(CSV_PATH, mode='a', header=False, index=False)
        else:
            new_df.to_csv(CSV_PATH, index=False)

        st.success("✅ Entri disimpan dan ditambahkan ke CSV!")

# Tampilkan entri yang ada
if st.session_state.entries:
    with st.expander("📚 Lihat Semua Entri"):
        for e in st.session_state.entries:
            st.markdown(f"**{e['date']}**\n\n> {e['text']}")

# === Analisis berdasarkan rentang tanggal ===
st.subheader("📊 Analisis Entri Berdasarkan Rentang Tanggal")

if st.session_state.entries:
    dates = [e["date"] for e in st.session_state.entries]
    min_date, max_date = min(dates), max(dates)

    date_range = st.date_input("Pilih rentang tanggal", value=(min_date, max_date))

    if len(date_range) == 2:
        start_date, end_date = date_range
        selected_entries = [e for e in st.session_state.entries if start_date <= e["date"] <= end_date]

        if selected_entries:
            texts = [e["text"] for e in selected_entries]
            combined_text = " ".join(texts)

            # === WordCloud ===
            st.markdown("### ☁️ WordCloud")
            wc = WordCloud(width=800, height=400, background_color="white").generate(combined_text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            # === Emosi ===
            st.markdown("### 😊 Klasifikasi Emosi")

            EMOTION_LABELS = {
                "LABEL_0": "sadness",
                "LABEL_1": "joy",
                "LABEL_2": "love",
                "LABEL_3": "anger",
                "LABEL_4": "fear",
                "LABEL_5": "surprise"
            }

            emotion_results = []
            for text in texts:
                preds = emotion_classifier(text)
                top_label = preds[0][0]['label']
                mapped_label = EMOTION_LABELS.get(top_label, top_label)
                emotion_results.append(mapped_label)

            if emotion_results:
                emotion_df = pd.DataFrame(emotion_results, columns=["emotion"])
                emotion_counts = emotion_df["emotion"].value_counts().reset_index()
                emotion_counts.columns = ["Emotion", "Count"]
                fig = px.bar(emotion_counts, x="Emotion", y="Count", title="Distribusi Emosi", color="Emotion")
                st.plotly_chart(fig)

                # === Dropdown Multiselect: Tampilkan entri sesuai emosi ===
                st.markdown("### 📥 Lihat Entri Berdasarkan Emosi")
                selected_emotions = st.multiselect("Pilih Emosi", sorted(emotion_df["emotion"].unique()))

                if selected_emotions:
                    for emo in selected_emotions:
                        st.markdown(f"#### Entri dengan Emosi: *{emo}*")
                        for e, label in zip(selected_entries, emotion_results):
                            if label == emo:
                                st.markdown(f"- **{e['date']}**: {e['text']}")
                else:
                    st.info("Pilih satu atau beberapa emosi untuk melihat entri terkait.")

            # === Ringkasan ===
            st.markdown("### 🧠 Ringkasan Entri")
            if len(combined_text.split()) >= 5:
                summary = summarizer(combined_text, max_length=130, min_length=30, do_sample=False)
                st.success(summary[0]['summary_text'])
            else:
                st.info("Teks terlalu pendek untuk diringkas.")

            # === NER ===
            st.markdown("### 🧬 Named Entity Recognition (NER)")
            ner_results = ner_pipeline(combined_text)
            entity_counter = {}

            for ent in ner_results:
                label = ent['entity_group'].upper()
                entity_counter[label] = entity_counter.get(label, 0) + 1

            if entity_counter:
                entity_df = pd.DataFrame(entity_counter.items(), columns=["Entity", "Count"])
                fig = px.pie(entity_df, values='Count', names='Entity', title='Distribusi Entitas Terdeteksi')
                st.plotly_chart(fig)

                with st.expander("📋 Daftar Entitas"):
                    for ent in ner_results:
                        st.markdown(f"- **{ent['entity_group'].upper()}**: {ent['word']}")

        else:
            st.warning("Tidak ada entri pada rentang tanggal yang dipilih.")
else:
    st.info("Belum ada entri. Silakan tambahkan entri terlebih dahulu.")
