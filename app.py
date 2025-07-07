import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import os

# ---------- STYLE ----------
st.set_page_config(page_title="Phishing Website Detection", layout="wide")
st.markdown("""
    <style>
    body {
        background-color: #f5f9ff;
        color: #1c1c1c;
    }
    .stApp h1, .stApp h2, .stApp h3 {
        color: #0b5394;
    }
    div[data-baseweb="select"] > div {
        background-color: #e7f0fa;
        border: 1px solid #a8c4e4;
        border-radius: 8px;
    }
    div.stButton > button {
        background-color: #0b5394;
        color: white;
        border-radius: 6px;
        height: 3em;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
    }
    div.stButton > button:hover {
        background-color: #073763;
        color: white;
        border: 1px solid #0b5394;
    }
    section[data-testid="stSidebar"] {
        background-color: #dceafc;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
@st.cache_data
def load_dataset():
    path = "phishing_website_dataset.csv"
    return pd.read_csv(path, delimiter=';')

@st.cache_data
def load_excel(path):
    return pd.read_excel(path)

# ---------- PLOT METRICS ----------
def plot_metrics(results_df):
    plt.figure(figsize=(14, 10))
    grouping_columns = ['n_neighbors', 'weights', 'metric']
    has_grouping = all(col in results_df.columns for col in grouping_columns)

    feature_col = accuracy_col = precision_col = recall_col = f1_col = None
    for col in results_df.columns:
        if col.lower() in ['num features', 'fitur', 'features']:
            feature_col = col
        elif col.lower() in ['accuracy', 'akurasi']:
            accuracy_col = col
        elif col.lower() in ['precision', 'presisi']:
            precision_col = col
        elif col.lower() in ['recall']:
            recall_col = col
        elif col.lower() in ['f1 score', 'f1score', 'f1']:
            f1_col = col

    if has_grouping and feature_col and accuracy_col:
        for (n_neighbors, weights, metric), group_data in results_df.groupby(grouping_columns):
            plt.plot(group_data[feature_col], group_data[accuracy_col], marker='o', label=f'Acc n={n_neighbors}, w={weights}, m={metric}')
            if precision_col:
                plt.plot(group_data[feature_col], group_data[precision_col], marker='x', linestyle='--', label=f'Prec n={n_neighbors}, w={weights}, m={metric}')
            if recall_col:
                plt.plot(group_data[feature_col], group_data[recall_col], marker='^', linestyle='-.', label=f'Recall n={n_neighbors}, w={weights}, m={metric}')
            if f1_col:
                plt.plot(group_data[feature_col], group_data[f1_col], marker='s', linestyle=':', label=f'F1 n={n_neighbors}, w={weights}, m={metric}')
    elif feature_col and accuracy_col:
        plt.plot(results_df[feature_col], results_df[accuracy_col], marker='o', label='Accuracy')
        if precision_col:
            plt.plot(results_df[feature_col], results_df[precision_col], marker='x', linestyle='--', label='Precision')
        if recall_col:
            plt.plot(results_df[feature_col], results_df[recall_col], marker='^', linestyle='-.', label='Recall')
        if f1_col:
            plt.plot(results_df[feature_col], results_df[f1_col], marker='s', linestyle=':', label='F1 Score')
    else:
        st.error(f"Kolom yang dibutuhkan tidak ditemukan: {list(results_df.columns)}")
        return

    plt.xlabel('Jumlah Fitur')
    plt.ylabel('Skor')
    plt.title('KNN Performance Metrics')
    if feature_col:
        plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend(loc='best', fontsize='small')
    st.pyplot(plt.gcf())
    plt.clf()

# ---------- HALAMAN ----------

def halaman_1_pengantar():
    st.title("Apa Itu Phishing?!")
    st.write("""
    ## **Waspadai Penipuan di Dunia Maya**
    Phishing adalah salah satu bentuk kejahatan siber di mana pelaku menyamar sebagai pihak terpercaya untuk **menipu korban agar memberikan informasi sensitif**.

    ## **Bagaimana Phishing Website Bekerja?**
    Situs palsu menyamar sebagai layanan resmi agar korban memasukkan data pribadi mereka.

    ## **Ciri-Ciri Umum Phishing Website**
    - 🔗 URL tidak biasa (contoh: g00gle.com)
    - 🔒 Tidak ada HTTPS atau sertifikat palsu
    - 📧 Tautan dari email mencurigakan
    - 💬 Tampilan mirip tapi tidak persis
    - ⚠️ Permintaan informasi pribadi secara langsung

    🎯 **Ingat:** Satu klik bisa berakibat fatal. Lindungi dirimu!
    """)

def halaman_2_hasil_penelitian():
    st.title("Hasil Penelitian - KNN pada Phishing Websites")
    df = load_dataset()
    st.subheader("Dataset")
    st.dataframe(df)

    file_paths = {
        "80:20": "hasil_knn_8020.xlsx",
        "10-Fold": "hasil_tuning_KFOLD_knn.xlsx",
        "80:20 Custom": "split_result_k3_distance_manhattan.xlsx",
        "10-Fold Custom": "result_k3_distance_manhattan.xlsx"
    }

    for name, path in file_paths.items():
        st.header(f"Hasil Pengujian {name}")
        result_df = load_excel(path)
        st.dataframe(result_df)
        st.subheader(f"Grafik Performansi {name}")
        plot_metrics(result_df)

    st.header("Best Model (80:20 Split, k=7, Manhattan)")
    st.write("- Accuracy: 0.972411")
    st.write("- Precision: 0.969502")
    st.write("- Recall: 0.981316")
    st.write("- F1 Score: 0.975373")

def halaman_3_prediksi_manual():
    st.title("Percobaan Baru - Prediksi Manual")
    df = load_dataset()

    N_NEIGHBORS = 7
    WEIGHTS = 'distance'
    METRIC = 'manhattan'

    features = [...]  # sama seperti di bagianmu tadi (jika butuh saya lanjutkan lengkap)
    label_map = {...}
    friendly_feature_labels = {...}

    input_data = {}
    for feature in features:
        options = [1, 0, -1]
        labels = label_map[feature]
        display_label = friendly_feature_labels.get(feature, feature)
        choice = st.selectbox(display_label, options, format_func=lambda x: labels[options.index(x)])
        input_data[feature] = choice

    if st.button("Prediksi"):
        input_df = pd.DataFrame([input_data])
        X = df[features]
        y = df[df.columns[-1]]

        knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights=WEIGHTS, metric=METRIC)
        knn.fit(X, y)
        prediction = knn.predict(input_df)[0]

        if prediction == -1:
            st.error("🚫 Website ini DIPERKIRAKAN sebagai PHISHING")
        elif prediction == 1:
            st.success("✅ Website ini DIPERKIRAKAN LEGITIMATE")
        else:
            st.warning(f"Kode prediksi tidak dikenal: {prediction}")

# ---------- MAIN ----------
def main():
    menu = st.sidebar.radio("Navigasi", ["📖 Pengantar", "📊 Hasil Penelitian", "🧪 Prediksi Manual"])
    if menu == "📖 Pengantar":
        halaman_1_pengantar()
    elif menu == "📊 Hasil Penelitian":
        halaman_2_hasil_penelitian()
    elif menu == "🧪 Prediksi Manual":
        halaman_3_prediksi_manual()

if __name__ == "__main__":
    main()
