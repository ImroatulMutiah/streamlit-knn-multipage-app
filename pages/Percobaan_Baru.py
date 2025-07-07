
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


st.markdown("""
    <style>
    /* Gaya global */
    body {
        background-color: #f5f9ff;
        color: #1c1c1c;
    }

    /* Header */
    .stApp h1 {
        color: #0b5394;
    }

    .stApp h2, .stApp h3 {
        color: #0b5394;
    }

    /* Dropdown (selectbox) */
    div[data-baseweb="select"] > div {
        background-color: #e7f0fa;
        border: 1px solid #a8c4e4;
        border-radius: 8px;
    }

    /* Tombol Prediksi */
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

    /* Box hasil prediksi */
    .stAlert {
        border-radius: 8px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #dceafc;
    }
    </style>
""", unsafe_allow_html=True)


st.title("Percobaan Baru - Input Data dan Prediksi")

# Best model parameters from research results
N_NEIGHBORS = 7
WEIGHTS = 'distance'
METRIC = 'manhattan'

# Load dataset to get feature names and for input validation
@st.cache_data
def load_dataset():
    import os
    dataset_path = os.path.join("..", "OneDrive", "Documents", "Tugas Mutiah", "Skripsi", "phishing_website_dataset.csv")
    df = pd.read_csv(dataset_path, delimiter=';')
    return df

def main():
    
    st.header("Masukkan Data Baru untuk Prediksi")
    df = load_dataset()

    # Urutan fitur sesuai dataset asli
    features = ['having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
                'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
                'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
                'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
                'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
                'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
                'Statistical_report']

    # Mapping label input ke user-friendly
    label_map = {
        'having_IP_Address': ["Ya", "-", "Tidak"],
        'URL_Length': [">54", "54 - 75", "<54"],
        'Shortining_Service': ["Ya", "-", "Tidak"],
        'having_At_Symbol': ["Ya", "-", "Tidak"],
        'double_slash_redirecting': ["Ya", "-", "Tidak"],
        'Prefix_Suffix': ["Ya", "-", "Tidak"],
        'having_Sub_Domain': ["0", "1", ">2"],
        'SSLfinal_State': ["Valid", "Tidak Ada", "Tidak Valid"],
        'Domain_registeration_length': [">1 Tahun", "-", "<1 Tahun"],
        'Favicon': ["Ya", "-", "Tidak"],
        'port': ["Ya", "-", "Tidak"],
        'HTTPS_token': ["Ya", "-", "Tidak"],
        'Request_URL': [">50%", "-", "<50%"],
        'URL_of_Anchor': [">50%", "31% - 49%", "<31%"],
        'Links_in_tags': [">50%", "17% - 50%", "<17%"],
        'SFH': ["Ya", "Tidak Ada", "Tidak"],
        'Submitting_to_email': ["Ya", "-", "Tidak"],
        'Abnormal_URL': ["Ya", "-", "Tidak"],
        'Redirect': ["1", "Tidak Ada", "-"],
        'on_mouseover': ["Ya", "-", "Tidak"],
        'RightClick': ["Ya", "-", "Tidak"],
        'popUpWidnow': ["Ya", "-", "Tidak"],
        'Iframe': ["Ya", "-", "Tidak"],
        'age_of_domain': [">6 Bulan", "-", "<6 Bulan"],
        'DNSRecord': ["Ya", "-", "Tidak"],
        'web_traffic': ["Tinggi", "Sedang", "Rendah"],
        'Page_Rank': ["Tinggi", "-", "Rendah"],
        'Google_Index': ["Ya", "-", "Tidak"],
        'Links_pointing_to_page': ["Banyak", "Sedang", "Sedikit"],
        'Statistical_report': ["Ya", "-", "Tidak"]
    }

    # Label ramah pengguna
    friendly_feature_labels = {
        'having_IP_Address': 'Punya Alamat IP?',
        'URL_Length': 'Panjang URL Berapa?',
        'Shortining_Service': 'Apakah URL menggunakan layanan pemendek tautan (seperti bit.ly)?',
        'having_At_Symbol': 'Mengandung simbol @?',
        'double_slash_redirecting': 'Mengandung // setelah protokol?',
        'Prefix_Suffix': 'Domain mengandung tanda "-"?',
        'having_Sub_Domain': 'Berapa Jumlah Subdomain?',
        'SSLfinal_State': 'Apakah situs memiliki sertifikat SSL yang valid (https)?',
        'Domain_registeration_length': 'Berapa Lama Registrasi Domain?',
        'Favicon': 'Apakah ikon situs (favicon) dimuat dari domain yang sama?',
        'port': 'Menggunakan port tidak standar?',
        'HTTPS_token': 'Mengandung token HTTPS?',
        'Request_URL': 'Sumber daya di domain yang sama?',
        'URL_of_Anchor': 'Apakah link di halaman ini mengarah ke situs yang sama?',
        'Links_in_tags': 'Tag mengarah ke domain yang sama?',
        'SFH': 'Ke mana formulir data dikirimkan saat diisi?',
        'Submitting_to_email': 'Mengirim form ke email?',
        'Abnormal_URL': 'URL tidak sesuai domain?',
        'Redirect': 'Apakah URL mengarahkan ulang ke halaman lain sebelum memuat?',
        'on_mouseover': 'Apakah status browser berubah saat kursor diarahkan ke link atau tombol?',
        'RightClick': 'Klik kanan dinonaktifkan?',
        'popUpWidnow': 'Menggunakan pop-up?',
        'Iframe': 'Apakah halaman menggunakan bingkai (iframe) untuk menampilkan konten?',
        'age_of_domain': 'Berapa Usia Domain?',
        'DNSRecord': 'Apakah domain memiliki catatan DNS aktif?',
        'web_traffic': 'Seberapa sering situs ini dikunjungi oleh pengguna internet?',
        'Page_Rank': 'Peringkat halaman?',
        'Google_Index': 'Apakah halaman ini terdaftar di hasil pencarian Google?',
        'Links_pointing_to_page': 'Berapa Jumlah link menuju halaman?',
        'Statistical_report': 'Dilaporkan sebagai phishing?'
    }

    # Ambil input user
    input_data = {}
    for feature in features:
        options = [1, 0, -1]
        labels = label_map[feature]
        display_label = friendly_feature_labels.get(feature, feature.replace('_', ' '))
        choice = st.selectbox(
    label=display_label,
    options=options,
    format_func=lambda x: labels[options.index(x)]
)
        input_data[feature] = choice

    if st.button("Prediksi"):
        input_df = pd.DataFrame([input_data])
        X = df[features]
        y = df[df.columns[-1]]

        knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights=WEIGHTS, metric=METRIC)
        knn.fit(X, y)

        prediction = knn.predict(input_df)[0]
        if prediction == -1:
            st.error("🚫 Hasil Prediksi: Website ini DIPERKIRAKAN sebagai PHISHING")
            st.warning("⚠️ Harap berhati-hati dan jangan memberikan informasi sensitif pada website ini.")
        elif prediction == 1:
            st.success("✅ Hasil Prediksi: Website ini DIPERKIRAKAN sebagai LEGITIMATE (AMAN)")
            st.info("ℹ️ Website ini tampaknya aman, namun tetap perhatikan tanda-tanda keamanan standar.")
        else:
            st.warning(f"⚠️ Hasil Prediksi: Kode prediksi tidak dikenal: {prediction}")
if __name__ == "__main__":
    main()
