import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Gaya tampilan
st.markdown("""
    <style>
    body { background-color: #f5f9ff; color: #1c1c1c; }
    .stApp h1, .stApp h2, .stApp h3 { color: #0b5394; }
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
    .stAlert { border-radius: 8px; }
    section[data-testid="stSidebar"] { background-color: #dceafc; }
    </style>
""", unsafe_allow_html=True)

st.title("Percobaan Baru - Input Data dan Prediksi")

# Parameter model terbaik
N_NEIGHBORS = 7
WEIGHTS = 'distance'
METRIC = 'manhattan'

# Load dataset dari file di folder yang sama
@st.cache_data
def load_dataset():
    return pd.read_csv("phishing_website_dataset.csv", delimiter=';')

def main():
    st.header("Masukkan Data Baru untuk Prediksi")
    df = load_dataset()

    # Daftar fitur sesuai urutan
    features = ['having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
                'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
                'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
                'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
                'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
                'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
                'Statistical_report']

    # Opsi input ramah pengguna
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

    friendly_feature_labels = {
        'having_IP_Address': 'Punya Alamat IP?',
        'URL_Length': 'Panjang URL Berapa?',
        'Shortining_Service': 'Apakah menggunakan URL pendek?',
        'having_At_Symbol': 'Mengandung simbol @?',
        'double_slash_redirecting': 'Mengandung // setelah protokol?',
        'Prefix_Suffix': 'Mengandung tanda "-"?',
        'having_Sub_Domain': 'Jumlah Subdomain?',
        'SSLfinal_State': 'Status SSL?',
        'Domain_registeration_length': 'Lama registrasi domain?',
        'Favicon': 'Favicon dimuat dari domain sama?',
        'port': 'Gunakan port tidak standar?',
        'HTTPS_token': 'Mengandung HTTPS palsu?',
        'Request_URL': 'Rasio sumber daya lokal?',
        'URL_of_Anchor': 'Anchor menuju situs yang sama?',
        'Links_in_tags': 'Link dalam tag menuju domain sama?',
        'SFH': 'Form dikirim ke?',
        'Submitting_to_email': 'Kirim ke email?',
        'Abnormal_URL': 'URL tidak normal?',
        'Redirect': 'Redirect sebelum halaman tampil?',
        'on_mouseover': 'Status browser berubah saat hover?',
        'RightClick': 'Klik kanan dinonaktifkan?',
        'popUpWidnow': 'Mengandung pop-up?',
        'Iframe': 'Mengandung iframe?',
        'age_of_domain': 'Umur domain?',
        'DNSRecord': 'Punya DNS aktif?',
        'web_traffic': 'Tingkat traffic?',
        'Page_Rank': 'Peringkat halaman?',
        'Google_Index': 'Terindex Google?',
        'Links_pointing_to_page': 'Jumlah link ke halaman?',
        'Statistical_report': 'Dilaporkan phishing?'
    }

    # Input pengguna
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

    # Prediksi
    if st.button("Prediksi"):
        input_df = pd.DataFrame([input_data])
        X = df[features]
        y = df[df.columns[-1]]

        knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights=WEIGHTS, metric=METRIC)
        knn.fit(X, y)
        prediction = knn.predict(input_df)[0]

        if prediction == -1:
            st.error("🚫 Website ini DIPERKIRAKAN sebagai PHISHING.")
            st.warning("⚠️ Jangan masukkan data pribadi ke situs ini!")
        elif prediction == 1:
            st.success("✅ Website ini DIPERKIRAKAN AMAN.")
            st.info("ℹ️ Tetap berhati-hati dan cek keamanan situs.")
        else:
            st.warning(f"⚠️ Hasil tidak dikenal: {prediction}")

if __name__ == "__main__":
    main()
