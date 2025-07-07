import streamlit as st

# Tambahkan gaya visual agar konsisten dengan halaman lain
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

# Konten utama halaman Home
def main():
    st.title("Apa Itu Phishing?!")
    st.write("""
## **Waspadai Penipuan di Dunia Maya**

Phishing adalah salah satu bentuk kejahatan siber di mana pelaku menyamar sebagai pihak terpercaya untuk **menipu korban agar memberikan informasi sensitif**, seperti username, password, nomor kartu kredit, hingga data pribadi lainnya. Biasanya, aksi phishing dilakukan melalui **website palsu** yang dibuat sangat mirip dengan situs resmi.

---

## **Bagaimana Phishing Website Bekerja?**

Website phishing sering kali menyamar sebagai halaman login dari layanan populer seperti email, perbankan, atau e-commerce. Ketika pengguna tidak waspada dan memasukkan informasi mereka, data tersebut langsung dikirim ke pelaku — bukan ke perusahaan resmi.

---

## **Ciri-Ciri Umum Phishing Website**

Agar Anda terhindar dari ancaman phishing, kenali beberapa tanda-tanda website palsu berikut ini:

🔗 **URL Tidak Biasa**  
Sering menggunakan alamat website yang mirip dengan aslinya, tapi dengan sedikit perbedaan (contoh: `g00gle.com`, `bukalapak-kami.net`).

🔒 **Tidak Ada HTTPS atau Sertifikat Keamanan Palsu**  
Website resmi biasanya menggunakan HTTPS dan memiliki ikon gembok. Situs phishing kadang tidak memilikinya, atau menggunakan sertifikat murahan.

📧 **Tautan dari Email Mencurigakan**  
Phishing sering dimulai dari email yang mendesak Anda untuk "segera login" atau "verifikasi akun" dengan tautan mencurigakan.

💬 **Tampilan Website Mirip Tapi Tidak Persis**  
Desain, logo, atau warna mungkin terlihat aneh, kabur, atau berbeda dari website resmi.

⚠️ **Permintaan Informasi Pribadi Secara Langsung**  
Situs resmi jarang meminta informasi sensitif secara langsung. Jika sebuah website tiba-tiba meminta data penting, berhati-hatilah!

---

🎯 **Ingat:** Satu klik yang salah bisa berakibat fatal.  
**Lindungi diri Anda, kenali ancamannya, dan jadilah pengguna internet yang cerdas!**
    """)

if __name__ == "__main__":
    main()
