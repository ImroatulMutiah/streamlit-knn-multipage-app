import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

st.title("Hasil Penelitian Phishing Websites menggunakan K-Nearest Neighbors")

@st.cache_data
def load_dataset():
    return pd.read_csv("pages/phishing_website_dataset.csv", delimiter=';')

@st.cache_data
def load_results_80_20():
    return pd.read_excel("pages/hasil_knn 8020.xlsx")

@st.cache_data
def load_results_kfold():
    return pd.read_excel("pages/hasil_tuning KFOLD_knn.xlsx")

@st.cache_data
def load_results_80_20_custom():
    return pd.read_excel("pages/split_result_k3_distance_manhattan.xlsx")

@st.cache_data
def load_results_kfold_custom():
    return pd.read_excel("pages/result_k3_distance_manhattan.xlsx")

def plot_metrics(results_df):
    plt.figure(figsize=(14, 10))
    grouping_columns = ['n_neighbors', 'weights', 'metric']
    has_grouping = all(col in results_df.columns for col in grouping_columns)

    feature_col = accuracy_col = precision_col = recall_col = f1_col = None
    for col in results_df.columns:
        cl = col.lower()
        if cl in ['num features', 'fitur', 'features']:
            feature_col = col
        elif cl in ['accuracy', 'akurasi']:
            accuracy_col = col
        elif cl in ['precision', 'presisi']:
            precision_col = col
        elif cl in ['recall']:
            recall_col = col
        elif cl in ['f1 score', 'f1score', 'f1']:
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
        st.error(f"Kolom yang diperlukan tidak ditemukan. Kolom tersedia: {list(results_df.columns)}")
        return

    plt.xlabel('Jumlah Fitur')
    plt.ylabel('Skor')
    plt.title('Metode KNN - Grafik Evaluasi')
    if feature_col:
        plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend(loc='best', fontsize='small')
    st.pyplot(plt.gcf())
    plt.clf()

def main():
    st.header("Dataset")
    df = load_dataset()
    st.dataframe(df)

    st.header("Hasil Pengujian 80:20 Split")
    results_80_20 = load_results_80_20()
    st.dataframe(results_80_20)
    st.subheader("Grafik Performansi 80:20")
    plot_metrics(results_80_20)

    st.header("Hasil Pengujian 10-Fold Cross Validation")
    results_kfold = load_results_kfold()
    st.dataframe(results_kfold)
    st.subheader("Grafik Performansi K-Fold")
    plot_metrics(results_kfold)

    st.header("Pengujian 80:20 dengan Parameter Custom")
    results_80_20_custom = load_results_80_20_custom()
    st.dataframe(results_80_20_custom)
    st.subheader("Grafik Parameter Custom 80:20")
    plot_metrics(results_80_20_custom)

    st.header("Pengujian K-Fold dengan Parameter Custom")
    results_kfold_custom = load_results_kfold_custom()
    st.dataframe(results_kfold_custom)
    st.subheader("Grafik Parameter Custom K-Fold")
    plot_metrics(results_kfold_custom)

    st.header("Best Model (80:20 Split, k=7, Manhattan Distance)")
    st.write("Jumlah Fitur: 30")
    st.write("Parameter Model:")
    st.markdown("- n_neighbors: **7**  \n- metric: **manhattan**  \n- weights: **distance**")

    feature_scores_data = {
        'Feature': ['SSLfinal_State', 'URL_of_Anchor', 'Prefix_Suffix', 'web_traffic', 'having_Sub_Domain',
                   'Request_URL', 'SFH', 'Links_in_tags', 'Domain_registeration_length', 'Google_Index',
                   'Statistical_report', 'URL_Length', 'popUpWidnow', 'Page_Rank', 'Redirect',
                   'HTTPS_token', 'having_At_Symbol', 'Submitting_to_email', 'RightClick', 'having_IP_Address',
                   'DNSRecord', 'port', 'Abnormal_URL', 'age_of_domain', 'on_mouseover',
                   'Iframe', 'double_slash_redirecting', 'Favicon', 'Shortining_Service', 'Links_pointing_to_page'],
        'Weight': [0.350641, 0.334745, 0.094622, 0.082344, 0.076275,
                  0.033808, 0.033090, 0.030760, 0.025923, 0.012840,
                  0.007115, 0.006006, 0.004615, 0.004309, 0.004203,
                  0.004203, 0.003225, 0.002606, 0.002181, 0.002118,
                  0.002063, 0.001868, 0.001820, 0.001199, 0.000000,
                  0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
    }

    st.write("Fitur Terpilih Berdasarkan Information Gain:")
    for i, (feature, score) in enumerate(zip(feature_scores_data['Feature'], feature_scores_data['Weight']), 1):
        st.write(f"{i}. {feature} (Score: {score:.6f})")

    st.write("Metrik Performa:")
    st.markdown("- Accuracy: **0.972411**  \n- Precision: **0.969502**  \n- Recall: **0.981316**  \n- F1 Score: **0.975373**")

if __name__ == "__main__":
    main()
