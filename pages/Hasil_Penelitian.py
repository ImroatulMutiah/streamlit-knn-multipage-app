import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

st.title("Hasil Penelitian Phishing Websites menggunakan K-Nearest Neighbors")

@st.cache_data
def load_dataset():
    # Load dataset used in the notebooks (assuming CSV or Excel file available)
    # Adjust the path to the actual dataset file location
    import os
    dataset_path = os.path.join("..", "OneDrive", "Documents", "Tugas Mutiah", "Skripsi", "phishing_website_dataset.csv")
    df = pd.read_csv(dataset_path, delimiter=';')
    return df

@st.cache_data
def load_results_80_20():
    # Load results from 80:20 split experiment
    import os
    path_80_20 = os.path.join("..", "OneDrive", "Documents", "Tugas Mutiah", "Skripsi", "hasil_knn 8020.xlsx")
    df = pd.read_excel(path_80_20)
    return df

@st.cache_data
def load_results_kfold():
    # Load results from 10-fold cross-validation experiment
    import os
    path_kfold = os.path.join("..", "OneDrive", "Documents", "Tugas Mutiah", "Skripsi", "hasil_tuning KFOLD_knn.xlsx")
    df = pd.read_excel(path_kfold)
    return df

@st.cache_data
def load_results_80_20_custom():
    # Load results from 80:20 split with custom parameters
    import os
    path_80_20_custom = os.path.join("..", "OneDrive", "Documents", "Tugas Mutiah", "Skripsi", "split_result_k3_distance_manhattan.xlsx")
    df = pd.read_excel(path_80_20_custom)
    return df

@st.cache_data
def load_results_kfold_custom():
    # Load results from k-fold with custom parameters
    import os
    path_kfold_custom = os.path.join("..", "OneDrive", "Documents", "Tugas Mutiah", "Skripsi", "result_k3_distance_manhattan.xlsx")
    df = pd.read_excel(path_kfold_custom)
    return df

def plot_metrics(results_df):
    plt.figure(figsize=(14, 10))
    
    # Check if the dataframe has grouping columns
    grouping_columns = ['n_neighbors', 'weights', 'metric']
    has_grouping = all(col in results_df.columns for col in grouping_columns)
    
    # Check for different possible column names
    feature_col = None
    accuracy_col = None
    precision_col = None
    recall_col = None
    f1_col = None
    
    # Try different column name variations
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
        # Original plotting with grouping
        for (n_neighbors, weights, metric), group_data in results_df.groupby(grouping_columns):
            plt.plot(group_data[feature_col], group_data[accuracy_col], marker='o', linestyle='-', 
                    label=f'Acc n={n_neighbors}, w={weights}, m={metric}')
            if precision_col:
                plt.plot(group_data[feature_col], group_data[precision_col], marker='x', linestyle='--', 
                        label=f'Prec n={n_neighbors}, w={weights}, m={metric}')
            if recall_col:
                plt.plot(group_data[feature_col], group_data[recall_col], marker='^', linestyle='-.', 
                        label=f'Recall n={n_neighbors}, w={weights}, m={metric}')
            if f1_col:
                plt.plot(group_data[feature_col], group_data[f1_col], marker='s', linestyle=':', 
                        label=f'F1 n={n_neighbors}, w={weights}, m={metric}')
    elif feature_col and accuracy_col:
        # Simple plotting without grouping
        plt.plot(results_df[feature_col], results_df[accuracy_col], marker='o', linestyle='-', label='Accuracy')
        if precision_col:
            plt.plot(results_df[feature_col], results_df[precision_col], marker='x', linestyle='--', label='Precision')
        if recall_col:
            plt.plot(results_df[feature_col], results_df[recall_col], marker='^', linestyle='-.', label='Recall')
        if f1_col:
            plt.plot(results_df[feature_col], results_df[f1_col], marker='s', linestyle=':', label='F1 Score')
    else:
        # If no matching columns found, display available columns
        st.error(f"Could not find expected columns. Available columns: {list(results_df.columns)}")
        return
    
    plt.xlabel('Number of Features Used')
    plt.ylabel('Score')
    plt.title('KNN Performance Metrics')
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

    st.subheader("Grafik Performansi 80:20 Split")
    plot_metrics(results_80_20)

    st.header("Hasil Pengujian 10-Fold Cross Validation")
    results_kfold = load_results_kfold()
    st.dataframe(results_kfold)

    st.subheader("Grafik Performansi 10-Fold Cross Validation")
    plot_metrics(results_kfold)

    st.header("Hasil Pengujian 80:20 Split dengan Parameter Custom")
    results_80_20_custom = load_results_80_20_custom()
    st.dataframe(results_80_20_custom)

    st.subheader("Grafik Performansi 80:20 Split dengan Parameter Custom")
    plot_metrics(results_80_20_custom)

    st.header("Hasil Pengujian K-Fold Cross Validation dengan Parameter Custom")
    results_kfold_custom = load_results_kfold_custom()
    st.dataframe(results_kfold_custom)

    st.subheader("Grafik Performansi K-Fold Cross Validation dengan Parameter Custom")
    plot_metrics(results_kfold_custom)

    st.header("Best Model (80:20 Split, k=7, Manhattan Distance)")
    # Since we know the best model uses all 30 features
    st.write("Jumlah Fitur: 30")
    st.write("Parameter Model:")
    st.write("- n_neighbors: 7")
    st.write("- metric: manhattan")
    st.write("- weights: distance")
    
    # Feature scores data
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

    # Display all 30 features in order of importance
    st.write("Fitur yang Digunakan (berdasarkan Information Gain):")
    for i, (feature, score) in enumerate(zip(feature_scores_data['Feature'], feature_scores_data['Weight']), 1):
        st.write(f"{i}. {feature} (Score: {score:.6f})")

    # Display metrics for the best model
    st.write("Metrik Performa:")
    st.write("- Accuracy: 0.972411")
    st.write("- Precision: 0.969502")
    st.write("- Recall: 0.981316")
    st.write("- F1 Score: 0.975373")

if __name__ == "__main__":
    main()
