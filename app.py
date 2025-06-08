
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Memuat dataset dan model
@st.cache
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache
def load_data():
    # Ganti dengan path dataset yang sesuai
    df = pd.read_csv('https://raw.githubusercontent.com/salsabiladi/TUBES_DATAMINING/refs/heads/main/digital_impact_mental_health.csv')
    return df

# Memuat dataset dan model
df = load_data()
model = load_model()

# Hanya memilih fitur yang relevan untuk analisis
selected_features = [
    'social_media_hours', 'phone_usage_hours', 'sleep_duration_hours', 
    'weekly_depression_score', 'tablet_usage_hours', 'physical_activity_hours_per_week', 'mental_health_score'
]
df = df[selected_features]

# Sidebar untuk navigasi halaman
st.sidebar.title("Navigasi Dashboard")
page = st.sidebar.radio("Pilih Halaman:", ["Pengenalan", "Analisis Clustering", "Rekomendasi Gaya Hidup"])

if page == "Pengenalan":
    st.title("🧑‍💻 Analisis Pola Konsumsi Digital dan Kesehatan Mental 🧑‍💻")
    st.write("""
    Selamat datang di dashboard analisis pola konsumsi digital dan dampaknya terhadap kesehatan mental menggunakan **K-Means Clustering**.
    Berdasarkan pola konsumsi digital yang meliputi penggunaan media sosial, ponsel, tablet, serta data kesehatan mental seperti stres dan kecemasan, model ini akan mengelompokkan pengguna menjadi beberapa cluster.
    """)

elif page == "Analisis Clustering":
    st.title("📊 Hasil Clustering berdasarkan Pola Konsumsi Digital dan Kesehatan Mental")
    st.write("""
    Aplikasi ini melakukan analisis pola konsumsi digital berdasarkan data penggunaan media sosial, ponsel, waktu tidur, dan faktor-faktor kesehatan mental.
    Hasil clustering akan membagi pengguna dalam beberapa kelompok berdasarkan pola konsumsi digital dan dampaknya terhadap kesehatan mental mereka.
    """)

    # Visualisasi Hasil Clustering dengan PCA
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df)

    kmeans = KMeans(n_clusters=2)
    df['Cluster'] = kmeans.fit_predict(df_pca)

    plt.figure(figsize=(8, 6))
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['Cluster'], cmap='viridis')
    plt.title("Visualisasi Clustering K-Means")
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(label='Cluster')
    st.pyplot(plt)

elif page == "Rekomendasi Gaya Hidup":
    st.title("Rekomendasi Berdasarkan Hasil Clustering")
    st.write("""
    Berdasarkan hasil analisis clustering, aplikasi ini memberikan rekomendasi yang lebih personal tentang pola konsumsi digital yang lebih sehat.
    Anda dapat memasukkan beberapa karakteristik untuk mendapatkan rekomendasi berdasarkan cluster yang diprediksi.
    """)

    # Input Pengguna untuk Prediksi Cluster
    social_media_hours = st.number_input('Social Media Hours per Day', min_value=0, max_value=24, value=3)
    phone_usage_hours = st.number_input('Phone Usage Hours per Day', min_value=0, max_value=24, value=4)
    sleep_duration = st.number_input('Sleep Duration (hours)', min_value=0, max_value=24, value=7)
    weekly_depression_score = st.number_input('Weekly Depression Score', min_value=0, max_value=10, value=5)
    tablet_usage_hours = st.number_input('Tablet Usage Hours per Day', min_value=0, max_value=24, value=2)
    physical_activity_hours = st.number_input('Physical Activity Hours per Week', min_value=0, max_value=20, value=5)
    mental_health_score = st.number_input('Mental Health Score', min_value=0, max_value=10, value=5)

    # Normalisasi data input
    scaler = StandardScaler()
    input_data = np.array([[social_media_hours, phone_usage_hours, sleep_duration, weekly_depression_score,
                            tablet_usage_hours, physical_activity_hours, mental_health_score]])

    # Normalisasi input data
    input_data_scaled = scaler.fit_transform(input_data)

    # Prediksi dengan K-Means
    if st.button('Predict Cluster and Get Recommendation'):
        try:
            cluster = model.predict(input_data_scaled)[0]

            st.header("Predicted Cluster")
            if cluster == 0:
                st.success("You belong to Cluster 1: High Performance Members")
                st.write("""
                **Characteristics of High Performance Members**:
                - Higher social media usage and screen time
                - Higher stress levels
                - Needs to reduce screen time and improve sleep quality
                - Focus on physical activity and mindfulness
                
                **Recommendation**:
                - **Reduce screen time**, especially on social media.
                - Increase **physical activity** and focus on mental wellness (e.g., meditation, yoga).
                """)
            else:
                st.warning("You belong to Cluster 2: Development Members")
                st.write("""
                **Characteristics of Development Members**:
                - Moderate screen time and social media usage
                - Moderate stress levels
                - Needs to improve consistency in workout and sleep habits
                
                **Recommendation**:
                - **Balance screen time** with regular breaks.
                - Start with **moderate workouts** focusing on form and technique.
                - **Improve sleep quality** for better mental health.
                """)

        except Exception as e:
            st.error(f'Terjadi kesalahan dalam prediksi: {str(e)}')


