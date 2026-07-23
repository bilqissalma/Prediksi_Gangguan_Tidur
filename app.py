import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("sleep_disorder_rf.joblib")
label_encoders = joblib.load("label_encoders.joblib")
feature_names = joblib.load("feature_names.joblib")


EXPECTED_FEATURES = [
    "sleep_quality_score",
    "sleep_duration_hrs",
    "mental_health_condition",
    "cognitive_performance_score",
    "bmi",
    "stress_score",
    "sleep_latency_mins",
    "wake_episodes_per_night"
]


# ============================================================
# CEK FITUR MODEL
# ============================================================

if feature_names != EXPECTED_FEATURES:

    st.error(
        "Fitur model tidak sesuai dengan model final 8 fitur."
    )

    st.write(
        "Fitur terbaca:"
    )

    st.write(feature_names)

    st.stop()



# ============================================================
# CSS LAMA ANDA TETAP DI SINI
# COPY CSS LAMA ANDA TANPA PERUBAHAN
# ============================================================



# ============================================================
# HEADER
# ============================================================

st.markdown(
"""
<div class="hero">

<h1>
🌙 Analisis Risiko Gangguan Tidur
</h1>

<h3>
Random Forest Classification — Model Final 8 Fitur
</h3>

<p>
Sistem prediksi tingkat risiko gangguan tidur
berdasarkan pola tidur dan karakteristik pengguna.
</p>

</div>
""",
unsafe_allow_html=True
)



st.write(
"""
Sistem prediksi risiko gangguan tidur menggunakan
algoritma Random Forest berdasarkan 8 fitur utama.
"""
)



# ============================================================
# INPUT DATA PENGGUNA
# ============================================================


st.markdown(
"""
## Masukkan Data Pengguna
"""
)



col1, col2 = st.columns(2)



with col1:


    sleep_quality_score = st.slider(
        "Kualitas Tidur",
        min_value=1,
        max_value=10,
        value=7
    )


    sleep_duration_hrs = st.number_input(
        "Durasi Tidur (jam)",
        min_value=0.0,
        max_value=15.0,
        value=7.0,
        step=0.25
    )


    mental_health_condition = st.selectbox(
        "Kondisi Mental",
        options=[
            "Baik",
            "Sedang",
            "Buruk"
        ]
    )


    cognitive_performance_score = st.number_input(
        "Performa Kognitif",
        min_value=0,
        max_value=100,
        value=80
    )



with col2:


    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=50.0,
        value=23.0,
        step=0.1
    )


    stress_score = st.slider(
        "Tingkat Stres",
        min_value=1,
        max_value=10,
        value=5
    )


    sleep_latency_mins = st.number_input(
        "Waktu Mulai Tidur (menit)",
        min_value=0,
        max_value=180,
        value=20
    )


    wake_episodes_per_night = st.number_input(
        "Frekuensi Terbangun per Malam",
        min_value=0,
        max_value=20,
        value=1
    )



# ============================================================
# KONVERSI KONDISI MENTAL
# ============================================================

mental_mapping = {

    "Baik": 0,
    "Sedang": 1,
    "Buruk": 2

}


mental_health_encoded = mental_mapping[
    mental_health_condition
]



# ============================================================
# MEMBUAT INPUT MODEL
# ============================================================

input_data = pd.DataFrame([{

    "sleep_quality_score":
        sleep_quality_score,


    "sleep_duration_hrs":
        sleep_duration_hrs,


    "mental_health_condition":
        mental_health_encoded,


    "cognitive_performance_score":
        cognitive_performance_score,


    "bmi":
        bmi,


    "stress_score":
        stress_score,


    "sleep_latency_mins":
        sleep_latency_mins,


    "wake_episodes_per_night":
        wake_episodes_per_night

}])



input_data = input_data[EXPECTED_FEATURES]
# ============================================================
# PREDIKSI MODEL
# ============================================================


st.markdown("---")


if st.button(
    "🔍 Prediksi Risiko Gangguan Tidur",
    use_container_width=True
):


    # Prediksi kelas

    prediction = model.predict(
        input_data
    )[0]


    prediction_name = s.inverse_transform(
        [int(prediction)]
    )[0]



    # Probabilitas

    probability = model.predict_proba(
        input_data
    )[0]


    class_names = label_encoders.classes_



    confidence = probability.max() * 100



    # ========================================================
    # HASIL PREDIKSI
    # ========================================================


    st.markdown(
    """
    <div class="result-card">

    <h2>
    Hasil Prediksi
    </h2>

    </div>
    """,
    unsafe_allow_html=True
    )



    col1, col2 = st.columns(2)



    with col1:

        st.metric(
            "Kategori Risiko",
            prediction_name
        )



    with col2:

        st.metric(
            "Tingkat Keyakinan",
            f"{confidence:.2f}%"
        )



    # ========================================================
    # PENJELASAN HASIL
    # ========================================================


    if prediction_name == "Healthy":

        explanation = """
        Berdasarkan data yang dimasukkan,
        model memprediksi kondisi tidur berada
        dalam kategori sehat.
        """


    elif prediction_name == "Mild":

        explanation = """
        Model memprediksi adanya risiko gangguan tidur
        tingkat ringan. Perbaikan pola tidur dan gaya hidup
        dapat membantu meningkatkan kualitas tidur.
        """


    elif prediction_name == "Moderate":

        explanation = """
        Model menunjukkan risiko gangguan tidur tingkat sedang.
        Perlu perhatian terhadap faktor tidur, stres,
        dan kebiasaan harian.
        """


    else:

        explanation = """
        Model memprediksi risiko gangguan tidur tingkat berat.
        Disarankan melakukan evaluasi lebih lanjut terhadap
        faktor penyebab gangguan tidur.
        """



    st.info(
        explanation
    )



    # ========================================================
    # PROBABILITAS MASING-MASING KELAS
    # ========================================================


    st.markdown(
    """
    ### Probabilitas Prediksi
    """
    )


    probability_df = pd.DataFrame({

        "Kategori":
            class_names,


        "Probabilitas":
            probability,


        "Persentase":
            [
                f"{x*100:.2f}%"
                for x in probability
            ]

    })


    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )



    # ========================================================
    # DATA INPUT
    # ========================================================


    st.markdown(
    """
    ### Data Input Pengguna
    """
    )


    display_input = input_data.copy()


    display_input["mental_health_condition"] = (
        mental_health_condition
    )


    st.dataframe(
        display_input,
        use_container_width=True,
        hide_index=True
    )