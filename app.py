# ============================================================
# ANALISIS RISIKO GANGGUAN TIDUR
# RANDOM FOREST - MODEL FINAL 8 FITUR
# STREAMLIT DEPLOYMENT
# ============================================================


import joblib
import pandas as pd
import streamlit as st


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Analisis Risiko Gangguan Tidur",
    page_icon="🌙",
    layout="wide"
)



# ============================================================
# FITUR FINAL MODEL
# ============================================================

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



FEATURE_LABELS = {

    "sleep_quality_score":
        "Kualitas Tidur",

    "sleep_duration_hrs":
        "Durasi Tidur (jam)",

    "mental_health_condition":
        "Kondisi Mental",

    "cognitive_performance_score":
        "Performa Kognitif",

    "bmi":
        "BMI",

    "stress_score":
        "Tingkat Stres",

    "sleep_latency_mins":
        "Waktu Mulai Tidur (menit)",

    "wake_episodes_per_night":
        "Frekuensi Terbangun per Malam"

}



RESULT_LABELS = {

    "healthy":
        "Risiko Gangguan Tidur Rendah",

    "mild":
        "Risiko Gangguan Tidur Ringan",

    "moderate":
        "Risiko Gangguan Tidur Sedang",

    "severe":
        "Risiko Gangguan Tidur Tinggi"

}



# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "sleep_disorder_rf.joblib"
    )


    label_encoders = joblib.load(
        "label_encoders.joblib"
    )


    feature_names = list(
        joblib.load(
            "feature_names.joblib"
        )
    )


    return (
        model,
        label_encoders,
        feature_names
    )



try:

    model, label_encoders, feature_names = load_model()


except Exception as error:

    st.error(
        "Model gagal dimuat. "
        "Pastikan file .joblib berada dalam folder yang sama."
    )

    st.exception(error)

    st.stop()



# ============================================================
# VALIDASI MODEL 8 FITUR
# ============================================================


if set(feature_names) != set(EXPECTED_FEATURES):

    st.error(
        "Fitur model tidak sesuai dengan model final 8 fitur."
    )

    st.write(
        "Fitur terbaca:",
        feature_names
    )

    st.stop()



if hasattr(model, "n_features_in_"):

    if model.n_features_in_ != 8:

        st.error(
            "Jumlah fitur model bukan 8."
        )

        st.stop()



# ============================================================
# FUNGSI DECODE HASIL
# ============================================================


def decode_target(value):

    encoder = label_encoders.get(
        "sleep_disorder_risk"
    )


    if encoder:

        return encoder.inverse_transform(
            [value]
        )[0]


    return value



# ============================================================
# HEADER WEBSITE
# ============================================================


st.title(
    "🌙 Analisis Risiko Gangguan Tidur"
)


st.write(
    """
    Sistem prediksi risiko gangguan tidur menggunakan
    algoritma Random Forest berdasarkan 8 fitur utama.
    """
)



st.divider()



# ============================================================
# INPUT USER
# ============================================================


st.subheader(
    "Masukkan Data Pengguna"
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

        max_value=24.0,

        value=7.0,

        step=0.5

    )


    mental_health_condition = st.selectbox(

        "Kondisi Mental",

        options=[0,1],

        format_func=lambda x:
            "Baik" if x == 0 else "Memiliki Gangguan"

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

        max_value=60.0,

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

        max_value=300,

        value=20

    )


    wake_episodes_per_night = st.number_input(

        "Frekuensi Terbangun per Malam",

        min_value=0,

        max_value=20,

        value=1

    )# ============================================================
# MEMBUAT DATAFRAME INPUT
# ============================================================

if st.button(
    "🔍 Lakukan Prediksi",
    use_container_width=True
):


    input_data = pd.DataFrame([{

        "sleep_quality_score":
            sleep_quality_score,

        "sleep_duration_hrs":
            sleep_duration_hrs,

        "mental_health_condition":
            mental_health_condition,

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



    # Menyesuaikan urutan fitur dengan model

    input_data = input_data[
        feature_names
    ]



    # ========================================================
    # PROSES PREDIKSI
    # ========================================================

    try:

        prediction = model.predict(
            input_data
        )[0]


        probability = model.predict_proba(
            input_data
        )[0]



    except Exception as error:

        st.error(
            "Prediksi gagal dilakukan."
        )

        st.exception(error)

        st.stop()



    # ========================================================
    # KONVERSI HASIL PREDIKSI
    # ========================================================

    result = decode_target(
        prediction
    )


    result_key = str(
        result
    ).lower()



    display_result = RESULT_LABELS.get(
        result_key,
        result
    )



    # ========================================================
    # MENAMPILKAN HASIL
    # ========================================================

    st.divider()


    st.subheader(
        "🎯 Hasil Prediksi"
    )



    if result_key == "healthy":

        icon = "🟢"


    elif result_key == "mild":

        icon = "🟡"


    elif result_key == "moderate":

        icon = "🟠"


    else:

        icon = "🔴"



    st.success(

        f"{icon} {display_result}"

    )



    # ========================================================
    # PROBABILITAS SETIAP KELAS
    # ========================================================


    st.subheader(
        "📊 Probabilitas Prediksi"
    )



    class_names = [

        decode_target(
            kelas
        )

        for kelas in model.classes_

    ]



    probability_table = pd.DataFrame({

        "Kategori":

            [

                RESULT_LABELS.get(
                    str(k).lower(),
                    k
                )

                for k in class_names

            ],


        "Probabilitas":

            [

                f"{nilai*100:.2f}%"

                for nilai in probability

            ]

    })



    st.dataframe(

        probability_table,

        use_container_width=True,

        hide_index=True

    )



    # ========================================================
    # DATA YANG DIGUNAKAN MODEL
    # ========================================================


    st.subheader(
        "📋 Data Input"
    )



    display_data = pd.DataFrame({

        "Fitur":

            [

                FEATURE_LABELS[
                    fitur
                ]

                for fitur in feature_names

            ],


        "Nilai":

            [

                input_data.iloc[0][fitur]

                for fitur in feature_names

            ]

    })



    st.dataframe(

        display_data,

        use_container_width=True,

        hide_index=True

    )



    # ========================================================
    # INTERPRETASI SINGKAT
    # ========================================================


    st.subheader(
        "💡 Interpretasi"
    )



    if result_key == "healthy":

        st.info(
            """
            Hasil prediksi menunjukkan risiko gangguan tidur rendah
            berdasarkan pola tidur dan karakteristik yang dimasukkan.
            """
        )


    elif result_key == "mild":

        st.warning(
            """
            Hasil prediksi menunjukkan risiko gangguan tidur ringan.
            Beberapa pola tidur memiliki kemiripan dengan kelompok
            yang mengalami gangguan tidur ringan.
            """
        )


    elif result_key == "moderate":

        st.warning(
            """
            Hasil prediksi menunjukkan risiko gangguan tidur sedang.
            Disarankan untuk memperhatikan pola tidur dan faktor
            yang dapat memengaruhi kualitas tidur.
            """
        )


    else:

        st.error(
            """
            Hasil prediksi menunjukkan risiko gangguan tidur tinggi.
            Hasil ini merupakan prediksi model dan bukan diagnosis medis.
            """
        )



    st.caption(
        """
        Hasil prediksi merupakan keluaran model machine learning
        dan tidak menggantikan pemeriksaan tenaga kesehatan.
        """
    )