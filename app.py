# ============================================================
# ANALISIS RISIKO GANGGUAN TIDUR
# RANDOM FOREST CLASSIFIER - MODEL B (TANPA DEMOGRAFI)
# ============================================================
import os

# Membatasi penggunaan thread agar stabil di Streamlit Cloud
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Analisis Risiko Gangguan Tidur",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    try:
        with open("style.css", encoding="utf-8") as css_file:
            st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("File style.css tidak ditemukan. Aplikasi tetap dijalankan tanpa CSS khusus.")


load_css()


@st.cache_resource
def load_artifacts():

    model = joblib.load("sleep_disorder_rf.joblib")
    label_encoders = joblib.load("label_encoders.joblib")
    feature_names = joblib.load("feature_names.joblib")

    # Model disimpan dengan n_jobs=-1.
    # Untuk deployment, batasi agar tidak memakai semua CPU.
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1

    return model, label_encoders, feature_names


try:
    model, label_encoders, feature_names = load_artifacts()
except FileNotFoundError as error:
    st.error(
        "Pastikan sleep_disorder_rf.joblib, label_encoders.joblib, "
        "feature_names.joblib, dan style.css berada di folder yang sama dengan app.py."
    )
    st.exception(error)
    st.stop()


def get_encoder_options(column_name: str, fallback: list[str]) -> list[str]:
    if column_name in label_encoders:
        return list(label_encoders[column_name].classes_)
    return fallback


def encode_input(input_data: dict) -> dict:
    encoded_data = input_data.copy()
    for column_name, encoder in label_encoders.items():
        if column_name == "sleep_disorder_risk":
            continue
        if column_name in encoded_data:
            encoded_data[column_name] = encoder.transform(
                [str(encoded_data[column_name])]
            )[0]
    return encoded_data


def reset_prediction_form() -> None:
    for key in list(st.session_state.keys()):
        if key != "step":
            del st.session_state[key]
    st.session_state.step = 1


with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/sleep.png", width=70)
    st.markdown("## Sleep Disorder AI")
    st.caption("Random Forest Classifier — Model B")
    st.divider()
    menu = st.radio("Menu", ["🏠 Prediksi Risiko"])
    st.divider()
    st.caption("Universitas Amikom Yogyakarta")


if menu == "🏠 Prediksi Risiko":
    st.markdown(
        """
        <div class="header">
            <h1>🌙 Analisis Risiko Gangguan Tidur</h1>
            <h4>Random Forest Classification — Model B</h4>
            <p>
                Sistem klasifikasi tingkat risiko gangguan tidur berdasarkan
                gaya hidup, pola tidur, dan kondisi kesehatan.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="dashboard-card">
                <h2>🎯 Tujuan</h2>
                <p>Membantu mengidentifikasi tingkat risiko gangguan tidur menggunakan algoritma Random Forest.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="dashboard-card">
                <h2>📊 Output</h2>
                <p>
                    ✔️ Kategori risiko gangguan tidur<br>
                    ✔️ Probabilitas setiap kelas<br>
                    ✔️ Interpretasi hasil prediksi
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if "step" not in st.session_state:
        st.session_state.step = 1

    if st.session_state.step <= 3:
        st.progress(st.session_state.step / 3)
        st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.step == 1:
        st.markdown(
            """
            <div class="form-card">
                <h3>Step 1 dari 3</h3>
                <h2>🏃 Data Gaya Hidup</h2>
                <p>Silakan lengkapi informasi mengenai gaya hidup pengguna.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Jumlah Kafein yang Dikonsumsi Sebelum Tidur (mg)", min_value=0.0, value=100.0, step=10.0, key="caffeine")
            st.number_input("Jumlah Alkohol yang Dikonsumsi Sebelum Tidur (Unit)", min_value=0.0, value=0.0, step=0.5, key="alcohol")
            st.number_input("Durasi Penggunaan Gadget Sebelum Tidur (Menit)", min_value=0.0, value=60.0, step=5.0, key="screen")
        with col2:
            st.selectbox(
                "Hari Terakhir Melakukan Olahraga",
                get_encoder_options("exercise_day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
                key="exercise",
            )
            st.number_input("Jumlah Langkah Kaki Hari Ini", min_value=0, value=5000, step=100, key="steps")

        st.markdown("<br>", unsafe_allow_html=True)
        _, col_next = st.columns([4, 1])
        with col_next:
            if st.button("Selanjutnya ➜", use_container_width=True):
                st.session_state.step = 2
                st.rerun()

    elif st.session_state.step == 2:
        st.markdown(
            """
            <div class="form-card">
                <h3>Step 2 dari 3</h3>
                <h2>😴 Pola Tidur</h2>
                <p>Silakan lengkapi informasi mengenai pola tidur pengguna.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Rata-rata Lama Tidur per Malam (Jam)", min_value=0.0, max_value=24.0, value=7.0, step=0.5, key="sleep_duration")
            st.number_input("Nilai Kualitas Tidur (0–10)", min_value=0, max_value=10, value=7, key="sleep_quality")
            st.number_input("Persentase Tidur REM (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.1, key="rem")
            st.number_input("Persentase Tidur nyenyak/Deep Sleep (%)", min_value=0.0, max_value=100.0, value=18.0, step=0.1, key="deep_sleep")
            st.number_input("Lama Tidur Siang (Menit)", min_value=0.0, value=20.0, step=5.0, key="nap")
        with col2:
            st.number_input("Lama Waktu untuk Mulai Tidur (Menit)", min_value=0.0, value=15.0, step=1.0, key="latency")
            st.number_input("Jumlah Terbangun Saat Tidur (Kali)", min_value=0, value=1, step=1, key="wake")
            st.selectbox("Kebiasaan Waktu Tidur", get_encoder_options("chronotype", ["Pagi", "Normal", "Malam"]), key="chronotype")
            st.number_input("Perbedaan Lama Tidur Saat Akhir Pekan (Jam)", value=1.0, step=0.5, key="weekend_sleep")
            st.selectbox("Apakah Merasa Segar Setelah Bangun?", get_encoder_options("felt_rested", ["No", "Yes"]), key="felt_rested")

        st.divider()
        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("⬅ Kembali", use_container_width=True, key="back_step2"):
                st.session_state.step = 1
                st.rerun()
        with col_next:
            if st.button("Selanjutnya ➜", use_container_width=True, key="next_step2"):
                st.session_state.step = 3
                st.rerun()

    elif st.session_state.step == 3:
        st.markdown(
            """
            <div class="form-card">
                <h3>Step 3 dari 3</h3>
                <h2>❤️ Kondisi Kesehatan</h2>
                <p>Lengkapi informasi kesehatan dan kondisi pendukung tidur.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Indeks Massa Tubuh (BMI)", min_value=5.0, max_value=80.0, value=22.5, step=0.1, key="bmi")
            st.number_input("Tingkat Stres (Skala 0–10)", min_value=0, max_value=10, value=5, key="stress")
            st.selectbox("Kondisi Kesehatan Mental Saat Ini", get_encoder_options("mental_health_condition", ["None"]), key="mental")
            st.number_input("Detak Jantung Saat Istirahat (BPM)", min_value=20, max_value=250, value=72, key="heart")
            st.selectbox("Apakah Menggunakan Obat Tidur?", get_encoder_options("sleep_aid_used", ["No", "Yes"]), key="sleep_aid")
        with col2:
            st.selectbox("Apakah Bekerja dengan Sistem Shift?", get_encoder_options("shift_work", ["No", "Yes"]), key="shift_work")
            st.number_input("Suhu Kamar Saat Tidur (°C)", min_value=0.0, max_value=50.0, value=24.0, step=0.5, key="room")
            st.number_input("Lama Jam Kerja Hari Ini (Jam)", min_value=0.0, max_value=24.0, value=8.0, step=0.5, key="work")
            st.number_input("Skor Kemampuan Konsentrasi (0–100)", min_value=0, max_value=100, value=75, key="cognitive")

        if "season" in feature_names or "day_type" in feature_names:
            st.divider()
            st.markdown(
                """
                <div class="form-card">
                    <h2>🌙 Informasi Tambahan</h2>
                    <p>Lengkapi informasi tambahan yang digunakan oleh model.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            extra_col1, extra_col2 = st.columns(2)
            with extra_col1:
                if "season" in feature_names:
                    st.selectbox("Musim", get_encoder_options("season", ["Summer"]), key="season")
            with extra_col2:
                if "day_type" in feature_names:
                    st.selectbox("Jenis Hari", get_encoder_options("day_type", ["Weekday"]), key="day_type")

        st.divider()
        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("⬅ Kembali", use_container_width=True, key="back_step3"):
                st.session_state.step = 2
                st.rerun()
        with col_next:
            if st.button("🔍 Lakukan Prediksi", use_container_width=True):
                st.session_state.step = 4
                st.rerun()

    elif st.session_state.step == 4:
        st.markdown(
            """
            <div class="form-card">
                <h2>🎯 Hasil Prediksi</h2>
                <p>Berikut merupakan hasil analisis risiko gangguan tidur berdasarkan data yang telah dimasukkan.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        input_data = {
            "caffeine_mg_before_bed": st.session_state.get("caffeine", 100.0),
            "alcohol_units_before_bed": st.session_state.get("alcohol", 0.0),
            "screen_time_before_bed_mins": st.session_state.get("screen", 60.0),
            "exercise_day": st.session_state.get("exercise", "Monday"),
            "steps_that_day": st.session_state.get("steps", 5000),
            "sleep_duration_hrs": st.session_state.get("sleep_duration", 7.0),
            "sleep_quality_score": st.session_state.get("sleep_quality", 7),
            "rem_percentage": st.session_state.get("rem", 20.0),
            "deep_sleep_percentage": st.session_state.get("deep_sleep", 18.0),
            "sleep_latency_mins": st.session_state.get("latency", 15.0),
            "wake_episodes_per_night": st.session_state.get("wake", 1),
            "nap_duration_mins": st.session_state.get("nap", 20.0),
            "chronotype": st.session_state.get("chronotype", "Morning"),
            "weekend_sleep_diff_hrs": st.session_state.get("weekend_sleep", 1.0),
            "felt_rested": st.session_state.get("felt_rested", "Yes"),
            "bmi": st.session_state.get("bmi", 22.5),
            "stress_score": st.session_state.get("stress", 5),
            "mental_health_condition": st.session_state.get("mental", "None"),
            "heart_rate_resting_bpm": st.session_state.get("heart", 72),
            "sleep_aid_used": st.session_state.get("sleep_aid", "No"),
            "shift_work": st.session_state.get("shift_work", "No"),
            "room_temperature_celsius": st.session_state.get("room", 24.0),
            "work_hours_that_day": st.session_state.get("work", 8.0),
            "cognitive_performance_score": st.session_state.get("cognitive", 75),
            "season": st.session_state.get("season", "Summer"),
            "day_type": st.session_state.get("day_type", "Weekday"),
        }

        raw_model_input = {feature: input_data[feature] for feature in feature_names if feature in input_data}
        missing_features = [feature for feature in feature_names if feature not in raw_model_input]
        if missing_features:
            st.error("Input aplikasi belum menyediakan fitur berikut: " + ", ".join(missing_features))
            st.stop()

        # =====================================================
        # ENCODING INPUT
        # =====================================================

        try:
            encoded_input = encode_input(raw_model_input)

        except ValueError as error:
            st.error(str(error))
            st.stop()


        # =====================================================
        # ENCODING KHUSUS EXERCISE DAY
        # =====================================================

        exercise_map = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6
        }

        if "exercise_day" in encoded_input:
            value = encoded_input["exercise_day"]

            if isinstance(value, str):
                encoded_input["exercise_day"] = exercise_map[value]


        # =====================================================
        # ENCODING KHUSUS YES / NO
        # =====================================================

        yes_no_map = {
            "No": 0,
            "Yes": 1
        }

        yes_no_columns = [
            "sleep_aid_used",
            "shift_work",
            "felt_rested"
        ]

        for column in yes_no_columns:

            if column in encoded_input:

                value = encoded_input[column]

                if isinstance(value, str):

                    if value not in yes_no_map:
                        st.error(
                            f"Nilai pada {column} tidak dikenali: {value}"
                        )
                        st.stop()

                    encoded_input[column] = yes_no_map[value]


        # =====================================================
        # MEMBUAT DATAFRAME SESUAI URUTAN FITUR MODEL
        # =====================================================

        X = pd.DataFrame([encoded_input])

        X = X[feature_names]

        try:
            X = X.astype(float)

        except ValueError as error:

            st.error(
                "Masih terdapat input kategorikal yang belum dikonversi menjadi angka."
            )

            # Menampilkan hanya fitur yang masih berupa teks
            categorical_columns = X.select_dtypes(
                include=["object"]
            ).columns.tolist()

            st.write("Kolom yang masih berupa teks:", categorical_columns)
            st.dataframe(X[categorical_columns])

            st.exception(error)
            st.stop()


        # =====================================================
        # PREDIKSI MODEL
        # =====================================================

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]

        target_encoder = label_encoders["sleep_disorder_risk"]
        hasil = target_encoder.inverse_transform([prediction])[0]
        result_name = str(hasil).lower()

        if result_name == "healthy":
            warna, icon = "#22C55E", "🟢"
        elif result_name == "mild":
            warna, icon = "#FACC15", "🟡"
        elif result_name == "moderate":
            warna, icon = "#FB923C", "🟠"
        else:
            warna, icon = "#EF4444", "🔴"

        st.markdown(
            f"""
            <div style="background:white;padding:25px;border-radius:18px;border-left:8px solid {warna};box-shadow:0px 6px 18px rgba(0,0,0,.08);margin-bottom:25px;">
                <h2 style="color:{warna};margin-bottom:10px;">{icon} {hasil}</h2>
                <p style="font-size:17px;color:#5B6475;line-height:1.8;">
                    Berdasarkan hasil klasifikasi Random Forest Model B, pengguna termasuk dalam kategori risiko <b>{hasil}</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<h2 class="prediction-section">📈 Probabilitas Setiap Kategori</h2>', unsafe_allow_html=True)
        class_names = target_encoder.inverse_transform(model.classes_)
        for class_name, score in zip(class_names, probability):
            st.markdown(f'<div class="prediction-label">{class_name}</div>', unsafe_allow_html=True)
            st.progress(float(score))
            st.markdown(f'<div class="prediction-percent">{score * 100:.2f}%</div>', unsafe_allow_html=True)

        st.divider()
        with st.expander("📋 Lihat Data yang Digunakan Model"):
            st.dataframe(pd.DataFrame([raw_model_input], columns=feature_names), use_container_width=True)

        st.markdown('<h2 class="prediction-section">💡 Interpretasi</h2>', unsafe_allow_html=True)
        if result_name == "healthy":
            st.success("Pengguna memiliki risiko gangguan tidur yang rendah. Pertahankan pola tidur, gaya hidup, dan kondisi kesehatan yang baik.")
        elif result_name == "mild":
            st.warning("Pengguna mulai menunjukkan beberapa faktor risiko. Perbaiki pola tidur dan kurangi faktor risiko.")
        elif result_name == "moderate":
            st.warning("Risiko gangguan tidur berada pada tingkat sedang. Perhatikan kualitas tidur, tingkat stres, dan gaya hidup secara lebih serius.")
        else:
            st.error("Risiko gangguan tidur berada pada tingkat tinggi. Disarankan berkonsultasi dengan tenaga kesehatan untuk evaluasi lebih lanjut.")

        st.caption("Hasil aplikasi merupakan prediksi model dan bukan diagnosis medis.")
        st.divider()
        if st.button("🔄 Prediksi Lagi", use_container_width=True):
            reset_prediction_form()
            st.rerun()
