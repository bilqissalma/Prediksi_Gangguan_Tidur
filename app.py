# ============================================================
# ANALISIS RISIKO GANGGUAN TIDUR
# RANDOM FOREST CLASSIFIER - MODEL FINAL 8 FITUR
# ============================================================
import os
from datetime import date, datetime, time, timedelta

# Membatasi penggunaan thread agar stabil di Streamlit Cloud
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import joblib
import pandas as pd
import streamlit as st


EXPECTED_FEATURES = [
    "sleep_quality_score",
    "sleep_duration_hrs",
    "mental_health_condition",
    "cognitive_performance_score",
    "bmi",
    "stress_score",
    "sleep_latency_mins",
    "wake_episodes_per_night",
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
        "Indeks Massa Tubuh (BMI)",

    "stress_score":
        "Tingkat Stres",

    "sleep_latency_mins":
        "Waktu Mulai Tidur (menit)",

    "wake_episodes_per_night":
        "Frekuensi Terbangun per Malam"

}

RESULT_LABELS = {
    "healthy": "Risiko Gangguan Tidur Rendah",
    "mild": "Risiko Gangguan Tidur Ringan",
    "moderate": "Risiko Gangguan Tidur Sedang",
    "severe": "Risiko Gangguan Tidur Tinggi",
    "high": "Risiko Gangguan Tidur Tinggi",
}

APP_VERSION = "readability-v6-2026-07-20"

st.set_page_config(
    page_title="Analisis Risiko Gangguan Tidur",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    """Memuat CSS tambahan apabila file tersedia."""
    try:
        with open("style.css", encoding="utf-8") as css_file:
            st.markdown(
                f"<style>{css_file.read()}</style>",
                unsafe_allow_html=True,
            )
    except FileNotFoundError:
        # CSS bersifat opsional. Aplikasi tetap dapat berjalan tanpa file ini.
        pass


load_css()

# CSS tambahan untuk memastikan teks penting tetap terbaca meskipun style.css
# menggunakan warna teks yang terlalu terang. Diletakkan setelah load_css()
# agar aturan ini menjadi prioritas terakhir.
st.markdown(
    """
    <style>
    /* Teks checkbox, termasuk "Saya tidak tidur siang" */
    div[data-testid="stCheckbox"] label,
    div[data-testid="stCheckbox"] label p,
    div[data-testid="stCheckbox"] span {
        color: #0B1F44 !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }

    /* Kotak centang dan tanda centang dibuat lebih tegas */
    div[data-testid="stCheckbox"] input + div {
        border-color: #0B1F44 !important;
    }
    div[data-testid="stCheckbox"] input:checked + div {
        background-color: #0B1F44 !important;
        border-color: #0B1F44 !important;
    }

    /* Seluruh teks di kotak success, warning, error, dan info */
    div[data-testid="stAlert"],
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] li,
    div[data-testid="stAlert"] strong {
        color: #0B1F44 !important;
        opacity: 1 !important;
    }

    /* Catatan di bawah hasil prediksi */
    div[data-testid="stCaptionContainer"] p {
        color: #334155 !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    """Memuat model, label encoder, dan urutan fitur hasil pelatihan."""
    loaded_model = joblib.load("sleep_disorder_rf.joblib")
    loaded_encoders = joblib.load("label_encoders.joblib")
    loaded_features = list(joblib.load("feature_names.joblib"))

    # Membatasi penggunaan CPU saat deployment.
    if hasattr(loaded_model, "n_jobs"):
        loaded_model.n_jobs = 1

    return loaded_model, loaded_encoders, loaded_features


try:
    model, label_encoders, feature_names = load_artifacts()
except FileNotFoundError as error:
    st.error(
        "File model belum lengkap. Pastikan sleep_disorder_rf.joblib, "
        "label_encoders.joblib, dan feature_names.joblib berada di folder "
        "yang sama dengan app.py."
    )
    st.exception(error)
    st.stop()
except Exception as error:
    st.error(
        "Artefak model gagal dimuat. Pastikan file berasal dari proses "
        "pelatihan terbaru dan versi library pada deployment sesuai."
    )
    st.exception(error)
    st.stop()


# Mencegah aplikasi memakai model lama yang masih menggunakan banyak fitur.
if len(feature_names) != 8 or set(feature_names) != set(EXPECTED_FEATURES):
    st.error(
        "feature_names.joblib tidak sesuai dengan model final 8 fitur. "
        "Unggah ulang sleep_disorder_rf.joblib dan feature_names.joblib "
        "yang dibuat setelah pelatihan ulang."
    )
    st.write("Fitur yang terbaca:", feature_names)
    st.stop()

if hasattr(model, "n_features_in_") and model.n_features_in_ != len(feature_names):
    st.error(
        "Jumlah fitur pada model tidak sama dengan jumlah fitur pada "
        "feature_names.joblib. Gunakan artefak dari proses pelatihan yang sama."
    )
    st.stop()


def minutes_between(
    start_time: time,
    end_time: time,
    *,
    equal_means_zero: bool = False,
) -> float:
    """Menghitung selisih menit dan menangani pergantian hari."""
    start_dt = datetime.combine(date.today(), start_time)
    end_dt = datetime.combine(date.today(), end_time)

    if end_dt < start_dt or (end_dt == start_dt and not equal_means_zero):
        end_dt += timedelta(days=1)

    return (end_dt - start_dt).total_seconds() / 60


def calculate_step_one(raw_input: dict | None = None) -> tuple[dict, list[str]]:
    """Menghitung fitur tahap pertama dari salinan input yang permanen."""
    if raw_input is None:
        raw_input = st.session_state.get("step_one_raw")

    if not raw_input:
        return {}, [
            "Data pada Step 1 tidak ditemukan. Silakan kembali dan isi "
            "kembali pola tidur hari kerja."
        ]

    sleep_duration_hrs = minutes_between(
        raw_input["actual_sleep_time"],
        raw_input["wake_time"],
    ) / 60

    sleep_latency_mins = minutes_between(
        raw_input["bed_attempt_time"],
        raw_input["actual_sleep_time"],
        equal_means_zero=True,
    )

    screen_time_before_bed_mins = minutes_between(
        raw_input["last_screen_time"],
        raw_input["bed_attempt_time"],
        equal_means_zero=True,
    )

    errors: list[str] = []

    if not 2 <= sleep_duration_hrs <= 16:
        errors.append(
            "Durasi tidur yang dihitung harus berada antara 2 dan 16 jam. "
            "Periksa kembali waktu mulai tidur dan waktu bangun."
        )

    if not 0 <= sleep_latency_mins <= 300:
        errors.append(
            "Waktu untuk mulai tertidur harus berada antara 0 dan 300 menit. "
            "Periksa kembali urutan waktunya."
        )

    if not 0 <= screen_time_before_bed_mins <= 720:
        errors.append(
            "Jeda penggunaan gadget sebelum tidur harus berada antara "
            "0 dan 720 menit. Periksa kembali urutan waktunya."
        )

    values = {
        "sleep_duration_hrs": round(sleep_duration_hrs, 2),
        "sleep_latency_mins": round(sleep_latency_mins, 2),
        "wake_episodes_per_night": int(raw_input["wake_episodes"]),
        "screen_time_before_bed_mins": round(
            screen_time_before_bed_mins,
            2,
        ),
        "work_hours_that_day": float(raw_input["work_hours"]),
    }

    return values, errors


def calculate_final_input() -> tuple[dict, list[str]]:
    """Menggabungkan seluruh input dan menghitung delapan fitur model."""
    step_one_values = st.session_state.get("step_one_values")
    if not step_one_values:
        return {}, [
            "Data Step 1 tidak tersedia. Silakan kembali ke Step 1 dan isi "
            "kembali data pola tidur hari kerja."
        ]

    errors: list[str] = []
    height_m = float(st.session_state.height_cm) / 100
    weight_kg = float(st.session_state.weight_kg)
    bmi = weight_kg / (height_m**2)

    weekend_duration_hrs = minutes_between(
        st.session_state.weekend_sleep_time,
        st.session_state.weekend_wake_time,
    ) / 60

    weekend_sleep_diff_hrs = abs(
        weekend_duration_hrs - step_one_values["sleep_duration_hrs"]
    )

    if st.session_state.no_nap:
        nap_duration_mins = 0.0
    else:
        nap_duration_mins = minutes_between(
            st.session_state.nap_start_time,
            st.session_state.nap_end_time,
            equal_means_zero=True,
        )

    if not 10 <= bmi <= 80:
        errors.append(
            "BMI yang dihitung berada di luar rentang 10–80. "
            "Periksa kembali tinggi dan berat badan."
        )

    if not 2 <= weekend_duration_hrs <= 16:
        errors.append(
            "Durasi tidur akhir pekan yang dihitung harus berada antara "
            "2 dan 16 jam."
        )

    if not 0 <= weekend_sleep_diff_hrs <= 14:
        errors.append(
            "Selisih durasi tidur hari kerja dan akhir pekan tidak wajar. "
            "Periksa kembali waktu tidur yang dimasukkan."
        )

    if not 0 <= nap_duration_mins <= 360:
        errors.append(
            "Durasi tidur siang harus berada antara 0 dan 360 menit."
        )

    final_input = {
        **step_one_values,
        "bmi": round(bmi, 2),
        "weekend_sleep_diff_hrs": round(weekend_sleep_diff_hrs, 2),
        "nap_duration_mins": round(nap_duration_mins, 2),
    }

    return final_input, errors


def reset_prediction_form() -> None:
    """Menghapus input sebelumnya dan kembali ke tahap pertama."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.step = 1


def decode_target(value):
    """Mengembalikan label target asli apabila encoder tersedia."""
    target_encoder = label_encoders.get("sleep_disorder_risk")
    if target_encoder is None:
        return str(value)
    return str(target_encoder.inverse_transform([value])[0])


with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/sleep.png", width=70)
    st.markdown("## Sleep Disorder AI")
    st.caption("Random Forest — Model Final 8 Fitur")
    st.caption(f"Versi aplikasi: {APP_VERSION}")
    st.divider()
    menu = st.radio("Menu", ["🏠 Prediksi Risiko"])
    st.divider()
    st.caption("Universitas Amikom Yogyakarta")


if menu == "🏠 Prediksi Risiko":
    st.markdown(
        """
        <div class="header">
            <h1>🌙 Analisis Risiko Gangguan Tidur</h1>
            <h4>Random Forest Classification — Model Final 8 Fitur</h4>
            <p>
                Sistem memperkirakan tingkat risiko gangguan tidur menggunakan
                data waktu, durasi, jumlah kejadian, serta data fisik yang
                dapat dimasukkan oleh pengguna tanpa penilaian berbentuk skala.
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
                    ✔️ Ringkasan data yang digunakan model
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if "step" not in st.session_state:
        st.session_state.step = 1

    if st.session_state.step <= 2:
        st.progress(st.session_state.step / 2)
        st.markdown("<br>", unsafe_allow_html=True)

    # ========================================================
    # STEP 1: POLA TIDUR HARI KERJA
    # ========================================================
    if st.session_state.step == 1:
        st.markdown(
            """
            <div class="form-card">
                <h3>Step 1 dari 2</h3>
                <h2>😴 Pola Tidur Hari Kerja</h2>
                <p>
                    Masukkan waktu dan jumlah kejadian yang paling sering
                    menggambarkan pola tidur pada hari kerja atau hari biasa.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        saved_step_one = st.session_state.get("step_one_raw", {})
        col1, col2 = st.columns(2)

        with col1:
            st.time_input(
                "Waktu mulai mencoba tidur",
                value=saved_step_one.get("bed_attempt_time", time(22, 30)),
                key="bed_attempt_time",
                step=timedelta(minutes=1),
                help="Waktu ketika Anda sudah berbaring dan mulai mencoba tidur. Waktu dapat dipilih per menit.",
            )
            st.time_input(
                "Perkiraan waktu benar-benar tertidur",
                value=saved_step_one.get("actual_sleep_time", time(22, 45)),
                key="actual_sleep_time",
                step=timedelta(minutes=1),
            )
            st.time_input(
                "Waktu bangun",
                value=saved_step_one.get("wake_time", time(6, 0)),
                key="wake_time",
                step=timedelta(minutes=1),
            )

        with col2:
            st.time_input(
                "Waktu terakhir menggunakan ponsel/laptop sebelum tidur",
                value=saved_step_one.get("last_screen_time", time(21, 45)),
                key="last_screen_time",
                step=timedelta(minutes=1),
            )
            st.number_input(
                "Berapa kali biasanya terbangun dalam satu malam?",
                min_value=0,
                max_value=30,
                value=int(saved_step_one.get("wake_episodes", 1)),
                step=1,
                key="wake_episodes",
            )
            st.number_input(
                "Berapa jam bekerja atau belajar dalam sehari?",
                min_value=0.0,
                max_value=24.0,
                value=float(saved_step_one.get("work_hours", 8.0)),
                step=0.1,
                format="%.1f",
                key="work_hours",
            )

        st.caption(
            "Durasi tidur, waktu untuk mulai tertidur, dan jeda penggunaan "
            "gadget akan dihitung otomatis oleh sistem. Seluruh input waktu "
            "dapat dipilih dengan ketelitian satu menit."
        )

        st.markdown("<br>", unsafe_allow_html=True)
        _, col_next = st.columns([4, 1])
        with col_next:
            if st.button("Selanjutnya ➜", use_container_width=True):
                step_one_raw = {
                    "bed_attempt_time": st.session_state.get(
                        "bed_attempt_time", time(22, 30)
                    ),
                    "actual_sleep_time": st.session_state.get(
                        "actual_sleep_time", time(22, 45)
                    ),
                    "wake_time": st.session_state.get(
                        "wake_time", time(6, 0)
                    ),
                    "last_screen_time": st.session_state.get(
                        "last_screen_time", time(21, 45)
                    ),
                    "wake_episodes": st.session_state.get("wake_episodes", 1),
                    "work_hours": st.session_state.get("work_hours", 8.0),
                }

                step_one_values, validation_errors = calculate_step_one(
                    step_one_raw
                )
                if validation_errors:
                    for message in validation_errors:
                        st.error(message)
                else:
                    # Simpan ke key permanen. Key widget dapat dibersihkan
                    # Streamlit ketika Step 1 tidak lagi dirender.
                    st.session_state.step_one_raw = step_one_raw
                    st.session_state.step_one_values = step_one_values
                    st.session_state.step = 2
                    st.rerun()

    # ========================================================
    # STEP 2: DATA FISIK DAN POLA AKHIR PEKAN
    # ========================================================
    elif st.session_state.step == 2:
        st.markdown(
            """
            <div class="form-card">
                <h3>Step 2 dari 2</h3>
                <h2>⚖️ Data Fisik dan Pola Akhir Pekan</h2>
                <p>
                    Masukkan tinggi, berat badan, waktu tidur akhir pekan,
                    dan informasi tidur siang.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        step_one_preview = st.session_state.get("step_one_values")
        if not step_one_preview:
            st.warning(
                "Data Step 1 tidak ditemukan pada sesi ini. "
                "Aplikasi akan kembali ke Step 1."
            )
            st.session_state.step = 1
            st.rerun()

        preview_col1, preview_col2, preview_col3 = st.columns(3)
        preview_col1.metric(
            "Durasi tidur hari kerja",
            f"{step_one_preview['sleep_duration_hrs']:.2f} jam",
        )
        preview_col2.metric(
            "Waktu untuk tertidur",
            f"{step_one_preview['sleep_latency_mins']:.0f} menit",
        )
        preview_col3.metric(
            "Jeda gadget sebelum tidur",
            f"{step_one_preview['screen_time_before_bed_mins']:.0f} menit",
        )

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.number_input(
                "Berat badan (kg)",
                min_value=20.0,
                max_value=300.0,
                value=60.0,
                step=0.1,
                format="%.1f",
                key="weight_kg",
            )
            st.number_input(
                "Tinggi badan (cm)",
                min_value=100.0,
                max_value=250.0,
                value=165.0,
                step=0.1,
                format="%.1f",
                key="height_cm",
            )
            st.caption(
                "BMI akan dihitung otomatis dari berat dan tinggi badan. "
                "Berat dan tinggi dapat dimasukkan dengan ketelitian 0,1."
            )

        with col2:
            st.time_input(
                "Waktu mulai tidur pada akhir pekan",
                value=time(23, 30),
                key="weekend_sleep_time",
                step=timedelta(minutes=1),
            )
            st.time_input(
                "Waktu bangun pada akhir pekan",
                value=time(7, 30),
                key="weekend_wake_time",
                step=timedelta(minutes=1),
            )
            st.caption(
                "Sistem menghitung selisih absolut antara durasi tidur "
                "hari kerja dan akhir pekan."
            )

        st.divider()
        st.checkbox(
            "Saya tidak tidur siang",
            value=False,
            key="no_nap",
        )

        if not st.session_state.no_nap:
            nap_col1, nap_col2 = st.columns(2)
            with nap_col1:
                st.time_input(
                    "Waktu mulai tidur siang",
                    value=time(13, 0),
                    key="nap_start_time",
                    step=timedelta(minutes=1),
                )
            with nap_col2:
                st.time_input(
                    "Waktu selesai tidur siang",
                    value=time(13, 30),
                    key="nap_end_time",
                    step=timedelta(minutes=1),
                )

        st.divider()
        col_back, col_predict = st.columns(2)
        with col_back:
            if st.button(
                "⬅ Kembali",
                use_container_width=True,
                key="back_step2",
            ):
                st.session_state.step = 1
                st.rerun()

        with col_predict:
            if st.button(
                "🔍 Lakukan Prediksi",
                use_container_width=True,
                key="predict_button",
            ):
                final_input, validation_errors = calculate_final_input()

                if validation_errors:
                    for message in validation_errors:
                        st.error(message)
                else:
                    st.session_state.final_input = final_input
                    st.session_state.step = 3
                    st.rerun()

    # ========================================================
    # STEP 3: HASIL PREDIKSI
    # ========================================================
    elif st.session_state.step == 3:
        if "final_input" not in st.session_state:
            st.session_state.step = 1
            st.rerun()

        st.markdown(
            """
            <div class="form-card">
                <h2>🎯 Hasil Prediksi</h2>
                <p>
                    Berikut merupakan hasil analisis risiko gangguan tidur
                    berdasarkan delapan fitur yang telah dihitung.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        raw_model_input = st.session_state.final_input

        missing_features = [
            feature
            for feature in feature_names
            if feature not in raw_model_input
        ]
        if missing_features:
            st.error(
                "Input aplikasi belum menyediakan fitur berikut: "
                + ", ".join(missing_features)
            )
            st.stop()

        # Menyusun dataframe sesuai urutan fitur yang tersimpan bersama model.
        X = pd.DataFrame(
            [{feature: raw_model_input[feature] for feature in feature_names}],
            columns=feature_names,
        ).astype(float)

        try:
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]
        except Exception as error:
            st.error(
                "Prediksi gagal. Pastikan model dan feature_names.joblib "
                "berasal dari proses pelatihan delapan fitur yang sama."
            )
            st.exception(error)
            st.stop()

        original_result = decode_target(prediction)
        result_key = original_result.strip().lower()
        display_result = RESULT_LABELS.get(result_key, original_result)

        if result_key == "healthy":
            warna, icon = "#22C55E", "🟢"
        elif result_key == "mild":
            warna, icon = "#FACC15", "🟡"
        elif result_key == "moderate":
            warna, icon = "#FB923C", "🟠"
        else:
            warna, icon = "#EF4444", "🔴"

        st.markdown(
            f"""
            <div style="background:white;padding:25px;border-radius:18px;border-left:8px solid {warna};box-shadow:0px 6px 18px rgba(0,0,0,.08);margin-bottom:25px;">
                <h2 style="color:{warna};margin-bottom:10px;">{icon} {display_result}</h2>
                <p style="font-size:17px;color:#5B6475;line-height:1.8;">
                    Berdasarkan delapan data pola tidur dan kebiasaan harian
                    yang dimasukkan, model memperkirakan hasil Anda berada pada
                    kategori <b>{display_result}</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<h2 class="prediction-section">📈 Probabilitas Setiap Kategori</h2>',
            unsafe_allow_html=True,
        )

        class_names = [decode_target(class_value) for class_value in model.classes_]
        for class_name, score in zip(class_names, probability):
            class_key = class_name.strip().lower()
            class_display = RESULT_LABELS.get(class_key, class_name)
            st.markdown(
                f'<div class="prediction-label">{class_display}</div>',
                unsafe_allow_html=True,
            )
            st.progress(float(score))
            st.markdown(
                f'<div class="prediction-percent">{score * 100:.2f}%</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        with st.expander("📋 Lihat Data yang Digunakan Model"):
            display_data = pd.DataFrame(
                {
                    "Fitur": [FEATURE_LABELS[feature] for feature in feature_names],
                    "Nilai": [raw_model_input[feature] for feature in feature_names],
                }
            )
            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True,
            )

        st.markdown(
            '<h2 class="prediction-section">💡 Apa Arti Hasil Ini?</h2>',
            unsafe_allow_html=True,
        )

        if result_key == "healthy":
            st.success(
                "**Risiko gangguan tidur rendah.** Berdasarkan delapan data yang "
                "dimasukkan, pola Anda lebih banyak menyerupai kelompok dengan "
                "risiko rendah pada data pelatihan model. Pertahankan waktu tidur "
                "dan bangun yang teratur, durasi tidur yang cukup, serta batasi "
                "penggunaan gadget menjelang tidur. Apabila Anda tetap sering "
                "sulit tidur, terbangun berulang kali, atau sangat mengantuk pada "
                "siang hari, pertimbangkan berkonsultasi dengan tenaga kesehatan."
            )
        elif result_key == "mild":
            st.warning(
                "**Risiko gangguan tidur ringan.** Artinya, beberapa pola pada "
                "data yang dimasukkan mulai menyerupai karakteristik kelompok yang "
                "memiliki gangguan tidur, tetapi tingkat risikonya belum termasuk "
                "sedang atau tinggi. Perhatikan keteraturan waktu tidur dan bangun, "
                "cukupkan durasi tidur, kurangi penggunaan gadget sebelum tidur, "
                "dan amati seberapa sering Anda terbangun pada malam hari. Jika "
                "keluhan terjadi berulang atau mulai mengganggu aktivitas, "
                "pertimbangkan konsultasi dengan tenaga kesehatan."
            )
        elif result_key == "moderate":
            st.warning(
                "**Risiko gangguan tidur sedang.** Beberapa pola tidur dan kebiasaan "
                "harian yang dimasukkan cukup menyerupai kelompok dengan risiko "
                "gangguan tidur sedang pada data pelatihan. Sebaiknya evaluasi "
                "jadwal tidur, durasi tidur, penggunaan gadget, tidur siang, dan "
                "frekuensi terbangun. Catat pola tidur selama beberapa hari dan "
                "pertimbangkan berkonsultasi dengan tenaga kesehatan apabila "
                "keluhan menetap atau mengganggu kegiatan sehari-hari."
            )
        else:
            st.error(
                "**Risiko gangguan tidur tinggi.** Pola data yang dimasukkan banyak "
                "menyerupai kelompok dengan risiko tinggi pada data pelatihan model. "
                "Hasil ini bukan diagnosis, tetapi menjadi tanda bahwa pola tidur "
                "perlu diperhatikan lebih serius. Disarankan berkonsultasi dengan "
                "tenaga kesehatan, terutama apabila Anda mengalami sulit tidur "
                "berkepanjangan, sering terbangun, mendengkur keras, terbangun "
                "seperti kehabisan napas, atau mengantuk berat pada siang hari."
            )

        st.caption(
            "Hasil aplikasi merupakan prediksi model, bukan diagnosis medis. "
            "Probabilitas menunjukkan keyakinan model terhadap kelas prediksi, "
            "bukan kepastian kondisi kesehatan."
        )
        st.divider()

        if st.button("🔄 Prediksi Lagi", use_container_width=True):
            reset_prediction_form()
            st.rerun()