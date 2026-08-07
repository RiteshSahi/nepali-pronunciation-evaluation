import os
import streamlit as st
import pandas as pd

from predict import predict

# =====================================================
# Page Config
# =====================================================

st.set_page_config(
    page_title="Nepali Pronunciation Evaluation",
    page_icon="🎤",
    layout="centered"
)

# =====================================================
# Paths
# =====================================================

TEMP_FOLDER = "../temp"

os.makedirs(
    TEMP_FOLDER,
    exist_ok=True
)

# =====================================================
# Title
# =====================================================

st.title("🎤 Nepali Pronunciation Evaluation System")

st.write(
    "Evaluate your Nepali pronunciation using AI."
)

st.divider()

# =====================================================
# Gender
# =====================================================

gender = st.radio(
    "Select Gender",
    [
        "Male",
        "Female"
    ],
    horizontal=True
)

st.divider()

# =====================================================
# Reference Folder (Gender-Based)
# =====================================================

REFERENCE_FOLDER = f"../dataset/{gender.lower()}/app_reference/"

# =====================================================
# Load Sentences
# =====================================================

df = pd.read_csv(
    os.path.join(
        REFERENCE_FOLDER,
        "sentences.csv"
    )
)

# =====================================================
# Sentence Selection
# =====================================================

voice = st.selectbox(
    "Select Sentence",
    df["audio_id"]
)

row = df[
    df["audio_id"] == voice
].iloc[0]

sentence = row["sentence"]

st.subheader("Sentence")

st.info(sentence)

# =====================================================
# Reference Audio
# =====================================================

reference_audio = os.path.join(
    REFERENCE_FOLDER,
    f"{voice}.wav"
)

st.subheader("Reference Pronunciation")

st.audio(reference_audio)

st.divider()

# =====================================================
# User Recording
# =====================================================

st.subheader("🎤 Record Your Pronunciation")

audio = st.audio_input(
    "Click the microphone and record."
)

audio_path = None

if audio is not None:

    audio_path = os.path.join(
        TEMP_FOLDER,
        "user.wav"
    )

    with open(
        audio_path,
        "wb"
    ) as f:

        f.write(audio.read())

    st.success("Recording completed.")

    st.audio(audio_path)

st.divider()

# =====================================================
# Evaluation
# =====================================================

if st.button(
    "Evaluate Pronunciation",
    use_container_width=True
):

    if audio_path is None:

        st.error(
            "Please record your pronunciation first."
        )

    else:

        with st.spinner(
            "Evaluating pronunciation..."
        ):

            prediction, confidence, features = predict(
                audio_path,
                voice,
                gender.lower()
            )

            dtw = features["dtw"]
            duration = features["duration"]
            wer = features["wer"]
            cer = features["cer"]
            zcr = features.get("zcr", 0)

        st.success("Evaluation Completed")

        st.divider()

        # ----------------------------------------------
        # Result
        # ----------------------------------------------

        if prediction == "Good":

            st.success(
                "✅ Good Pronunciation"
            )

            st.info(
                "Your pronunciation is close to the native reference."
            )

        else:

            st.error(
                "❌ Bad Pronunciation"
            )

            st.warning(
                "Your pronunciation differs from the reference. Listen to the reference audio and try again."
            )

        # ----------------------------------------------
        # Confidence
        # ----------------------------------------------

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.divider()

        # ----------------------------------------------
        # Technical Details
        # ----------------------------------------------

        with st.expander(
            "Show Technical Details"
        ):

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "DTW Distance",
                    f"{dtw:.4f}"
                )

                st.metric(
                    "WER",
                    f"{wer:.4f}"
                )

            with col2:

                st.metric(
                    "Duration Difference",
                    f"{duration:.2f} sec"
                )

                st.metric(
                    "CER",
                    f"{cer:.4f}"
                )

            st.metric(
                "ZCR",
                f"{zcr:.4f}"
            )