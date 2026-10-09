
import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import mobilenet_v2

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="PlantCare | Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "plant_disease_model.keras"
CLASS_NAMES_PATH = os.path.join("Results", "class_names.json")
IMAGE_SIZE = (160, 160)
TEST_ACCURACY = 44.92

# ============================================================
# CUSTOM DESIGN
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f6faf5;
    }
    [data-testid="stSidebar"] {
        background-color: #e8f2e6;
    }
    .hero {
        padding: 35px 30px;
        border-radius: 18px;
        background: linear-gradient(120deg, #174d35, #39865b);
        color: white;
        margin-bottom: 22px;
    }
    .hero h1 {
        color: white;
        font-size: 40px;
    }
    .hero p {
        color: #eff9ef;
        font-size: 17px;
    }
    .section-card {
        background: white;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #e1ebdf;
        margin-bottom: 12px;
    }
    .section-card h3 {
        color: #20583a;
    }
    .small-note {
        color: #5c6b60;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    return model, class_names


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("# 🌿 PlantCare")
    st.caption("Plant Disease Detection System")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "About Project",
            "Dashboard",
            "Results",
            "Conclusion",
        ],
        index=0,
    )

    st.divider()
    st.caption("Academic project")
    st.caption("Dataset: PlantDoc")
    st.caption("Model: MobileNetV2")


# ============================================================
# HOME PAGE
# ============================================================
if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <h1>🌿 Plant Disease Detection</h1>
            <p>
                Exploring plant leaf classification using image processing
                and deep learning.
            </p>
            <p>
                Upload a leaf image, explore the model's prediction,
                and learn about the project.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Welcome to PlantCare")

    st.write(
        "Plant diseases can affect plant health and agricultural productivity. "
        "This academic project explores how a trained image-classification "
        "model can classify plant leaf images into learned categories."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="section-card">
                <h3>📷 Image Analysis</h3>
                Upload a plant leaf image for model analysis.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="section-card">
                <h3>🧪 27 Classes</h3>
                The trained model predicts among 27 dataset classes.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="section-card">
                <h3>📊 Model Evaluation</h3>
                Explore the test results and current limitations.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "Start with the Dashboard to upload an image, or visit About Project "
        "to learn how the system was developed."
    )

    if st.button("🔍 Open Dashboard", type="primary"):
        st.session_state["selected_page"] = "Dashboard"
        st.rerun()


# ============================================================
# ABOUT PROJECT PAGE
# ============================================================
elif page == "About Project":
    st.title("📘 About the Project")
    st.write(
        "Plant Disease Detection Using AI/CNN is an academic project "
        "that explores image classification of plant leaves using "
        "the PlantDoc dataset and a MobileNetV2-based model."
    )

    st.subheader("Problem Statement")
    st.write(
        "Identifying plant leaf conditions visually can be challenging. "
        "This project investigates whether a trained image-classification "
        "model can assign an uploaded leaf image to one of its learned classes."
    )

    st.subheader("Objectives")
    st.markdown(
        """
        - Prepare and inspect the plant leaf image dataset.
        - Resize and preprocess images for model input.
        - Train a MobileNetV2-based classification model.
        - Evaluate the model on a separate test dataset.
        - Build a web interface for uploading images and viewing predictions.
        """
    )

    st.subheader("Dataset")
    a, b, c = st.columns(3)
    a.metric("PlantDoc Classes", "27")
    b.metric("Training Images", "2,316")
    c.metric("Test Images", "236")

    st.write(
        "The dataset contains images from different plant and leaf-condition "
        "categories. Its Background class is also included among the "
        "model's classification categories."
    )

    st.subheader("Technologies Used")
    st.markdown(
        """
        - **Python** — programming language
        - **TensorFlow / Keras** — model training and inference
        - **MobileNetV2** — transfer-learning image feature extractor
        - **NumPy and Pillow** — image preparation
        - **Streamlit** — interactive website
        """
    )

    st.subheader("Methodology")
    st.markdown(
        """
        **Dataset → Image preprocessing → MobileNetV2 model →
        Model evaluation → Image upload → Predicted class**
        """
    )

    st.warning(
        "The model predicts dataset categories; it does not reliably "
        "confirm plant disease. Results can be incorrect."
    )


# ============================================================
# DASHBOARD PAGE
# ============================================================
elif page == "Dashboard":
    st.title("🔬 Plant Image Analysis Dashboard")
    st.write("Upload an image to view the model's classification result.")

    if not os.path.isfile(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    if not os.path.isfile(CLASS_NAMES_PATH):
        st.error(f"Class names file not found: {CLASS_NAMES_PATH}")
        st.stop()

    try:
        model, class_names = load_model_and_classes()
    except Exception as error:
        st.error(f"Could not load the model: {error}")
        st.stop()

    st.success(f"Model loaded successfully — {len(class_names)} classes")

    uploaded_file = st.file_uploader(
        "Choose a plant leaf image",
        type=["jpg", "jpeg", "png"],
        key="leaf_uploader",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("This image could not be opened. Please try another image.")
            st.stop()

        left, right = st.columns([1, 1])

        with left:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        with right:
            st.subheader("Prediction")

            if st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    resized_image = image.resize(IMAGE_SIZE)
                    image_array = np.array(
                        resized_image, dtype=np.float32
                    )
                    image_array = np.expand_dims(image_array, axis=0)
                    image_array = mobilenet_v2.preprocess_input(image_array)

                    probabilities = model.predict(
                        image_array, verbose=0
                    )[0]

                    predicted_index = int(np.argmax(probabilities))
                    predicted_class = class_names[predicted_index]
                    confidence = float(probabilities[predicted_index]) * 100

                st.session_state["last_prediction"] = {
                    "class": predicted_class,
                    "confidence": confidence,
                    "filename": uploaded_file.name,
                }

                st.metric("Predicted Class", predicted_class)
                st.metric("Model Confidence", f"{confidence:.2f}%")

                st.caption(
                    "Confidence is the model's highest predicted probability, "
                    "not a guarantee that the prediction is correct."
                )

                st.subheader("Top 3 Predictions")
                top_indices = np.argsort(probabilities)[-3:][::-1]

                for index in top_indices:
                    score = float(probabilities[index])
                    st.write(
                        f"**{class_names[int(index)]}** — "
                        f"{score * 100:.2f}%"
                    )
                    st.progress(max(0.0, min(1.0, score)))

    if "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]
        st.divider()
        st.subheader("Most Recent Prediction")
        st.write(f"Image: `{result['filename']}`")
        st.write(f"Predicted class: **{result['class']}**")
        st.write(f"Confidence: **{result['confidence']:.2f}%**")

    st.warning(
        "Evaluation limitation: the model achieved 44.92% accuracy on "
        "the separate 236-image PlantDoc test set. Predictions may be wrong "
        "and should not be used as a confirmed agricultural diagnosis."
    )


# ============================================================
# RESULTS PAGE
# ============================================================
elif page == "Results":
    st.title("📊 Results and Model Performance")
    st.write(
        "This page summarizes the model's measured performance on the "
        "separate PlantDoc test dataset."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy", "44.92%")
    col2.metric("Test Images", "236")
    col3.metric("Classification Classes", "27")

    st.subheader("Training Graphs")

    graph1 = os.path.join("Results", "07_training_accuracy.png")
    graph2 = os.path.join("Results", "07_training_loss.png")

    left, right = st.columns(2)

    with left:
        if os.path.isfile(graph1):
            st.image(graph1, caption="Training and Validation Accuracy",
                     use_container_width=True)
        else:
            st.info("Training accuracy graph is not available yet.")

    with right:
        if os.path.isfile(graph2):
            st.image(graph2, caption="Training and Validation Loss",
                     use_container_width=True)
        else:
            st.info("Training loss graph is not available yet.")

    st.subheader("Confusion Matrix")

    possible_matrices = [
        os.path.join("Results", "08_confusion_matrix.png"),
        os.path.join("Results", "confusion_matrix.png"),
        os.path.join("Results", "08_confusion_matrix_normalized.png"),
    ]

    matrix_path = next(
        (path for path in possible_matrices if os.path.isfile(path)),
        None,
    )

    if matrix_path:
        st.image(matrix_path, use_container_width=True)
    else:
        st.info(
            "A confusion matrix image was not found under the expected "
            "filenames in the Results folder."
        )

    st.subheader("Interpretation")
    st.write(
        "The test accuracy shows that the model still makes substantial "
        "classification errors. Performance varies between classes, so "
        "the predicted label should be treated as an experimental result."
    )

    st.warning(
        "The 44.92% figure is test accuracy, not a guarantee for an individual "
        "image. Training or validation accuracy must not be presented as "
        "test accuracy."
    )


# ============================================================
# CONCLUSION PAGE
# ============================================================
elif page == "Conclusion":
    st.title("🎓 Conclusion and Future Scope")

    st.subheader("Conclusion")
    st.write(
        "This project developed a plant leaf image-classification workflow "
        "using the PlantDoc dataset, image preprocessing, a MobileNetV2-based "
        "model, and a Streamlit interface for interactive predictions."
    )

    st.write(
        "The model achieved 44.92% accuracy on the separate 236-image test "
        "dataset. This demonstrates a working experimental pipeline, but "
        "also shows that classification performance needs improvement."
    )

    st.subheader("Current Limitations")
    st.markdown(
        """
        - Predictions can be incorrect, including for familiar leaf images.
        - Performance is not equally strong across all 27 classes.
        - The model has not been established as a reliable diagnostic tool.
        - Results may change with image quality, lighting, and backgrounds.
        """
    )

    st.subheader("Future Scope")
    st.markdown(
        """
        - Improve and expand the training dataset.
        - Investigate class imbalance and data quality.
        - Experiment with suitable training and fine-tuning settings.
        - Evaluate improvements on an untouched test set.
        - Improve image handling and the user interface.
        - Test the system on diverse real-world leaf images.
        """
    )

    st.success(
        "Project outcome: an educational plant-image classification "
        "prototype with an interactive web interface."
    )

    st.caption(
        "This system is for educational demonstration and is not a substitute "
        "for expert agricultural advice."
    )


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    "<p style='text-align:center;color:#66756a;'>"
    "🌿 Plant Disease Detection Using AI/CNN | PlantDoc Dataset | "
    "Academic Project</p>",
    unsafe_allow_html=True,
)

# PURPOSE OF THIS PROGRAM
# Creates a multi-section Streamlit website with Home, About Project,
# Dashboard, Results, and Conclusion pages.
# Loads the trained MobileNetV2 model and the saved class-name mapping.
# Predicts a class for an uploaded plant image using the same preprocessing
# method used during training and displays the top three probabilities.
# Shows available training graphs and confusion matrix from the Results folder.
# Reports the measured 44.92% test accuracy on 236 separate PlantDoc images.
# Limitation: predictions may be wrong; this is an educational prototype,
# not a reliable plant disease diagnosis system.
