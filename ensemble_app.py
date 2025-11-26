import streamlit as st
from PIL import Image
import numpy as np
import dataset
import train
import utils

# Sidebar for navigation
page = st.sidebar.radio("Navigation", ["Dataset", "Train Models", "Ensemble Prediction"])

if page == "Dataset":
    st.title("Dataset Management")
    if st.button("Download and Prepare Dataset"):
        with st.spinner("Downloading and preparing dataset..."):
            dataset_path = dataset.download_and_prepare_dataset()
        st.success(f"Dataset prepared at {dataset_path}")

elif page == "Train Models":
    st.title("Train Models")
    dataset_path = "dataset/TRAIN"
    model_choice = st.selectbox("Choose model to train:", ["ResNet-18", "VGG-16"])
    num_epochs = st.slider("Number of epochs:", 1, 10, 2)

    if st.button("Train Model"):
        with st.spinner(f"Training {model_choice}..."):
            if model_choice == "ResNet-18":
                train_losses, test_losses, train_accs, test_accs = train.train_resnet18(dataset_path, num_epochs)
            else:
                train_losses, test_losses, train_accs, test_accs = train.train_vgg16(dataset_path, num_epochs)
        st.success(f"{model_choice} trained successfully!")
        st.write(f"Final Test Accuracy: {test_accs[-1]:.2f}%")

elif page == "Ensemble Prediction":
    st.title("Ensemble Model Demo: ResNet-18 + VGG-16")

    # Class names (from the dataset)
    class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

    st.write("Upload an image to get predictions from both models and their ensemble.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Load models
        resnet_model = utils.load_resnet18(class_names)
        vgg_model = utils.load_vgg16(class_names)

        # Get predictions
        pred_resnet, prob_resnet = utils.predict(resnet_model, image)
        pred_vgg, prob_vgg = utils.predict(vgg_model, image)

        # Ensemble
        ensemble_method = st.selectbox("Choose ensemble method:", ["Voting", "Averaging", "Weighted Averaging"])

        if ensemble_method == "Voting":
            ensemble_pred = utils.ensemble_voting(pred_resnet, pred_vgg)
            avg_prob = None
        elif ensemble_method == "Averaging":
            ensemble_pred = utils.ensemble_averaging(prob_resnet, prob_vgg)
            avg_prob = (prob_resnet + prob_vgg) / 2
        else:
            weight_resnet = st.slider("Weight for ResNet-18", 0.0, 1.0, 0.5)
            ensemble_pred, avg_prob = utils.ensemble_weighted(prob_resnet, prob_vgg, weight_resnet)

        # Display results
        st.subheader("Predictions:")
        st.write(f"ResNet-18 Prediction: {class_names[pred_resnet]}")
        st.write(f"VGG-16 Prediction: {class_names[pred_vgg]}")
        st.write(f"Ensemble Prediction: {class_names[ensemble_pred]}")

        # Probabilities
        st.subheader("Probabilities:")
        st.write("ResNet-18:")
        for i, prob in enumerate(prob_resnet):
            st.write(f"{class_names[i]}: {prob:.4f}")

        st.write("VGG-16:")
        for i, prob in enumerate(prob_vgg):
            st.write(f"{class_names[i]}: {prob:.4f}")

        if avg_prob is not None:
            if ensemble_method == "Averaging":
                st.write("Ensemble (Averaged Probabilities):")
            else:
                st.write(f"Ensemble (Weighted Averaged Probabilities, ResNet weight: {weight_resnet:.2f}):")
            for i, prob in enumerate(avg_prob):
                st.write(f"{class_names[i]}: {prob:.4f}")