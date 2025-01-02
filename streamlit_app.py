import streamlit as st
from PIL import Image
import numpy as np
import torch
import cv2
import albumentations as albu
from albumentations.core.composition import Compose
import archs
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


cols1, cols2 = st.columns([1, 5])

with cols1:
    st.image('./logo.png')

with cols2:
    st.write("## Rajkiya Engineering College Kannauj")

st.title("Brain Tumor Segmentation")
st.sidebar.image("./logo.png", width=120)
st.sidebar.write("### Early Brain Tumor Detection System using Modified U-Net")
st.sidebar.write("\n**Guided By:**  \nAshwini Kumar Upadhyaya  \nAsst. Professor, Rec Kannauj");
# st.sidebar.write("\n\n\n**Project By:**  \nShivam Singh(54)  \nDeependu Mishra(28)  \nSudhir Tiwari(59)  \nAshish Yadav(23)  \nNitesh Kumar(39)")
st.sidebar.write("\n\n\n**Project By:**  \nAshish Yadav(23)  \nDeependu Mishra(28)  \nNitesh Kumar(39)  \nShivam Singh(54)  \nSudhir Tiwari(59)")

def parse_args():
    # Dummy function, no need for argparse in Streamlit
    return {'image_path': None, 'model_name': 'UNet'}

def segment_image(image, model):
    # Load model configuration
    config = {
        'input_h': 256,
        'input_w': 256,
        'input_channels': 3,
        'num_classes': 1  # Assuming binary segmentation
    }

    # Move model to CPU
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Convert PIL image to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Apply transformations
    transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])
    transformed_image = transform(image=image_cv)['image']
    transformed_image = torch.unsqueeze(torch.from_numpy(transformed_image.transpose(2, 0, 1)), dim=0).float()

    # Move input image to CPU
    transformed_image = transformed_image.to(device)

    # Predict segmentation mask
    with torch.no_grad():
        output = model(transformed_image)
        output = torch.sigmoid(output).cpu().numpy()

    # Save segmented image
    output_image = output[0].transpose(1, 2, 0)
    output_image = (output_image * 255).astype('uint8')

    return output_image

def compute_dice(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return 2 * intersection / (union + 1e-7)

def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def compute_precision(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    return true_positive / (true_positive + false_positive + 1e-7)

def compute_hausdorff(y_true, y_pred):
    return np.max(cv2.distanceTransform(np.uint8(y_true), cv2.DIST_L2, 3) * y_pred)

def compute_sensitivity(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    return true_positive / (true_positive + false_negative + 1e-7)

def compute_jaccard(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-7)

def evaluate_segmentation(gt_image, segmented_image):
    try:
        # Convert PIL images to NumPy arrays if needed
        if isinstance(gt_image, Image.Image):
            gt_image = np.array(gt_image)
        if isinstance(segmented_image, Image.Image):
            segmented_image = np.array(segmented_image)
        
        # Ensure both images have the same dimensions
        if gt_image.shape != segmented_image.shape:
            segmented_image_resized = cv2.resize(segmented_image, (gt_image.shape[1], gt_image.shape[0]))
        else:
            segmented_image_resized = segmented_image
        
        # Thresholding
        _, gt_binary = cv2.threshold(gt_image, 128, 1, cv2.THRESH_BINARY)
        _, segmented_binary = cv2.threshold(segmented_image_resized, 51, 1, cv2.THRESH_BINARY)
        
        # Compute evaluation metrics
        dice_score = compute_dice(gt_binary, segmented_binary)
        accuracy = compute_accuracy(gt_binary, segmented_binary)
        precision = compute_precision(gt_binary, segmented_binary)
        hausdorff = compute_hausdorff(gt_binary, segmented_binary)
        sensitivity = compute_sensitivity(gt_binary, segmented_binary)
        jaccard = compute_jaccard(gt_binary, segmented_binary)
        
        # Return selected evaluation metrics
        return {
            "Dice": dice_score,
            "Accuracy": accuracy,
            "Precision": precision,
            "Hausdorff": hausdorff,
            "Sensitivity": sensitivity,
            "Jaccard": jaccard
        }
    except Exception as e:
        st.write(e)
        return None


def main():

    args = parse_args()

    print("Available architectures:", archs.__dict__)
    # Load the pre-trained model
    model_name = args['model_name']

    print("Available model architectures:", list(archs.__dict__.keys()))


    if model_name not in archs.__dict__:
        raise ValueError(f"Model architecture '{model_name}' not found in the archs module.")
    
    num_classes = 1  # Assuming binary segmentation
    input_channels = 3  # Assuming RGB images

    # Load the pre-trained model with specified arguments
    model = archs.__dict__[model_name](num_classes=num_classes, input_channels=input_channels)
    model.load_state_dict(torch.load('models/%s/model.pth' % model_name, map_location=torch.device('cpu')))

    mri_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"], key=1)
    mask_file = st.file_uploader("Upload Mask Image", type=["png", "jpg", "jpeg"], key=2)

    if mri_file is not None and mask_file is not None:
        mriImage = Image.open(mri_file)
        mri_np = np.array(mriImage)

        maskImage = Image.open(mask_file)

        col1, col2, col3 = st.columns(3)

        # st.image(image, caption='Uploaded MRI Image', use_column_width=True)
        with col1:
            st.image(mriImage, caption='Uploaded MRI Image')

        with col2:
            st.image(maskImage, caption='Uploaded Mask Image')

        if st.button('Segment Image'):
            # segmented_image = segment_image(mri_np, "brain_UNet_woDS")
            # st.image(segmented_image, caption='Segmented Image', use_column_width=True)
            segmented_image = segment_image(mriImage, model)
            with col3:
                st.image(segmented_image, caption='Segmented Image')

            metrics = evaluate_segmentation(maskImage, segmented_image)
            if metrics:
                st.write("Dice Score:", metrics["Dice"])
                st.write("Accuracy:", metrics["Accuracy"])
                st.write("Precision:", metrics["Precision"])
                st.write("Jaccard:", metrics["Jaccard"])
                st.write("Sensitivity:", metrics["Sensitivity"])
                st.write("Hausdorff:", metrics["Hausdorff"])
                # Add other metrics similarly
            else:
                st.write("Error occurred during evaluation.")


    if mri_file is not None and mask_file is None:
        mriImage = Image.open(mri_file)
        mri_np = np.array(mriImage)

        col1, col2 = st.columns(2)
        # st.image(image, caption='Uploaded MRI Image', use_column_width=True)

        with col1:
            st.image(mriImage, caption='Uploaded MRI Image')

        if st.button('Segment Image'):
            # segmented_image = segment_image(mri_np, "brain_UNet_woDS")
            # st.image(segmented_image, caption='Segmented Image', use_column_width=True)
            segmented_image = segment_image(mriImage, model)
            with col2:
                st.image(segmented_image, caption='Segmented Image')

if __name__ == '__main__':
    main()
