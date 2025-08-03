import streamlit as st
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as T
import albumentations as A
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import SamProcessor, SamModel
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import cv2
import json
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# Classification
@st.cache_resource
def load_model_c(checkpoint_path, num_classes):
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def get_transform_t(image_size):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

def load_config(json_path):
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config

# Segmentation
def load_model_s(checkpoint_path):
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model

def get_transform_a(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2(),
    ])

# Streamlit
def file_uploader(content, key):
    return st.file_uploader(f"Upload your {content} here", type=["png", "jpg", "jpeg"], key=key)


def postprocess_mask(mask_tensor, original_size):
    mask = torch.sigmoid(mask_tensor).cpu().detach().numpy()
    mask = (mask > 0.5).astype(np.uint8)[0, 0] * 255
    mask_resized = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    return mask_resized

def overlay_mask_on_image(image_np, mask_np, alpha=0.5):
    if len(mask_np.shape) == 2:
        mask_np = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
    mask_colored = np.zeros_like(mask_np)
    mask_colored[:, :, 1] = mask_np[:, :, 1]  # Green channel
    overlay = cv2.addWeighted(image_np, 1, mask_colored, alpha, 0)
    return overlay

# ======================================================== User Interface ========================================================

st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>COVID-19 Classification And Segmentation</h1>",
    unsafe_allow_html=True
)

config = load_config("config.json")
class_names = config["categories"]

# ==================== CLASSIFICATION ====================
classification_model = load_model_c(config["checkpoint_path"][0], config["num_classes"])
transform = get_transform_t(config["image_size"][0])

st.title("Classification: ")
classification_image = file_uploader("X-Ray image", "classification_image")
if classification_image is not None:
    image = Image.open(classification_image).convert("RGB")

    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = classification_model(input_tensor)
        _, pred = torch.max(output, 1)
        predicted_class = class_names[pred.item()]



    col1, col2= st.columns(2)
    with col1:
        st.image(image, caption="Image you just uploaded")
        col_1, col_2, col_3 = st.columns([1,5,1])
        with col_2:
            click_c = st.button("Predict")
    with col2:
        if click_c:
            with col2:
                st.header("Your result:")
                st.header(predicted_class)



st.divider()

# ==================== SEGMENTATION ====================
st.title("Segmentation:")

col1, col2 = st.columns(2)

segmentation_model = load_model_s(config["checkpoint_path"][1])

with col1:
    segmentation_image = st.file_uploader("Upload X-Ray Image", type=["jpg", "jpeg", "png"])
    if segmentation_image is not None:
        image_pil = Image.open(segmentation_image).convert("RGB")
        st.image(image_pil, caption="X-Ray Image", use_container_width=True)

with col2:
    segmentation_mask = st.file_uploader("Upload Ground Truth Mask", type=["jpg", "jpeg", "png"])
    if segmentation_mask is not None:
        mask_pil = Image.open(segmentation_mask).convert("L")
        st.image(mask_pil, caption="Ground Truth Mask", use_container_width=True)

if segmentation_image is not None and segmentation_mask is not None:
    col1, col2, col3 = st.columns([1, 0.5, 1])
    with col2:
        click_segment = st.button("Run Segmentation")

    if click_segment:
        # Load image and mask
        image_np = np.array(image_pil)
        mask_np = np.array(mask_pil)
        mask_bin = (mask_np > 127).astype(np.uint8)

        # Get bounding box from GT mask
        bbox = get_bounding_box(mask_bin)

        # Prepare SAM processor
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        inputs = processor(images=image_np, input_boxes=[[bbox]], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = segmentation_model(
                pixel_values=inputs["pixel_values"],
                input_boxes=inputs["input_boxes"],
                multimask_output=False
            )
            pred_mask_tensor = outputs.pred_masks.squeeze()  # [H, W] tensor
            pred_mask_np = (pred_mask_tensor.cpu().numpy() > 0.5).astype(np.uint8) * 255

        pred_mask_resized = cv2.resize(pred_mask_np, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

        if mask_bin.shape != pred_mask_resized.shape:
            mask_bin_resized = cv2.resize(mask_bin, (pred_mask_resized.shape[1], pred_mask_resized.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
        else:
            mask_bin_resized = mask_bin

        intersection = np.logical_and(mask_bin_resized, pred_mask_resized // 255).sum()
        union = np.logical_or(mask_bin_resized, pred_mask_resized // 255).sum()
        iou_score = intersection / (union + 1e-8)

        overlay_gt = draw_bounding_box(image_np, mask_bin * 255)
        overlay_pred = draw_bounding_box(image_np, pred_mask_resized)

        overlay_combined = overlay_two_masks_on_image(image_np, mask_bin_resized * 255, pred_mask_resized)

        # Display results
        st.subheader(f"IoU: `{iou_score:.4f}`")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image_np, caption="Original Image", use_container_width=True)
        with col2:
            st.image(overlay_combined, caption="Overlay: GT Mask (Green) + Pred Mask (Red)", use_container_width=True)
        with col3:
            st.image(overlay_pred, caption="Predicted Mask with Bounding Box", use_container_width=True)










