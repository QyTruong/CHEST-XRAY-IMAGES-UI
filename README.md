# HOW TO INSTALL TO USE USER INTERFACE ?
---

## Step 1: git clone https://github.com/QyTruong/CHEST-XRAY-IMAGES-UI.git
<img width="2239" height="1246" alt="image" src="https://github.com/user-attachments/assets/0ac584b7-c729-43c3-8b3e-7c9602c4eea3" />


## Step 2: 
- Click this link https://github.com/yuuMQ/CHEST-XRAY-IMAGES-CLASSIFICATION-WITH-ViT, then scroll down to MY CHECKPOINT to download ViT_b16_models.zip
- Click this link https://github.com/yuuMQ/CHEST_XRAY_SEGMENTATION_WITH_SAM, then scroll down to MY CHECKPOINT to download SAM_models.zip


## Step 3:
- Extract ViT_b16_models.zip and SAM_models.zip to your project
![Uploading image.png…]()

- In ViT_b16_models.zip and SAM_models.zip have last_model.pt, you can delete those files if you don't need those for training

## Step 4:
- Then run in your project:
```text
streamlit run app.py
```
- User interface after running app.py
<img width="2559" height="1377" alt="image" src="https://github.com/user-attachments/assets/9f3f6957-356e-4d48-82ea-c771b2004522" />

---
# HOW TO USE USER INTERFACE ?
- Using for classification:
<img width="2559" height="1367" alt="image" src="https://github.com/user-attachments/assets/64320935-ec40-4fc8-83bb-6f57066126bc" />

---
- Using for segmentation:
<img width="2559" height="1371" alt="image" src="https://github.com/user-attachments/assets/b8fa73be-b4f5-4475-9a5f-45e937ddf663" />
<img width="2559" height="1362" alt="image" src="https://github.com/user-attachments/assets/4da8870f-60ef-4cb4-b27b-714eb11a8ef7" />
![Uploading image.png…]()







