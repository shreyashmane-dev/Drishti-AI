# 🚀 DRISHTI AI - Cloud GPU Training Guide (Google Colab & Kaggle)

Diabetic retinopathy deep learning backbones (**EfficientNet-B3**, **ConvNeXt-Base**) require substantial matrix compute. Because standard local laptops do not typically possess high-VRAM NVIDIA GPUs, **Google Colab** and **Kaggle** provide **free cloud GPUs (NVIDIA T4 / P100)** to train models 20x to 50x faster.

---

## 🌟 Method 1: Kaggle Notebook (Recommended & Fastest — 0 Setup)

Kaggle hosts the **APTOS 2019 Blindness Detection** dataset internally with **zero download time**. The newly redesigned **DrishtiVisionV2** training script runs standalone with **CBAM Attention** and **Dual-Task Anti-Collapse Loss**.

### Option A: Upload the Ready Notebook (.ipynb)
1. Go to [kaggle.com](https://www.kaggle.com/) and log in.
2. Click **"+ Create" > "New Notebook"**.
3. In top menu: **File > Import Notebook > Browse** and select [`notebooks/DrishtiAI_Kaggle_Training.ipynb`](file:///d:/Projects/DrishtiAI/notebooks/DrishtiAI_Kaggle_Training.ipynb).
4. In the right-hand **Notebook Settings** panel:
   - **Accelerator**: Select **GPU P100** or **GPU T4 x2** (Free 30 hours/week).
   - **Internet**: Toggle to **On**.
5. Click **"+ Add Input"** (top right), search for `aptos2019-blindness-detection` and click **Add**.
6. Click **"Run All"**.
   - Training completes in ~12–15 minutes with Automatic Mixed Precision (AMP).
   - The notebook automatically validates all 5 clinical stages to prove zero mode collapse.
7. Click the download link in Cell 5 or download `drishti_best_model.pth` from the **Output** panel.

### Option B: Copy & Paste Single Script
1. Open a new Kaggle notebook with **GPU P100 / T4 x2** and add `aptos2019-blindness-detection`.
2. In Cell 1, run:
   ```bash
   !pip install --quiet albumentations opencv-python-headless timm
   ```
3. Copy the entire contents of [`notebooks/kaggle_standalone_train.py`](file:///d:/Projects/DrishtiAI/notebooks/kaggle_standalone_train.py) into Cell 2 and run it.
4. Download `drishti_best_model.pth` from `/kaggle/working/artifacts/models/`.

---

## ⚡ Method 2: Google Colab (Free T4 GPU)

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File > Upload notebook** and select [`DrishtiAI_Colab_Training.ipynb`](file:///d:/Projects/DrishtiAI/notebooks/DrishtiAI_Colab_Training.ipynb) from your `DrishtiAI/notebooks/` folder.
3. Switch runtime to GPU:
   - Menu: **Runtime > Change runtime type > Hardware accelerator > T4 GPU**.
4. Run each cell sequentially:
   - **Cell 1**: Confirms GPU (`nvidia-smi`).
   - **Cell 2 & 3**: Installs dependencies.
   - **Cell 4**: Downloads the dataset using your Kaggle API key or Kaggle credentials.
   - **Cell 5**: Executes training with Automatic Mixed Precision (AMP) enabled.
   - **Cell 6**: Automatically triggers a browser download of `drishti_best_model.pth`.

---

## 💻 Method 3: Run Locally (CPU / Local Machine)

If you wish to test training locally on your current machine:

```powershell
# Run a test training for 1 epoch on fold 1 with a small batch size
python ai/train.py --backbone efficientnet_b3 --epochs 1 --batch_size 4 --target_fold 1
```

> **Note**: Because your local machine is running on CPU, local training will be slower. For full training (15 epochs per fold), use **Kaggle** or **Google Colab** above.

---

## 🔄 Using Your Trained Cloud Model in Drishti AI

Once downloaded from Colab or Kaggle:
1. Place the downloaded file into your local project directory at:
   `DrishtiAI/artifacts/models/drishti_best_model.pth`
2. Start the local server:
   ```powershell
   python app.py
   ```
3. Open `http://127.0.0.1:8000/frontend/index.html` in your browser. The Drishti AI pipeline will automatically load your newly trained cloud model weights for inference, Grad-CAM heatmaps, and clinical reporting!
