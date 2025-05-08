# 🧠 Traffy Fondue Duration Prediction Dashboard

This is a Streamlit-based interactive dashboard for analyzing and predicting the time required to resolve issues reported through the Traffy Fondue platform using a fine-tuned BERT model (WangchanBERTa).

---

## 📦 Features

- 📊 Visualize average fix duration by problem type and organization
- 🌍 Heatmap of resolution time across different districts
- 🔮 Predict expected resolution time using BERT-based model
- ⚡ Runs on your local machine with GPU/CPU support

---

## 📁 Project Structure

```bash

prediction/
├── best_model_state.bin # Trained model weights
├── streamlit.py # Main dashboard application
├── model.py # Model class and loading utility
├── predictor.py # Prediction logic and preprocessing
├── tokenizer/ # Tokenizer files (from HuggingFace)
├── requirements.txt # Python dependencies (optional)
└── README.md # This file

```
---

## 🛠️ Setup Instructions

### 1. Download trained medel from Kaggle

Go to [Link Text](https://www.kaggle.com/code/nacnano/traffy-fondue-duration/notebook).
In the output section, download file named "best_model_state.bin" and put it in project directory.

### 2. Clone the repository or copy files

```bash

git clone <your-repo-url>
cd dsde

```

### 3. Create a virtual environment

```bash

python -m venv venv
venv\\Scripts\\activate         # On Windows
# source venv/bin/activate     # On macOS/Linux

```

### 4. Install dependencies

```bash

pip install -r requirements.txt
# If you don't have a requirements.txt, install manually:
pip install streamlit torch transformers pandas plotly pydeck

```

### 5. Download Tokenizer (if not included)

```bash

# If the ./tokenizer folder is not present, you can download from HuggingFace:
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
tokenizer.save_pretrained("./tokenizer")

```

### 6. Run the app

```bash

streamlit run streamlit.py

```

## 🧪Example Input for Prediction

Problem Type: ขยะ

Description: มีขยะสะสมจำนวนมากหน้าบ้าน

Organization: สำนักงานเขตบางรัก

Date: วันที่แจ้งปัญหา

Area: ชื่อเขตหรือพื้นที่ เช่น บางรัก


## 🧑‍💻 Authors

Fine-tuned by: 

```bash

Punnawich Yiamsombat 6431331421
Chotpisit Adunsehawat 6531313221
Vijak Khajornritdacha 6532155621
Korakrit Vichitlekarn 6431301621

```

Model: WangchanBERTa

Dashboard by: Streamlit + Plotly + PyTorch
