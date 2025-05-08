import torch
from transformers import CamembertTokenizer
import numpy as np
from datetime import date
import streamlit as st

# Device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# โหลด tokenizer จาก local
tokenizer = CamembertTokenizer.from_pretrained("./tokenizer")

# One-hot vector สำหรับ 11 หน่วยงาน (ต้องตรงกับตอน train)
ORG_LIST = [
    'สำนักงานเขตบางรัก', 'สำนักงานเขตปทุมวัน', 'สำนักงานเขตดุสิต',
    'สำนักงานเขตคลองเตย', 'สำนักงานเขตบางนา', 'สำนักงานเขตลาดพร้าว',
    'สำนักงานเขตห้วยขวาง', 'สำนักงานเขตบางซื่อ', 'สำนักงานเขตจตุจักร',
    'สำนักงานเขตพระนคร', 'Other'
]

def make_org_vector(org: str):
    """สร้าง one-hot vector สำหรับ organization"""
    vector = [0] * len(ORG_LIST)
    if org in ORG_LIST:
        vector[ORG_LIST.index(org)] = 1
    else:
        vector[-1] = 1  # default เป็น Other
    return vector

def predict_duration(model, text: str, org: str, comment: str, type_: str, timestamp: date):
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=256,
        padding="max_length",
        truncation=True
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    def make_tensor(value):
        t = torch.tensor([[value]], dtype=torch.float32)
        return t.to(DEVICE)

    # ✅ แปลง feature ทั้งหมดเป็น [1, 1] แบบปลอดภัย
    specificity = make_tensor(1.0 if type_ != '{}' else 0.0)
    comment_length = make_tensor(len(comment))
    is_priority = make_tensor(1.0 if type_ in ['จราจร', 'ความปลอดภัย'] else 0.0)
    creation_month = make_tensor(timestamp.month)
    day_of_week = make_tensor(timestamp.weekday())
    org_array = np.array(make_org_vector(org), dtype=np.float32).reshape(1, -1)
    org_vector = torch.tensor(org_array, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            specificity=specificity,
            comment_length=comment_length,
            is_priority=is_priority,
            creation_month=creation_month,
            day_of_week=day_of_week,
            org_features=org_vector
        )
        prediction_scaled = output.item()



    return prediction_scaled * 10  # ไม่มี scaler ให้เดาว่า 0–10 วัน




