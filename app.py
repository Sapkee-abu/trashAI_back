import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. โหลดโมเดล (ใช้ตัวเดิมได้เลยครับ ไม่ต้องเทรนใหม่)
model = None
MODEL_FILENAME = "trash_classifier_b3_pro.keras" 

print(f"⏳ Loading {MODEL_FILENAME}...")
try:
    model = tf.keras.models.load_model(MODEL_FILENAME, compile=False)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 2. รายชื่อ Class ภาษาอังกฤษ (ต้องคงไว้เพื่อให้ AI ทำงานถูก)
raw_class_names = [
    "battery", "brown-glass", "cardboard", "carton-drink", "cigarette",
    "clothes", "e-waste", "food-waste", "glass", "green-glass",
    "metal-can", "paper", "plastic-bag", "plastic-bottle", "plastic-cup",
    "shoes", "spray-cans", "styrofoam", "trash-general", "white-glass"
]

# =========================================================
# 🔥 3. Logic ใหม่: แปลงทุกอย่างเป็น "กลุ่มถัง"
# =========================================================
def get_bin_group(label):
    
    # 🔴 กลุ่ม 1: ขยะอันตราย (Red Bin)
    # ถ้า AI ทายว่าเป็น: ถ่าน, ขยะอิเล็กทรอนิกส์, สเปรย์, บุหรี่
    if label in ["battery", "e-waste", "spray-cans", "cigarette"]:
        return {
            "group_name": "🔴 ถังแดง (ขยะอันตราย)", 
            "advice": "ขยะพิษ/อันตราย ห้ามทิ้งรวม! แยกใส่ถุงแดงแล้วส่งจุดรับทิ้ง"
        }

    # 🟡 กลุ่ม 2: ขยะรีไซเคิล (Yellow Bin)
    # ถ้า AI ทายว่าเป็น: ขวดน้ำ, แก้วน้ำ, กระดาษ, ลัง, กล่องนม, กระป๋อง, แก้ว
    elif label in ["plastic-bottle", "plastic-cup", 
                   "paper", "cardboard", "carton-drink",
                   "metal-can", "glass", "brown-glass", "green-glass", "white-glass"]:
        return {
            "group_name": "🟡 ถังเหลือง (รีไซเคิล)", 
            "advice": "ขยะรีไซเคิล: เทน้ำออก ล้างให้สะอาด ทำให้แห้ง/แบน ก่อนทิ้ง"
        }

    # 🟢 กลุ่ม 3: ขยะเปียก (Green Bin)
    # ถ้า AI ทายว่าเป็น: เศษอาหาร
    elif label in ["food-waste"]:
        return {
            "group_name": "🟢 ถังเขียว (ขยะเปียก)", 
            "advice": "ขยะอินทรีย์: กรองน้ำออก ทิ้งเฉพาะเศษอาหาร (นำไปหมักปุ๋ยได้)"
        }

    # 🔵 กลุ่ม 4: ขยะทั่วไป (Blue Bin)
    # ถ้า AI ทายว่าเป็น: ถุงพลาสติก, โฟม, ขยะทั่วไป, เสื้อผ้า, รองเท้า
    else:
        return {
            "group_name": "🔵 ถังน้ำเงิน (ขยะทั่วไป)", 
            "advice": "ขยะทั่วไป/เปื้อน: ย่อยสลายยากและไม่คุ้มรีไซเคิล ให้ทิ้งถังนี้"
        }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None: return {"error": "Model not loaded"}
    
    try:
        # เตรียมรูปภาพ (Resize 300x300)
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        w, h = image.size
        dim = min(w, h)
        image = image.crop(((w-dim)/2, (h-dim)/2, (w+dim)/2, (h+dim)/2))
        img_arr = np.array(image.resize((300, 300)))
        img_arr = np.expand_dims(img_arr, axis=0)

        # ให้ AI ทายผล
        pred = model.predict(img_arr, verbose=0)[0]
        idx = int(np.argmax(pred))
        conf = float(pred[idx]) * 100
        raw_label = raw_class_names[idx]

        # ตั้งเกณฑ์ความมั่นใจ (50%)
        if conf < 50.0:
            return {
                "prediction": "❓ ไม่แน่ใจ",
                "prediction_th": "ลองถ่ายใหม่",
                "bin": "ลองถ่ายใหม่",
                "confidence": round(conf, 2),
                "advice": "ภาพไม่ชัด หรือ AI ไม่รู้จัก ลองขยับกล้องเข้าใกล้ๆ ครับ"
            }

        # 🔥 เรียกใช้ Logic ใหม่ (แปลงเป็นชื่อถัง)
        info = get_bin_group(raw_label)

        return {
            # ส่งค่า "ชื่อถัง" ไปให้หน้าจอแสดงผล (แทนชื่อขยะเดิม)
            "prediction": info["group_name"],    # ตัวหนังสือใหญ่
            "prediction_th": info["group_name"], # ตัวหนังสือเล็ก (เผื่อแอปใช้อันนี้)
            "bin": info["group_name"],           # ชื่อถัง
            
            "confidence": round(conf, 2),
            "advice": info["advice"]
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def health(): return {"status": "Running", "mode": "Bin-Group Logic"}