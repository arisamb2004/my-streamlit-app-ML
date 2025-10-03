import streamlit as st
import joblib
import pandas as pd
from catboost import CatBoostRegressor
import numpy as np

# ====================================================================
# --- 0. Configuration & Language ---
# ====================================================================

if 'lang' not in st.session_state:
    st.session_state['lang'] = 'th'

LANGUAGE_DICT = {
    "th": {
        "title": "ระบบคาดการณ์มูลค่าสัตว์น้ำ (Fisheries Price Predictor)",
        "input_header": "ข้อมูลการป้อนเข้า (Inputs)",
        "predict_button": "คาดการณ์มูลค่า",
        "output_header": "ผลการคาดการณ์",
        "output_label_thb": "มูลค่าคาดการณ์ (หน่วย: พันบาท)",
        "total_baht": "มูลค่ารวมทั้งหมดโดยประมาณ",
        "avg_actual": "มูลค่ารวมเฉลี่ยจริงใน Dataset",
        "error_model": "ไม่พบไฟล์ Model, Scaler หรือ Dataset ที่จำเป็น",
        "error_prediction": "เกิดข้อผิดพลาดในการทำนาย",
        "input_fields": {
            "ปริมาณ(ตัน)": "1. ปริมาณ (ตัน)",
            "เดือน": "2. เดือน",
            "ประเภทการทำการประมง": "3. ประเภทการทำการประมง",
            "พื้นที่ทำการประมง": "4. พื้นที่ทำการประมง",
            "ขนาดเรือ": "5. ขนาดเรือ",
            "เครื่องมือ": "6. เครื่องมือ",
            "ชนิดสัตว์น้ำ": "7. ชนิดสัตว์น้ำ"
        }
    }
}

def _T(key):
    return LANGUAGE_DICT[st.session_state['lang']][key]

st.set_page_config(page_title="Fisheries Price Predictor", layout="wide")
st.title(_T("title"))

# ====================================================================
# --- 1. Load Model, Scaler, Dataset ---
# ====================================================================
# *** แก้ไข: ใช้ Relative Path แทน Absolute Path ***
MODEL_PATH = "catboost_model.cbm"  # ต้องมีไฟล์นี้ใน GitHub
SCALER_PATH = "scaler.pkl"        # ต้องมีไฟล์นี้ใน GitHub
DATA_PATH = "handlingEncoder.csv" # ต้องมีไฟล์นี้ใน GitHub
# *************************************************

try:
    model = CatBoostRegressor()
    # model.load_model(MODEL_PATH)
    # เนื่องจาก CatBoostRegressor() ถูกสร้างแล้ว
    # การโหลด model ควรใช้คำสั่งนี้
    model.load_model(MODEL_PATH) 
    
    scaler = joblib.load(SCALER_PATH)
    df_actual = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"{_T('error_model')}: {e}")
    st.stop()

# ====================================================================
# --- 2. Feature Maps ---
# ====================================================================
MONTH_MAP = {str(i): i for i in range(1, 13)}
TYPE_MAP = {"พาณิชย์": 0, "พื้นบ้าน": 1}

# One-hot columns ตามที่ทำตอน preprocess
OHE_TOOL = [col for col in df_actual.columns if col.startswith("เครื่องมือ_")]
OHE_BOAT_SIZE = [col for col in df_actual.columns if col.startswith("ขนาดเรือ_")]
OHE_AREA = [col for col in df_actual.columns if col.startswith("พื้นที่ทำการประมง_")]
OHE_SPECIES = [col for col in df_actual.columns if col.startswith("ชนิดสัตว์น้ำ_")]

# ***** ส่วนที่แก้ไข: จำกัดตัวเลือกพื้นที่ทำการประมง *****
# สำหรับ selectbox ให้ใช้ label ที่ต้องการ
AREA_LIST = ["อันดามัน", "อ่าวไทย"]
# *******************************************************

BASE_FEATURES = ["ปริมาณ", "เดือน", "ประเภทการทำการประมง"]
FEATURE_COLUMNS = BASE_FEATURES + OHE_TOOL + OHE_BOAT_SIZE + OHE_AREA + OHE_SPECIES

# ====================================================================
# --- 3. Preprocess Input ---
# ====================================================================
def preprocess_input(raw_data):
    df = pd.DataFrame(0, index=[0], columns=FEATURE_COLUMNS)
    df["ปริมาณ"] = raw_data["ปริมาณ"]
    df["เดือน"] = MONTH_MAP[str(raw_data["เดือน"])]
    df["ประเภทการทำการประมง"] = TYPE_MAP[raw_data["ประเภทการทำการประมง"]]

    # เครื่องมือ
    col_tool = f"เครื่องมือ_{raw_data['เครื่องมือ']}"
    if col_tool in df.columns:
        df[col_tool] = 1

    # ขนาดเรือ
    col_size = f"ขนาดเรือ_{raw_data['ขนาดเรือ']}"
    if col_size in df.columns:
        df[col_size] = 1

    # พื้นที่ทำการประมง
    col_area = f"พื้นที่ทำการประมง_{raw_data['พื้นที่ทำการประมง']}"
    if col_area in df.columns:
        df[col_area] = 1

    # ชนิดสัตว์น้ำ
    col_species = f"ชนิดสัตว์น้ำ_{raw_data['ชนิดสัตว์น้ำ']}"
    if raw_data['ชนิดสัตว์น้ำ'] == "ปลากระโทงแทงร่ม":
        col_species = "ชนิดสัตว์น้ำ_ปลากะโทงแทร่ม"
    if col_species in df.columns:
        df[col_species] = 1

    # Scale input
    scaled_data = scaler.transform(df.values)
    return scaled_data

# ====================================================================
# --- 4. Streamlit Form ---
# ====================================================================
with st.form("input_form"):
    st.subheader(_T("input_header"))

    vol = st.number_input(_T("input_fields")["ปริมาณ(ตัน)"], min_value=0.0, value=1.0)
    month = st.selectbox(_T("input_fields")["เดือน"], options=list(MONTH_MAP.keys()), index=0)
    fish_type = st.selectbox(_T("input_fields")["ประเภทการทำการประมง"], options=list(TYPE_MAP.keys()))
    # ใช้ AREA_LIST ที่ถูกแก้ไข
    area = st.selectbox(_T("input_fields")["พื้นที่ทำการประมง"], options=AREA_LIST)
    boat = st.selectbox(_T("input_fields")["ขนาดเรือ"], options=[b.split("_")[-1] for b in OHE_BOAT_SIZE])
    tool = st.selectbox(_T("input_fields")["เครื่องมือ"], options=[t.split("_")[-1] for t in OHE_TOOL])
    species = st.selectbox(_T("input_fields")["ชนิดสัตว์น้ำ"], options=[s.split("_")[-1] for s in OHE_SPECIES])

    submitted = st.form_submit_button(_T("predict_button"))

if submitted:
    try:
        raw_input = {
            "ปริมาณ": vol,
            "เดือน": int(month),
            "ประเภทการทำการประมง": fish_type,
            "พื้นที่ทำการประมง": area,
            "ขนาดเรือ": boat,
            "เครื่องมือ": tool,
            "ชนิดสัตว์น้ำ": species
        }

        processed_data = preprocess_input(raw_input)
        prediction = model.predict(processed_data)
        prediction_value = round(float(prediction[0]), 2)

        st.subheader(_T("output_header"))
        st.success(f"{_T('output_label_thb')}: {prediction_value} พันบาท")
        st.info(f"{_T('total_baht')}: {prediction_value*1000:,.2f} บาท")
        st.info(f"{_T('avg_actual')}: {df_actual['มูลค่า'].sum():,.2f} บาท")

    except Exception as e:
        st.error(f"{_T('error_prediction')}: {e}")
