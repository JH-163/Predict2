import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from streamlit_shap import st_shap

# 设置页面配置
st.set_page_config(page_title="Predictive Model App", layout="wide", initial_sidebar_state="expanded")

# 设置样式：字体、背景颜色、按钮颜色等
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
        color: #333333;
    }
    h1, h2, h3 {
        color: #3a7bd5;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stTextInput {
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True
)

# 页面标题和说明
st.title("App for Gynecologic abdominal postoperative urinary retention")
st.subheader("This app allows you to input clinical data and predicts outcomes using an machine learning model.")
st.write("---")  # 分隔线

# 布局优化：使用列布局来分配输入字段
col1, col2, col3 = st.columns(3)

with col1:
    Age = st.selectbox("Age (years)", ("≥50", "18-49"))
    Menopause = st.selectbox("Menopause", ("Yes", "No"))
    SM = st.selectbox("Surgical modality", ("celioscope", "laparotomy"))
with col3:
    SD = st.number_input("Surgical duration (hour)", min_value=0.0, value=5.0)
    IBL = st.number_input("Intraoperative blood loss (ml)", min_value=0.0, value=200.0)
#    Hospital_stay_duration = st.number_input("Hospital stay duration (day)", min_value=0.0, value=5.0)

with col2:
    PA = st.selectbox("Postoperative analgesia (Oral analgesics/Analgesic pumps)", ("Yes", "No"))
    TOA = st.selectbox("Types of anesthesia", ("neuraxial anesthesia", "general anesthesia"))
    PQ = st.selectbox("Pelvic organ prolapse quantification", ("0", "1", "2", "3"))

# 提交按钮和输出区域
if st.button("Predict"):
    
    # 载入模型
    clf = joblib.load("model.pkl")
    # 定义变量的最大值和最小值
    min_max_values = {
        "SD": (1, 50),
        "IBL": (1, 3000)
    #    "Hospital_stay_duration": (1, 50)
    }

    # 定义一个函数将数据标准化为min-max范围
    def minmax_scale(value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)

    SD = minmax_scale(SD, *min_max_values["SD"])
    IBL = minmax_scale(IBL, *min_max_values["IBL"])
    #Hospital_stay_duration = minmax_scale(Hospital_stay_duration, *min_max_values["Hospital_stay_duration"])
    
    # 将选择框转换为0/1，仅对布尔类型变量执行替换操作
    bool_columns = ["Age", "Menopause", "SM", "PA", 
                    "TOA"]
    
    X = pd.DataFrame([[Age, IBL, Menopause, PA, PQ, 
                       SD, SM, TOA]],
                     columns=clf.feature_names_in_)
    
    # 仅替换布尔型数据
    X[bool_columns] = X[bool_columns].replace({"Yes": 1, "No": 0})
    X[bool_columns] = X[bool_columns].replace({"≥50": 1, "18-49": 0})
    X[bool_columns] = X[bool_columns].replace({"celioscope": 1, "laparotomy": 0})
    X[bool_columns] = X[bool_columns].replace({"neuraxial anesthesia": 1, "general anesthesia": 0})
    X["PQ"] = X["PQ"].replace({"0": 0, "1": 1, "2": 2, "3": 3})
    
    # 模型预测
    prediction = clf.predict(X)[0]
    prediction_probability = clf.predict_proba(X)[0, 1]

    # 根据预测值输出不同的文本
    if prediction == 0:
        prediction_text = "Prediction: Urinary retention will not occur"
    else:
        prediction_text = "Prediction: Urinary retention will occur"

    # 显示预测结果
    st.success(f"{prediction_text}")
    st.info(f"Predicted probability: **{round(prediction_probability * 100, 2)}%**")

    # SHAP值解释模型
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    
    # 显示SHAP force plot
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0, :]), height=140, width=1500)
st.write("---")  # 分隔线
st.markdown("### Author Information")
st.markdown("**Name:** Jiahao Shi; Youbei Lin; Xiaojing Qin; Hongyu Li  \n**E-mail:** reda4673@sina.com")
