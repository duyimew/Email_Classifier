import streamlit as st
import joblib
import pandas as pd
import os

# --- 1. TẢI MODEL & VECTORIZER ---
@st.cache_resource
def load_models():
    # Kiểm tra xem file có tồn tại không
    if not os.path.exists('spam_model.pkl') or not os.path.exists('vectorizer.pkl'):
        return None, None
    m = joblib.load('spam_model.pkl')
    v = joblib.load('vectorizer.pkl')
    return m, v

model, vectorizer = load_models()

# --- 2. GIAO DIỆN WEB ---
st.set_page_config(page_title="Spam Email Detector", page_icon="📧")

st.title("📧 Phân loại Thư rác (Spam Classifier)")
st.write("Mô hình sử dụng thuật toán **Naive Bayes** để phát hiện thư rác.")

if model is None:
    st.error("⚠️ Không tìm thấy file model! Hãy chạy file huấn luyện để tạo 'spam_model.pkl' và 'vectorizer.pkl' trước.")
    st.stop()

# Tạo 2 tab cho 2 chức năng yêu cầu
tab1, tab2 = st.tabs(["✍️ Kiểm tra Email", "📂 Tải file CSV"])

# === TAB 1: KIỂM TRA TỪNG EMAIL ===
with tab1:
    st.header("Nhập nội dung Email")
    
    col1, col2 = st.columns(2)
    with col1:
        subject = st.text_input("Tiêu đề (Subject)", placeholder="Ví dụ: You won a lottery!")
    with col2:
        # Placeholder cho giao diện đẹp
        st.write("") 
        
    message = st.text_area("Nội dung (Message)", height=150, placeholder="Nhập nội dung email vào đây...")
    
    if st.button("🔍 Phân loại ngay", type="primary"):
        if not message and not subject:
            st.warning("Vui lòng nhập ít nhất Tiêu đề hoặc Nội dung!")
        else:
            # Tiền xử lý: Gộp Subject và Message giống lúc train
            full_text = (str(subject) + " " + str(message)).strip()
            
            # Vector hóa và dự đoán
            vec_input = vectorizer.transform([full_text])
            prediction = model.predict(vec_input)[0]
            proba = model.predict_proba(vec_input)[0]
            
            # Hiển thị kết quả
            st.divider()
            if prediction == 1:
                st.error(f"🚨 ĐÂY LÀ THƯ RÁC (SPAM)")
                st.metric("Độ tin cậy (Confidence)", f"{proba[1]:.2%}")
            else:
                st.success(f"✅ ĐÂY LÀ THƯ THƯỜNG (HAM)")
                st.metric("Độ tin cậy (Confidence)", f"{proba[0]:.2%}")

# === TAB 2: ĐÁNH GIÁ TỪ FILE CSV ===
with tab2:
    st.header("Đánh giá hàng loạt từ file CSV")
    st.info("File CSV cần có các cột: 'Subject', 'Message' và 'Spam/Ham' (để đối chiếu kết quả).")
    
    uploaded_file = st.file_uploader("Chọn file CSV của bạn", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Xem trước dữ liệu:")
            st.dataframe(df.head())
            
            if st.button("🚀 Chạy dự đoán cho toàn bộ file"):
                # 1. Tiền xử lý
                df['Subject'] = df['Subject'].fillna('')
                df['Message'] = df['Message'].fillna('')
                df['Content'] = df['Subject'] + " " + df['Message']
                
                # 2. Dự đoán
                X_vec = vectorizer.transform(df['Content'])
                df['Dự đoán'] = model.predict(X_vec)
                df['Nhãn dự đoán'] = df['Dự đoán'].map({1: 'spam', 0: 'ham'})
                
                # 3. Tính toán độ chính xác (nếu file có cột nhãn thật)
                if 'Spam/Ham' in df.columns:
                    df['Nhãn thực tế'] = df['Spam/Ham'].map({'spam': 1, 'ham': 0})
                    correct = (df['Dự đoán'] == df['Nhãn thực tế']).sum()
                    accuracy = correct / len(df)
                    
                    col_metric1, col_metric2 = st.columns(2)
                    col_metric1.metric("Tổng số email", len(df))
                    col_metric2.metric("Độ chính xác (Accuracy)", f"{accuracy:.2%}")
                
                # 4. Hiển thị bảng kết quả (Tô màu các dòng Spam)
                st.write("Kết quả chi tiết:")
                
                def highlight_spam(row):
                    return ['background-color: #ffcccc' if row['Nhãn dự đoán'] == 'spam' else '' for _ in row]
                
                st.dataframe(df[['Subject', 'Message', 'Nhãn dự đoán']].style.apply(highlight_spam, axis=1))
                
        except Exception as e:
            st.error(f"Có lỗi xảy ra khi đọc file: {e}")