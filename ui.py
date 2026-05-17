import streamlit as st
import requests
import os

API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG System", layout="wide")

st.title("🤖 RAG System - Chat với tài liệu")

# ========== SIDEBAR - UPLOAD FILE ==========
with st.sidebar:
    st.header("📤 Upload tài liệu")
    
    # Chọn file
    uploaded_file = st.file_uploader("Chọn file PDF", type=["pdf"])
    
    # Nút upload - QUAN TRỌNG!
    if st.button("🚀 Upload và xử lý", type="primary", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("⏳ Đang xử lý file, vui lòng chờ..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_BASE_URL}/upload", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ Upload thành công: {uploaded_file.name}")
                    st.info(f"📊 Đã tạo: {data['num_chunks']} chunks | {data['num_embeddings']} embeddings")
                    st.balloons()  # Hiệu ứng chúc mừng
                else:
                    st.error(f"❌ Lỗi: {response.text}")
        else:
            st.warning("⚠️ Vui lòng chọn file trước khi upload")
    
    st.divider()
    st.caption("💡 Hướng dẫn: Chọn file PDF -> Bấm Upload -> Hỏi câu hỏi bên dưới")

# ========== CHAT AREA ==========
st.subheader("💬 Hỏi về tài liệu")

# Khởi tạo chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Nhập câu hỏi..."):
    # Hiển thị user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Gọi API generate
    with st.chat_message("assistant"):
        with st.spinner("🤔 Đang suy nghĩ..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/generate-variants",
                    params={"question": prompt, "num_variants": 3}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "Không có câu trả lời")
                    st.write(answer)
                    
                    # Xem chi tiết variants (tùy chọn)
                    with st.expander("🔍 Xem các biến thể câu hỏi"):
                        for v in data.get("variants", []):
                            st.write(f"- {v}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"❌ Lỗi API: {response.status_code}")
                    
            except Exception as e:
                st.error(f"❌ Không kết nối được backend: {e}")
