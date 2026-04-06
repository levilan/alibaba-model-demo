FROM python:3.12-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 複製並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式代碼
COPY app.py .
COPY templates/ templates/
COPY static/ static/

# 建立輸出目錄
RUN mkdir -p outputs/images outputs/videos static/uploads

EXPOSE 5050

CMD ["python", "app.py"]
