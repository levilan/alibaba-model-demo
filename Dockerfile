FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ⚠️ 新增頂層 .py 模組時這裡要跟著加，否則 image 裡沒有那個檔、容器一啟動就
# ModuleNotFoundError（2026-09-09 admin.py 就是這樣讓部署失敗的）。
# tests/test_deploy_files.py 會擋下同類疏漏。
COPY app.py .
COPY admin.py .
# admin.py 用 importlib 載入 scripts/usage_stats.py 讀統計，所以 scripts/ 也要進 image
COPY scripts/ scripts/
COPY templates/ templates/
COPY static/ static/

RUN mkdir -p outputs/images outputs/videos outputs/audio static/uploads

EXPOSE 5050

CMD ["python", "app.py"]
