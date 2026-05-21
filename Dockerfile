FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf \
    TORCH_HOME=/tmp/torch \
    MPLCONFIGDIR=/tmp/mpl

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download Noto Sans Armenian font for PDF export
RUN mkdir -p /app/fonts && \
    apt-get update && apt-get install -y --no-install-recommends wget && \
    wget -q -O /app/fonts/NotoSansArmenian-Regular.ttf \
      "https://github.com/google/fonts/raw/main/ofl/notosansarmenian/NotoSansArmenian%5Bwdth%2Cwght%5D.ttf" && \
    apt-get purge -y wget && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY . .

# HF Spaces expects port 7860
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
