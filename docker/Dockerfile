FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \  
    curl \ 
    ffmpeg \
    libsm6 \
    libxext6 \
    software-properties-common \
    wget \ 
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY src/ /app/src/
COPY app.py /app/app.py
COPY img/ /app/img/
    
RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD ["curl", "--fail", "http://localhost:8501/_stcore/health"]

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]