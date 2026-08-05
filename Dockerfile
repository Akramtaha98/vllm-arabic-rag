FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENV VLLM_API_URL=http://localhost:8000/v1/chat/completions
ENV VLLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
