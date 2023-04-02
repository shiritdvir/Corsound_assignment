FROM python:3.9-slim
# FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY src /app/src
COPY main.py /app/main.py
COPY config.py /app/config.py

CMD ["python", "main.py"]
