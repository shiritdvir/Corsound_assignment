FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y wget
RUN mkdir /data && \
    wget -q --show-progress https://www.eecs.yorku.ca/~bil/Datasets/for-2sec.tar.gz -O /data/for-2sec.tar.gz && \
    tar -xzf /data/for-2sec.tar.gz -C /data || echo "Error while extracting the file" && \
    rm /data/for-2sec.tar.gz

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY src /app/src
COPY main.py /app/main.py
COPY config.py /app/config.py

CMD ["python", "main.py"]
