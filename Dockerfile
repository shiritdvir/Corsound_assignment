FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
WORKDIR /app
RUN pip install --trusted-host pypi.python.org numpy scikit-learn pyeer seaborn torchaudio matplotlib \
    sklearn huggingface-datasets transformers
COPY . /app
CMD ["python", "main.py"]