FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app/voxseg

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY voxseg ./voxseg/
COPY setup.py LICENSE README.md ./

RUN pip install ./

RUN chmod +x /app/voxseg/voxseg/main.py

ENTRYPOINT ["python", "/app/voxseg/voxseg/main.py"]