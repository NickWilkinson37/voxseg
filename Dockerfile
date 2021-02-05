FROM python:3.8

WORKDIR /voxseg

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./voxseg/ ./voxseg/voxseg/

COPY ./setup.py ./LICENSE ./README.md ./voxseg/

RUN pip install ./voxseg

CMD ["python", "./voxseg/voxseg/main.py"]