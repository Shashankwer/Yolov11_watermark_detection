FROM python:3.12-slim

RUN apt update 
RUN apt install -y --no-install-recommends \
    build-essential gcc

RUN groupadd -g 999 python && \
    useradd -r -u 999 -g python python

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /usr/app

COPY app .
EXPOSE 3000

CMD ["fastapi","run","app.py","--port","3000"]
