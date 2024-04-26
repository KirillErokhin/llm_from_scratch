FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip

COPY . /llm_from_scratch

WORKDIR /llm_from_scratch

RUN pip install --no-cache-dir -r requirements.txt