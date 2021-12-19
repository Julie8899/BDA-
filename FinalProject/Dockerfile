FROM python:3

WORKDIR /final_project

RUN apt-get update && apt-get install -y default-mysql-client

COPY ./requirements.txt .

RUN pip install --user -r requirements.txt