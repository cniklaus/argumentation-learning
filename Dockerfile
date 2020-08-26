FROM python:3.7-slim-buster

RUN apt-get update && \
   apt-get -y install gcc mono-mcs && \
   rm -rf /var/lib/apt/lists/*

# EXPOSE 80
COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
RUN python -m spacy download de

CMD ["python","./server.py"]
