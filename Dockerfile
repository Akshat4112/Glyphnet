
FROM ubuntu:20.04

RUN apt-get update -y && \
    apt-get install -y python3-pip python-dev

COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt 
RUN [ "python3","import nltk; nltk.download('all')" ]
EXPOSE 5001 
ENTRYPOINT [ "python3" ] 
CMD [ "src/scripts/trai_cnn.py" ] 