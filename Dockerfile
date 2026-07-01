FROM ubuntu:20.04

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev

COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt

# Training reads/writes the data directory; scripts run from inside code/.
WORKDIR /app/code
ENTRYPOINT [ "python3" ]
CMD [ "train.py" ]
