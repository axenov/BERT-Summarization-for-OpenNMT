FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

RUN apt-get -y update && \
    apt-get upgrade -y && \
    apt-get install -y python3-dev emacs &&\
    apt-get install -y python3-dev git &&\
    apt-get update -y


ADD requirements.txt .
RUN pip install -r requirements.txt
RUN git clone https://github.com/nvidia/apex && cd apex

ADD flaskController.py .
ADD summarizer.py .
ADD extractor.py .
ADD models ./models
ADD nif ./nif
ADD onmt ./onmt
ADD config ./config

RUN export LC_ALL=C.UTF-8
RUN export LANG=C.UTF-8

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN python3 -m nltk.downloader punkt -d /usr/share/nltk_data
RUN python3 -m nltk.downloader stopwords -d /usr/share/nltk_data


EXPOSE 5000

ENTRYPOINT FLASK_APP=flaskController.py flask run -h 0.0.0.0 -p 5000
#CMD ["/bin/bash"]
