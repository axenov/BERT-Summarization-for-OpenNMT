FROM pytorch/pytorch:latest

RUN apt-get -y update && \
    apt-get upgrade -y && \
    apt-get install -y python3-dev emacs &&\
    apt-get update -y


ADD requirements.txt .
RUN pip install -r requirements.txt && python setup.py install
RUN git clone https://github.com/nvidia/apex && cd apex && python setup.py install

ADD flaskController.py .
ADD summarizer.py .
ADD models ./models
ADD nif ./nif
ADD onmt ./onmt
ADD config ./config

RUN export LC_ALL=C.UTF-8
RUN export LANG=C.UTF-8

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

EXPOSE 5000

ENTRYPOINT FLASK_APP=flaskController.py flask run -h 0.0.0.0 -p 5000
#CMD ["/bin/bash"]
