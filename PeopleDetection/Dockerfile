FROM python:3.9

WORKDIR /usr/src/app/

COPY . /usr/src/app/

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install opencv-python-headless~=4.8.0

ARG ROOM=1
ENV ROOM="${ROOM}"
ARG TIME_ZONE=7
ENV TIME_ZONE="${TIME_ZONE}"

CMD python main.py $ROOM $TIME_ZONE