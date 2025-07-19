FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
      curl \
      gnupg
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list \
 && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
 && apt-get update

RUN DEBIAN_FRONTEND=noninteractive \
    TZ="Europe/London" \
    apt-get install -y \
      libedgetpu1-std \
      python3 \
      python3-pip \
      python3-opencv \
      python3-pycoral \
 && rm -rf /var/lib/apt/lists/*
RUN pip install pipenv

RUN mkdir /app
WORKDIR /app

COPY Pipfile Pipfile.lock ./

RUN pipenv install --system --deploy --ignore-pipfile

COPY entrypoint.py cctv_filter.py reolink_camera.py reolink_video.py detection.py labels.txt ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite ./

CMD [ "python3", "./entrypoint.py" ]
