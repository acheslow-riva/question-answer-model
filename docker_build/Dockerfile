FROM amazonlinux:2

COPY ./requirements.txt /requirements.txt
RUN echo $'[neuron] \nname=Neuron YUM Repository \nbaseurl=https://yum.repos.neuron.amazonaws.com \nenabled=1 \nmetadata_expire=0' > /etc/yum.repos.d/neuron.repo &&\
    rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB &&\
    yum update -y &&\
    yum install -y git kernel-devel.x86_64 aws-neuron-dkms aws-neuron-runtime-base \
        aws-neuron-runtime aws-neuron-tools python3 gcc-c++ &&\
    echo $'[global]\nextra-index-url = https://pip.repos.neuron.amazonaws.com' > /etc/pip.conf &&\
    pip3 install --upgrade pip &&\
    pip install --no-cache-dir -r requirements.txt &&\
    python3 -m nltk.downloader punkt

WORKDIR /transformer_model
ENV FLASK_APP manage.py
CMD tail -f /dev/null