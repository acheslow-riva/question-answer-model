FROM amazonlinux:2

RUN echo $'[neuron] \n\
name=Neuron YUM Repository \n\
baseurl=https://yum.repos.neuron.amazonaws.com \n\
enabled=1 \n\
metadata_expire=0' > /etc/yum.repos.d/neuron.repo
RUN rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
Run yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r) 2>&1 > /etc/build.log || echo "Yum didn't install something correctly!" 
RUN yum install aws-neuron-dkms -y
RUN yum install aws-neuron-runtime-base -y
RUN yum install aws-neuron-runtime -y
RUN yum install aws-neuron-tools -y
RUN yum update -y && yum install -y python3 gcc-c++

RUN echo $'[global]\n\
extra-index-url = https://pip.repos.neuron.amazonaws.com' > /etc/pip.conf

RUN pip3 install --upgrade pip
RUN pip install neuron-cc[tensorflow]
RUN pip install torch-neuron

RUN yum install git -y && git clone https://github.com/deepset-ai/FARM.git
RUN cd FARM && git checkout v0.5.0
COPY docker_dependencies/requirements.txt /FARM/requirements.txt
COPY docker_dependencies/language_model.py /FARM/farm/modeling/language_model.py
COPY docker_dependencies/infer.py /FARM/farm/infer.py
RUN cd FARM && pip install -r requirements.txt && \
    pip install --editable . 

# RUN git clone https://github.com/deepset-ai/haystack.git
# RUN cd haystack && pip install --editable .

COPY requirements.txt /transformer_model/
WORKDIR /transformer_model
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch-neuron
COPY transformer_model/ /transformer_model
CMD python3 manage.py dev
# CMD tail -F etc/hosts