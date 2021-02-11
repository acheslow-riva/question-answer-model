FROM amazonlinux:2

COPY ./requirements.txt /requirements.txt
RUN echo $'[neuron] \nname=Neuron YUM Repository \nbaseurl=https://yum.repos.neuron.amazonaws.com \nenabled=1 \nmetadata_expire=0' > /etc/yum.repos.d/neuron.repo &&\
    rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB &&\
    yum update -y &&\
    yum install -y git kernel-devel.x86_64 aws-neuron-dkms aws-neuron-runtime-base \
        aws-neuron-runtime aws-neuron-tools python3 gcc-c++ &&\
    echo $'[global]\nextra-index-url = https://pip.repos.neuron.amazonaws.com' > /etc/pip.conf &&\
    pip3 install --upgrade pip &&\
    pip install --no-cache-dir -r requirements.txt
    # git clone https://github.com/deepset-ai/FARM.git &&\
    # git clone https://github.com/deepset-ai/haystack.git &&\
    # cd /FARM && git checkout v0.5.0 && pip install --editable . &&\
    # cd /haystack && pip install --editable .

# COPY docker_dependencies/language_model.py /FARM/farm/modeling/language_model.py
# COPY docker_dependencies/infer.py /FARM/farm/infer.py

COPY transformer_model/ /transformer_model

WORKDIR /transformer_model
CMD tail -f /dev/null

# RUN yum install git -y && git clone https://github.com/deepset-ai/FARM.git
# RUN cd FARM && git checkout v0.5.0
# COPY docker_dependencies/requirements.txt /FARM/requirements.txt
# COPY docker_dependencies/language_model.py /FARM/farm/modeling/language_model.py
# COPY docker_dependencies/infer.py /FARM/farm/infer.py
# RUN cd FARM && pip install -r requirements.txt && \
#     pip install --editable . 

# # RUN git clone https://github.com/deepset-ai/haystack.git
# # RUN cd haystack && pip install --editable .

# COPY requirements.txt /transformer_model/
# WORKDIR /transformer_model
# RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install torch-neuron
# COPY transformer_model/ /transformer_model
# CMD python3 manage.py dev
# CMD tail -F etc/hosts