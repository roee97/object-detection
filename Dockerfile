FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

#
# SSH Setup
#

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:Roee0311' | chpasswd

# Allow logging in to root with password (this might be different for different ubuntu versions)
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

#ENV NOTVISIBLE "in users profile"
#RUN echo "export VISIBLE=now" >> /etc/profile
CMD service ssh start

# Init conda in bashrc
RUN conda init bash

#
# Libraries
#

RUN apt update && apt install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN conda install -c conda-forge matplotlib scikit-image jupyter albumentations scikit-learn opencv
# pycocotools
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install pytorch_lightning

# Optional
# RUN pip install Flask gunicorn opencv-python Polygon3 easydict rectpack pyclipper


#
# Finish
#

WORKDIR "/"
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
