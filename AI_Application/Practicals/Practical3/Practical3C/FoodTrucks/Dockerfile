# specify your base image to be ubuntu:18.04
FROM ubuntu:18.04 

# write down your name and email as the maintainer
LABEL maintainer="Sai Keerthan"
LABEL email="sai.keerthan01@gmail.com"


# install system-wide deps for python and node
RUN apt-get -yqq update
RUN apt-get -yqq install python3-pip python3-dev curl gnupg
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash
RUN apt-get install -yq nodejs

# install build dependencies for pip packages with native extensions
RUN apt-get install -y build-essential libffi-dev libssl-dev

# copy our application code
ADD flask-app /opt/flask-app

# set working directory to be /opt/flask-app
WORKDIR /opt/flask-app


# fetch app specific deps
RUN npm install
RUN npm run build

# install dependencies from requirements.txt - need to use pip3
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt



# expose port 5000
EXPOSE 5000

# start app by running app.py - need to use python3
CMD ["python3", "app.py"]
