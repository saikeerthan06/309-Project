# Test web-app to use with Pluralsight courses and Docker Deep Dive book
FROM alpine
# Alpine is a docker image container

LABEL maintainer="nigelpoulton@hotmail.com"
# key is maintainer and the value is the email address

# Install Node and NPM, to make sure all of the dependecies are installed
RUN apk add --update nodejs npm curl

# Copy app to /src | location where the app will be copied into the image under the src directory
COPY . /src

# Set the working directory to /src | when the docker starts up it will go to that directory
WORKDIR /src 

# Install dependencies
RUN  npm install

# Expose port 8080 | this is the port that the app will be running on
EXPOSE 8080

# set the environment variable to production | this is the environment that the app will be running in
ENTRYPOINT ["node", "./app.js"]
