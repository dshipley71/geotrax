###############################################################################
### docker system prune -a                                      (remove images)
### docker build -t media_extractor . --progress=plain        (build container)
### docker run -p 8501:8501 -t media_extractor                (run from brower)
### docker run -it --entrypoint /bin/bash media_extractor  (login to container)
###############################################################################

### Latest miniconda3 docker container used from dockerhub.
FROM continuumio/miniconda3:latest

### Port to be used for displaying project webpage (http://localhost:8051)
EXPOSE 8501

### Project location in the docker container.
WORKDIR /media_extractor

### Update the linux OS contained in the docker container with project dependencies
RUN apt-get update && apt-get upgrade -y && apt-get -y install \
    python3-opencv \
    libmagic1 && \
    rm -rf /var/lib/apt/lists/*

# This will copy a python project to a Docker container. Make sure the
# Dockerfile is at the top of the project folder. Run the docker build
# command from the same location.
COPY . .

### Install project dependencies.
RUN pip3 install -r requirements.txt

### Launch streamlit application using a "docker run" command
ENTRYPOINT ["streamlit", "run", "media_extractor.py", "--server.port=8501", "--server.address=0.0.0.0"]
