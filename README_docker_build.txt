Use continuumio/miniconda3:latest from DockerHub
docker login
docker build -t media_extractor . --progress=plain
docker run -p 8501:8501 media_extractor
docker run -p 8501:8501 -t media_extractor /bin/bash  # acts as a VM
docker system prune -a
docker rm <id_or_name>
docker rmi <name:tag>
docker ps -a 
docker images
docker tag media_extractor dshipley71/media_extractor:latest
docker image push dshipley71/media_extractor:latest
docker pull dshipley71/media_extractor
docker run -p 8501:8501 -t dshipley71/media_extractor
http://localhost:8501

Update project models in container:
(1) Perform initial build on the the low-side.
(2) Run a test on each page. This will download required models to the container.
(3) Login to the container and remove /media_extractor/output folder.
(4) Update container on DockerHub.

Notes:
- Media Extractor and Cluster & Pose output results need to have a download link
added to the page so that a user can get data.

To run ANY docker container, do the following:
(1) Install docker for your operating system

Run the following commands from docker desktop or terminal
(1) docker pull dshipley71/media_extractor
(2) docker run -p 8501:8501 -t dshipley71/media_extractor
(3) http://localhost:8051
(4) Enjoy!

To get the output from ME and CP, you will have to get inside the container:
(1) docker run -p 8501:8501 -t dshipley71/media_extractor
(2) Use terminal or Docker Desktop terminal to navigate to /media_extractor/output
- Note: This is a Linux container. Know your linux commands!
(3) Use SSH/SFTP to transfer data from container to local system.
- TODO: Add a downlink to the ME and CP web pages

https://hub.docker.com/repository/docker/dshipley71/media_extractor
