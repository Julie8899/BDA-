FROM ubuntu:18.04
COPY . /scripts
WORKDIR /scripts
RUN apt-get update && apt-get install -y default-mysql-client
CMD ["sh", "-c", "./run.sh"]