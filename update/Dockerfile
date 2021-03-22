FROM docker.ahrq.local:5000/python:3.7
ADD requirements.txt /app/

WORKDIR /app

# Start-cron script from https://github.com/renskiy/cron-docker-image/tree/master/debian
COPY start-cron /usr/sbin

RUN chmod +x /usr/sbin/start-cron && \
    apt-get update && \
    apt-get install -y --no-install-recommends cron && \
    apt-get clean && \
    pip install --no-cache-dir -r requirements.txt && \
    mkfifo --mode 0666 /var/log/cron.log