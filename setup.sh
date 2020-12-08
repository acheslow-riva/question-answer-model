#!/bin/bash
case $HOSTNAME in
"ml-dev-01.ahrq.local")
    MODE="DEV"
    ;;
*)
    MODE="LOCAL"
    ;;
esac

read_var() {
    VAR=$(grep $1 $2 | xargs)
    IFS="=" read -ra VAR <<< "$VAR"
    echo ${VAR[1]}
}

export AWS_ACCESS_KEY_ID=$(read_var AWS_ACCESS_KEY_ID .env)
export AWS_SECRET_ACCESS_KEY=$(read_var AWS_SECRET_ACCESS_KEY .env)

if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "AWS credentials not found!"
    exit 1
fi

if [ $MODE == "LOCAL" ]; then
    docker-compose -f docker-compose-dev.yml down
    docker-compose -f docker-compose-dev.yml build
    docker-compose -f docker-compose-dev.yml up -d node-1 
    sleep 35
    docker-compose -f docker-compose-dev.yml up -d
else
    mkdir /tmp/neuron_rtd_sock
    chmod o+rwx /tmp/neuron_rtd_sock
    sudo service neuron-rtd stop
    docker-compose down
    docker-compose build
    docker-compose up -d node-1 
    sleep 35
    docker-compose up -d
fi

