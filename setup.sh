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

docker-compose down
docker-compose build
docker-compose up -d node-1 
sleep 35
docker-compose up -d
