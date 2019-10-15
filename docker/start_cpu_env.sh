if [ -z "$1" ]; then echo "No args, please specify path to Dockerfile"; exit 0; fi
echo "taking Dockerfile from:" $1
docker build -t deep_phd_env_img -f $1 .
docker run all --name deep_phd_env -it -p 8888:8888 -v $PWD/../:/my_shared/ -d deep_phd_env_img 
docker exec -it deep_phd_env bash
docker stop deep_phd_env
docker rm deep_phd_env