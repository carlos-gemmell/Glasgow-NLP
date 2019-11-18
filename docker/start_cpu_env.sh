if [ -z "$1" ]; then echo "No args, please specify path to Dockerfile"; exit 0; fi
echo "taking Dockerfile from:" $1
docker build -t deep_phd_env_img -f $1 .
docker run --name deep_phd_env --rm -p ${2:-8888}:${2:-8888} -v $PWD/../:/my_shared/ deep_phd_env_img jupyter-lab --ip 0.0.0.0 --no-browser --allow-root --notebook-dir="/my_shared"
