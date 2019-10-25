# phd_by_carlos
This is a repository designed to be my portable workstation. I can use it on a CPU or GPU machine and is meant for development and training. It contains docker files for environment setup and links to other repositories that can be downloaded inside.

# Setup

## SSH
If you are using SSH to connect to a remote instance, remember to port forward :8888 to access the jupyter environment.
```
$ ssh -L 8889:localhost:8888 aquaktus@carlos-oldie.local
```

## Docker
You will need `docker or docker-gpu` since all environments require the generic environment.

On CPU
```
$ cd phd_by_carlos/docker
$ bash start_cpu_env.sh ./Dockerfile.cpu.tf2.0.pytorch1.3
```

On GPU
```
$ cd phd_by_carlos/docker
$ bash start_gpu_env.sh ./Dockerfile.gpu.tf2.0.pytorch1.3
```

Once in the docker container you can run the following to start the jupyter lab environment. Remeber you can either access the lab or notebook environemnt by going to `/lab` or `/tree`.
```
$ jupyter notebook --ip 0.0.0.0 --no-browser --allow-root &
```

