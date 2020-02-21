# Notes on working with the cluster

when copying over files or git cloning to the head node, remember to change the group permissions for writing files with:
```
ssh 2209560g@idagpu-head.dcs.gla.ac.uk
chgrp -R nfsnobody phd_by_carlos
chmod -R +w phd_by_carlos
```

pushing a container image to docker hub
```
docker login
docker build -t aquaktus/docker_ml_by_carlos:v2 -f ./Dockerfile.gpu.tf2.0.pytorch1.3 .
docker push aquaktus/docker_ml_by_carlos:v2
```

### Running an executable job on the cluster
Here we run a notebook as a script passing the location of a `config.json` file as an environment variable: `CONFIG`.
```
CONFIG=config.json runipy pytorch_copy_generator.ipynb
```

## OKD Config files

#### Cluster location
<https://console.ida.dcs.gla.ac.uk>


Notebook deployment config
```
apiVersion: apps.openshift.io/v1
kind: DeploymentConfig
metadata:
  name: pythonmlnotebookgpu2
  namespace: 2209560gproject
spec:
  replicas: 1
  strategy:
    resources: {}
  template:
    metadata:
      labels:
        app: pythonMLNotebook2
        deploymentconfig: pythonMLNotebookGPU
    spec:
      nodeSelector:
        node-role.ida/gputitan: "true"
        # node-role.ida/gpu2080ti: "true"
      containers:
      - name: deepo-ml-plus
        image: aquaktus/docker_ml_by_carlos:v3
        resources:
          requests:
            cpu: "1500m"
            memory: "4Gi"
            nvidia.com/gpu: 0 # this allows you to access all GPUs at the same time
          limits:
            cpu: "16000m"
            memory: "16Gi"
            nvidia.com/gpu: 0 # this allows you to access all GPUs at the same time
        command:
          - 'jupyter-lab'
        args:
          - '--no-browser'
          - '--ip=0.0.0.0'
          - '--allow-root'
          - '--NotebookApp.token='
          - '--notebook-dir="/nfs/"'
        ports:
          - containerPort: 8888
        imagePullPolicy: IfNotPresent
        volumeMounts:
          - mountPath: /nfs/
            name: nfs-access
      securityContext: {}
      serviceAccount: containerroot
      volumes:
      - name: nfs-access
        persistentVolumeClaim:
          claimName: 2209560gvol1claim


```

Service Config:
```
apiVersion: v1
kind: Service
metadata:
  name: jupyterservice
  namespace: 2209560gproject      
spec:
  selector:                  
    app: pythonMLNotebook   
  ports:
  - name: jupyter
    port: 8888               
    protocol: TCP
    targetPort: 8888         

```

Route Config:
```
apiVersion: v1
kind: Route
metadata:
  name: jupyterroute
  namespace: 2209560gproject
spec:
  path: /
  to:
    kind: Service
    name: jupyterservice
  port:
    targetPort: 8888

```


### Single Pod GPU script
##### CoNaLa Example
```
apiVersion: v1
kind: Pod
metadata:
  name: conala-tiny-transformer-custom-tok-75-seq-len-850-vocab
  namespace: 2209560gproject
spec:
  volumes:
  - name: nfs-access
    persistentVolumeClaim:
      claimName: 2209560gvol1claim
  nodeSelector:
    # node-role.ida/gputitan: "true"
    node-role.ida/gpu2080ti: "true"
  containers:
  - env: 
    name: conala-tiny-transformer-custom-tok-75-seq-len-850-vocab
    image: aquaktus/docker_ml_by_carlos:v6
    resources:
      requests:
        cpu: "1500m"
        memory: "4Gi"
        nvidia.com/gpu: 1 # this allows you to access all GPUs at the same time
      limits:
        cpu: "16000m"
        memory: "16Gi"
        nvidia.com/gpu: 1 # this allows you to access all GPUs at the same time
    imagePullPolicy: IfNotPresent
    command:
      - 'python3'     
    args:
      - '/nfs/phd_by_carlos/notebooks/main_autoReg.py'
      - '--output_dir'
      - '/nfs/phd_by_carlos/notebooks/conala-tiny-transformer-custom-tok-75-seq-len-850-vocab'
      - '--src_train_fp'
      - '/nfs/phd_by_carlos/notebooks/datasets/CoNaLa/conala-train.src'
      - '--tgt_train_fp'
      - '/nfs/phd_by_carlos/notebooks/datasets/CoNaLa/conala-train.tgt'
      - '--src_test_fp'
      - '/nfs/phd_by_carlos/notebooks/datasets/CoNaLa/conala-test.src'
      - '--tgt_test_fp'
      - '/nfs/phd_by_carlos/notebooks/datasets/CoNaLa/conala-test.tgt'
      - '--steps'
      - '500000'
      - '--log_interval'
      - '100'
      - '--max_seq_len'
      - '75'
      - '--eval_interval'
      - '5000'
      - '--save_interval'
      - '5000'
      - '--layers'
      - '2'
      - '--att_heads'
      - '4'
      - '--embed_dim'
      - '512'
      - '--dim_feedforward'
      - '1024'
    volumeMounts: 
    - mountPath: /nfs/
      name: nfs-access
  serviceAccount: containerroot
  restartPolicy: Never
```

##### Django validation example
```
apiVersion: v1
kind: Pod
metadata:
  name: conala-tiny-transformer-custom-tok-75-seq-len-850-vocab
  namespace: 2209560gproject
spec:
  volumes:
  - name: nfs-access
    persistentVolumeClaim:
      claimName: 2209560gvol1claim
  nodeSelector:
    # node-role.ida/gputitan: "true"
    node-role.ida/gpu2080ti: "true"
  containers:
  - env: 
    name: conala-tiny-transformer-custom-tok-75-seq-len-850-vocab
    image: aquaktus/docker_ml_by_carlos:v6
    resources:
      requests:
        cpu: "1500m"
        memory: "4Gi"
        nvidia.com/gpu: 1 # this allows you to access all GPUs at the same time
      limits:
        cpu: "16000m"
        memory: "16Gi"
        nvidia.com/gpu: 1 # this allows you to access all GPUs at the same time
    imagePullPolicy: IfNotPresent
    command:
      - 'python3'     
    args:
      - '/nfs/phd_by_carlos/notebooks/main_autoReg.py'
      - '--dataset_prefix'
      - '/nfs/phd_by_carlos/notebooks/datasets/django_folds/django.fold1-10'
      - '--train_prefix'
      - '.train'
      - '--valid_prefix'
      - '.valid'
      - '--test_prefix'
      - '.test'
      - '--src_prefix'
      - '.src'
      - '--steps'
      - '200000'
      - '--log_interval'
      - '100'
      - '--max_seq_len'
      - '75'
      - '--eval_interval'
      - '5000'
      - '--save_interval'
      - '5000'
      - '--layers'
      - '2'
      - '--att_heads'
      - '4'
      - '--embed_dim'
      - '512'
      - '--dim_feedforward'
      - '1024'
      - '--output_dir'
      - '/nfs/phd_by_carlos/notebooks/django-tiny-transformer-valid-testing'
    volumeMounts: 
    - mountPath: /nfs/
      name: nfs-access
  serviceAccount: containerroot
  restartPolicy: Never
```
