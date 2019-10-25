# Notes on working with the cluster

when copying over files or git cloning to the head node, remember to change the group permissions for writing files with:
```
ssh 2209560g@idagpu-head.dcs.gla.ac.uk
chgrp -R nfsnobody phd_by_carlos
chmod -R +w phd_by_carlos
```

Notebook deployment config
```
apiVersion: apps.openshift.io/v1
kind: DeploymentConfig
metadata:
  name: pythonmlnotebookgpu
  namespace: 2209560gproject
spec:
  replicas: 1
  strategy:
    resources: {}
  template:
    metadata:
      labels:
        app: pythonMLNotebook
        deploymentconfig: pythonMLNotebookGPU
    spec:
      nodeSelector:
        node-role.ida/gputitan: "true"
        # node-role.ida/gpu2080ti: "true"
      containers:
      - name: deepo-ml-plus
        image: aquaktus/docker_test_repo:v1
        resources:
          requests:
            cpu: "1500m"
            memory: "4Gi"
            nvidia.com/gpu: 1
          limits:
            cpu: "2500m"
            memory: "16Gi"
            nvidia.com/gpu: 1
        command:
          - 'jupyter'
        args:
          - 'notebook'
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