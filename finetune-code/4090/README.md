## Quick start to fine tune in 4090

```
# Build
$ docker build -t neurips-4090-train:latest  .
$ docker exec -ti <container_id> bash

# Inside the container run
$ python train.py
```
