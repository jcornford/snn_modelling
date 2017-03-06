
# Spiking neural network and Single Neuron modelling:
Just version control for the code

### Running NEURON docker image:

Docker image from jcornford/jupyter-neuron. (https://hub.docker.com/r/jcornford/jupyter-neuron/)
 
Dockerfile is essentially the same as
https://github.com/NeuralEnsemble/neuralensemble-docker/tree/master/simulation
only slightly modified with changes suggested in this thread:
https://groups.google.com/forum/#!topic/neuralensemble/4btzM0n_9O4

```
docker run -it --rm -p 8888:8888 --name NEURON -v /Volumes/LACIE/Neuron2017:/home/docker/mounted_vol jcornford/jupyter-neuron
```

For making movie of simulation results:

```
ffmpeg -r 1 -i image-%d.png -c:v libx264 -vf fps=40 -pix_fmt yuv420p out.mp4
```

### Network modelling
To upload main stuff



