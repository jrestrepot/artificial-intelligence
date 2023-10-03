# artificial-intelligence

Artificial Intelligence course work and projects.

In this repository we have five folders, which will be explained below 

## MLP:
This folder includes the code for a multilayer perceptron. It has two mains, one
main trains 45 different architectures of MLPs and then compares their losses. This
main is called mlp_main.py, and you can run it with the command "python MLP/mlp_main.py".
The other main is called encoder_main.py. It trains and tests two autoencoders:
a dimensionality reduction and a dimensionality expansion autoencoder. Then, it trains
a Pytorch's neural network on their latent spaces and tries to reconstruct the original
data.
## CNN (LeNet-5)
This folder includes the LeNet-5 architecture, and its main.py trains it with the original
MNIST code and then tests it with my classmates' handwritten digits. To run this code,
use "python CNN/main.py"
## GAN
The GAN folder includes a .ipynb that trains a GAN on the MNIST data. This notebook
installs all of its requirements in the first cell and can be run like one usually runs
a notebook
## Unsupervised
The Unsupervised folder contains a set of tools and algorithms to perform clustering.
Currently there are three algorithms:

- K-Nearest Neighbors 
- Connected Components Clustering
- Distance-based clustering

In the main.py we used global parameters that you can change to play with the different
algorithms. To run this, simply use "python Unsupervised/main.py"

## figures
The figures folder consists of .html and .png files, which are visualizations of the 
results for the models in this repository. The file names are very self-explanatory.

# SET UP
To set up the virtual environment, simple run the .sh file with the command: ".\setup.sh"