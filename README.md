# Generative Tensorial Reinforcement Learning (GENTRL) 
Supporting Information for the paper _"[Deep learning enables rapid identification of potent DDR1 kinase inhibitors](https://www.nature.com/articles/s41587-019-0224-x)"_.

The GENTRL model is a variational autoencoder with a rich prior distribution of the latent space. We used tensor decompositions to encode the relations between molecular structures and their properties and to learn on data with missing values. We train the model in two steps. First, we learn a mapping of a chemical space on the latent manifold by maximizing the evidence lower bound. We then freeze all the parameters except for the learnable prior and explore the chemical space to find molecules with a high reward.

![GENTRL](images/gentrl.png)


## Repository
In this repository, we provide an implementation of a GENTRL model with an example trained on a [MOSES](https://github.com/molecularsets/moses) dataset.

To run the training procedure,
1. [Install RDKit](https://www.rdkit.org/docs/Install.html) to process molecules
2. Install GENTRL model: `python setup.py install`
3. Install MOSES from the [repository](https://github.com/molecularsets/moses)
4. Run the [pretrain.ipynb](./examples/pretrain.ipynb) to train an autoencoder
5. Run the [train_rl.ipynb](./examples/train_rl.ipynb) to optimize a reward function

# tfGENTRL
implementation of GENTRL in Tensorflow


python setup.py install
running install
running bdist_egg
running egg_info
writing gentrl.egg-info/PKG-INFO
writing dependency_links to gentrl.egg-info/dependency_links.txt
writing top-level names to gentrl.egg-info/top_level.txt
reading manifest file 'gentrl.egg-info/SOURCES.txt'
writing manifest file 'gentrl.egg-info/SOURCES.txt'
installing library code to build/bdist.linux-x86_64/egg
running install_lib
running build_py
copying gentrl/encoder.py -> build/lib/gentrl
copying gentrl/decoder.py -> build/lib/gentrl
copying gentrl/dataloader.py -> build/lib/gentrl
copying gentrl/tokenizer.py -> build/lib/gentrl
copying gentrl/gentrl.py -> build/lib/gentrl
copying gentrl/lp.py -> build/lib/gentrl
copying gentrl/__init__.py -> build/lib/gentrl
creating build/bdist.linux-x86_64/egg
creating build/bdist.linux-x86_64/egg/gentrl
copying build/lib/gentrl/encoder.py -> build/bdist.linux-x86_64/egg/gentrl
copying build/lib/gentrl/decoder.py -> build/bdist.linux-x86_64/egg/gentrl
copying build/lib/gentrl/dataloader.py -> build/bdist.linux-x86_64/egg/gentrl
copying build/lib/gentrl/tokenizer.py -> build/bdist.linux-x86_64/egg/gentrl
copying build/lib/gentrl/gentrl.py -> build/bdist.linux-x86_64/egg/gentrl
copying build/lib/gentrl/lp.py -> build/bdist.linux-x86_64/egg/gentrl
copying build/lib/gentrl/__init__.py -> build/bdist.linux-x86_64/egg/gentrl
byte-compiling build/bdist.linux-x86_64/egg/gentrl/encoder.py to encoder.cpython-37.pyc
byte-compiling build/bdist.linux-x86_64/egg/gentrl/decoder.py to decoder.cpython-37.pyc
byte-compiling build/bdist.linux-x86_64/egg/gentrl/dataloader.py to dataloader.cpython-37.pyc
byte-compiling build/bdist.linux-x86_64/egg/gentrl/tokenizer.py to tokenizer.cpython-37.pyc
byte-compiling build/bdist.linux-x86_64/egg/gentrl/gentrl.py to gentrl.cpython-37.pyc
  File "build/bdist.linux-x86_64/egg/gentrl/gentrl.py", line 38
    class GENTRL(tf.Module)
                          ^
SyntaxError: invalid syntax

byte-compiling build/bdist.linux-x86_64/egg/gentrl/lp.py to lp.cpython-37.pyc
byte-compiling build/bdist.linux-x86_64/egg/gentrl/__init__.py to __init__.cpython-37.pyc
creating build/bdist.linux-x86_64/egg/EGG-INFO
copying gentrl.egg-info/PKG-INFO -> build/bdist.linux-x86_64/egg/EGG-INFO
copying gentrl.egg-info/SOURCES.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying gentrl.egg-info/dependency_links.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying gentrl.egg-info/top_level.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
zip_safe flag not set; analyzing archive contents...
creating 'dist/gentrl-0.1-py3.7.egg' and adding 'build/bdist.linux-x86_64/egg' to it
removing 'build/bdist.linux-x86_64/egg' (and everything under it)
Processing gentrl-0.1-py3.7.egg
Copying gentrl-0.1-py3.7.egg to /home/groups/ruthm/zyzhang/sw/sherlock2/anaconda-envs/tfGENTRL/lib/python3.7/site-packages
Adding gentrl 0.1 to easy-install.pth file

Installed /home/groups/ruthm/zyzhang/sw/sherlock2/anaconda-envs/tfGENTRL/lib/python3.7/site-packages/gentrl-0.1-py3.7.egg
Processing dependencies for gentrl==0.1
Finished processing dependencies for gentrl==0.1
