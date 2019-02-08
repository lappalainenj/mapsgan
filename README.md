# mapsgan
Multimodal pedestrian trajectory prediction.

### Core files
- The baseline and bicycle nn.Module is found in modules.py
- solver.py implements all solvers with respective generator- and discriminator steps
- utils.py implements all metrics
- sgan.py implements the code from the original SocialGAN paper

### Reproducing the results

- For all training procedures, we provide jupyter notebooks on "How to train...?".
- The data can be downloaded via the shell-script here https://github.com/agrimgupta92/sgan/tree/master/scripts
must be placed into ```../../data/```.
- Code is provided with docstrings and commented richly
- The results reported can be reproduced by putting the data inplace and running 
    - ```"/examples/16_evaluate_metrics.ipynb"``` for Figure 6 & 7 and Table 1
    - ```"/examples/yy_07_distribution plot.ipynb"``` for Figure 1
    - ```"yy_06_evaluate_saved_hyp.ipynb"``` for Figure 3-5
- scripts for training on a server are provided in the folder ```/scripts```
- final models can be found in ```models/final/eth```

