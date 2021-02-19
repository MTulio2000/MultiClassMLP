# MultiClassMLP

The MLP class is easy to use, but firt you need to install some libraries.
I used Ubuntu as a development environment and ran the following command to install the libraries
`sudo apt-get install -y qt5-default libeigen3-dev`

How can you use it?
The MLP class wait for this arguments:
* Neural Network configuration;
* Path to dataset;
* The learning rate;
* Epochs;
* Minimum accuracy;
* Maximum accuracy;

Example 1:
``` C++
//MLP *nn = new MLP(conf,path,lr,epochs,min,max);
MLP *nn = new MLP({4,3,3},"iris.csv",0.001,1000,0.8,0.9);
```
The configuration and the path to dataset are required, but the others aren't.

Example 2:
``` C++
MLP *nn = new MLP({4,3,3},"iris.csv");
```
By default, learningRate is 0.001, epochs is 100000, minAcurracy is 0.8 and the maximun accuracy is 0.9

If you need help, you can send me an E-mail at: marco2000carvalho@gmail.com
