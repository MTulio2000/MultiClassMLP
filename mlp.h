#ifndef MLP_H
#define MLP_H
#include <eigen3/Eigen/Dense>
#include <vector>
#include <QFile>
#include <QTextStream>
#include <iostream>

using namespace std;
using namespace Eigen;

class MLP
{
private:
    int epochs;
    int _size;
    double minAccuracy = 0.8;
    double maxAccuracy = 0.9;
    double learningRate;
    double accuracy(vector<VectorXd> in, vector<int> out);
    vector<VectorXd> layers,values;
    vector<MatrixXd> weights;
    vector<VectorXd> bias;
    VectorXd activation(VectorXd l,bool der = false);
    void classification(VectorXd data);
    void forward();
    void backpropagation(VectorXd delta);
    void initializevalues();
    void initializevalues(vector<int>cfg);
    void openDatabase(string path,vector<VectorXd> &in, vector<int> &out);
    void fit(string path);
public:
    MLP(vector<int> cfg,string path,double lr = 0.001,int epochs = 100000);
    int classify(VectorXd data);
};

#endif // MLP_H
