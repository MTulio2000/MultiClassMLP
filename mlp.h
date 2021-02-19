#ifndef MLP_H
#define MLP_H
#include <eigen3/Eigen/Dense>
#include <vector>
#include <QFile>
#include <QTextStream>
#include <iostream>

using namespace std;
using namespace Eigen;

typedef struct Data
{
    vector<VectorXd> in;
    vector<VectorXd> answer;
}Data;

class MLP
{
private:
    int epochs;
    int _size;
    int dataSize;
    double minAccuracy = 0.8;
    double maxAccuracy = 0.9;
    double learningRate;
    vector<int> randIndexVector;
    double accuracy();
    vector<VectorXd> layers,values;
    vector<MatrixXd> weights;
    vector<VectorXd> bias;
    VectorXd activation(VectorXd l,bool der = false);
    void classification(VectorXd data);
    void forward();
    void backpropagation(VectorXd delta);
    void initializevalues();
    void initializevalues(vector<int>cfg);
    void openDatabase(string path);
    void fit(string path);
    int randIndex();
    Data data;
public:
    MLP(vector<int> cfg,string path,double lr = 0.001,int epochs = 100000,double minAccuracy = 0.8,double maxAccuracy = 0.9);
    int classify(VectorXd data);
};

#endif // MLP_H
