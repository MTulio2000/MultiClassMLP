#include "mlp.h"

MLP::MLP(vector<int> cfg,QString path, double lr, int epochs,double minAccuracy,double maxAccuracy)
{
    this->minAccuracy = minAccuracy;
    this->maxAccuracy = maxAccuracy;
    this->learningRate = lr;
    this->epochs = epochs;
    this->cfg = cfg;
    initializevalues();
    this->fit(path);
}

void MLP::fit(QString path_file)
{
    openDatabase(path_file);
    int index,epochs = 0;
    double acc = accuracy();
    cout << "Initial Accuracy: " << acc<<endl;
    while((acc<minAccuracy||acc>maxAccuracy)&&epochs++<this->epochs)
    {
        for(int i = 0; i < dataSize;i++)
        {
            index = randIndex();
            classification(data.in[index]);
            backpropagation(data.answer[index]-layers[_size-1]);
        }
        acc = accuracy();
        if(acc>maxAccuracy)
            initializevalues();
        cout << "Accuracy: " << acc<<endl;
    }
    cout << "End Accuracy: " << acc<<endl;
    cout << "Epochs: " << epochs << endl;
}

int MLP::randIndex()
{
    int at,index;
    if(!randIndexVector.size())
        for(int i = 0; i<(int)data.answer.size();i++)
            randIndexVector.push_back(i);
    at = rand()%randIndexVector.size();
    index = randIndexVector[at];
    randIndexVector.erase(randIndexVector.begin()+at);
    return index;
}

int MLP::classify(VectorXd data)
{
    classification(data);
    int index = 0;
    double value = -INFINITY;
    for(int i = 0; i < layers[_size-1].size();i++)
        if(layers[_size-1][i]>value)
            value = layers[_size-1][i],index = i;
    return index;
}

void MLP::classification(VectorXd data)
{
    layers[0] = data;
    forward();
}

void MLP::forward()
{
    values[0] = layers[0];
    for(int i = 0; i < _size-1;i++)
    {
        values[i+1]=layers[i].transpose()*weights[i]+bias[i].transpose();
        layers[i+1] = activation(values[i+1]);
    }
}

void MLP::backpropagation(VectorXd error)
{
    VectorXd n_delta,delta(error.size()),derivative = activation(layers[_size-1],true);
    for(int i = 0; i < error.size();i++)
        delta[i] =-2 * derivative[i]*error[i];
    for(int i = _size-1;i>0;--i)
    {
        n_delta = VectorXd::Zero(layers[i-1].size());
        for(int j = 0; j < layers[i-1].size();j++)
            for(int k = 0; k < layers[i].size();k++)
                n_delta[j] += delta[k]*weights[i-1].coeff(j,k);
        for(int j = 0; j < layers[i-1].size();j++)
            for(int k = 0; k < layers[i].size();k++)
                weights[i-1].coeffRef(j,k) -= learningRate * delta [k] * values [i-1] [j];
        bias [i-1] -= learningRate * delta;
        delta = n_delta;
    }
}

void MLP::initializevalues()
{
    _size = cfg.size();
    weights.clear();
    bias.clear();
    layers.clear();
    values.clear();
    for(int i = 0; i < _size;i++)
    {
        if(i<_size-1)
        {
            weights.push_back(MatrixXd::Random(cfg[i],cfg[i+1]));
            bias.push_back(VectorXd::Random(cfg[i+1]));
        }
        layers.push_back(VectorXd::Zero(cfg[i]));
        values.push_back(VectorXd::Zero(cfg[i]));
    }
}

double MLP::accuracy()
{
    double hit = 0;
    int n = data.answer.size();
    for(int i = 0; i < n;i++)
        if(data.answer[i][classify(data.in[i])])
            hit++;
    return hit/(double)n;
}

VectorXd MLP::activation(VectorXd l,bool der)
{
    //sigmoid function
    for(int i = 0; i < l.size();i++)
        l[i] = der?(l[i]*(1.0 - l[i])):1.0/(1.0 + expf(-l[i]));//sigmoid
    return l;
}

void MLP::openDatabase(QString path)
{
    QFile *file = new QFile(path);
    file->open(QIODevice::ReadOnly|QIODevice::Text);
    if(file->isOpen())
    {
        QStringList row,list;
        QString line;
        QTextStream in(file);
        vector<Eigen::VectorXd> x;
        vector<int> y;
        Eigen::VectorXd answer,v;
        in.readLine();
        while (!in.atEnd())
        {
            row.clear();
            line = in.readLine();
            row = line.split(",");
            v = Eigen::VectorXd(row.size()-1);
            for(int i = 0; i < row.size()-1;i++)
                v[i] = row[i].toDouble();
            x.push_back(v);
            y.push_back((row[row.size()-1]).toInt());
        }
        file->close();
        for(int i = 0; i < (int)x.size();i++)
        {
            answer = Eigen::VectorXd::Zero(layers[_size-1].size());
            answer[y[i]] = 1;
            data.answer.push_back(answer);
            data.in.push_back(x[i]);
        }
        dataSize = x.size();
    }
    delete file;
}
