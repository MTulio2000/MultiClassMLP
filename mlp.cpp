#include "mlp.h"

MLP::MLP(vector<int> cfg,string path, double lr, int epochs)
{
    this->learningRate = lr;
    this->epochs = epochs;
    initializevalues(cfg);
    this->fit(path);
}

void MLP::fit(string path_file)
{
    vector<VectorXd> in;
    vector<int> out;
    VectorXd answer;
    openDatabase(path_file,in,out);
    int index,epochs = 0,at;
    srand(time(NULL));
    vector<int> randIndex;
    double acc = accuracy(in,out);
    while((acc<minAccuracy||acc>maxAccuracy)&&epochs++<this->epochs)
    {
        for(int i = 0; i<(int)out.size();i++)
            randIndex.push_back(i);
        for(int i = 0; i < (int)out.size()&&randIndex.size();i++)
        {
            answer = VectorXd::Zero(layers[_size-1].size());
            at = rand()%randIndex.size();
            index = randIndex[at];
            classification(in[index]);
            answer[out[index]] = 1.0;
            randIndex.erase(randIndex.begin()+at);
            backpropagation(answer-layers[_size-1]);
        }
        acc = accuracy(in,out);
        if(acc>maxAccuracy)
            initializevalues();
    }
    cout << "Accuracy: "<<acc << endl;
    cout << "Epochs: "<<epochs << endl;
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
    vector<int>cfg;
    cfg.push_back(weights[0].rows());
    for(int i = 0; i < (int)weights.size();i++)
        cfg.push_back(weights[i].cols());
    initializevalues(cfg);
}

void MLP::initializevalues(vector<int> cfg)
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

double MLP::accuracy(vector<VectorXd> in, vector<int> out)
{
    double hit = 0;
    int n = out.size();
    for(int i = 0; i < n;i++)
        if(classify(in[i])==out[i])
            hit++;
    return hit/(double)n;
}

VectorXd MLP::activation(VectorXd l,bool der)
{
    for(int i = 0; i < l.size();i++)
        l[i] = der?(l[i]*(1.0 - l[i])):1.0/(1.0 + expf(-l[i]));//sigmoid
    return l;
}

void MLP::openDatabase(string path,vector<VectorXd> &x, vector<int> &y)
{
    QFile file(QString::fromStdString(path));
    if(file.open(QIODevice::ReadOnly|QIODevice::Text))
    {
        QStringList row;
        vector<QString> list;
        QString line;
        int index = 0;
        QTextStream in(&file);
        in.readLine();
        while (!in.atEnd())
        {
            row.clear();
            line = in.readLine();
            row = line.split(",");

            VectorXd v(row.size()-1);
            for(int i = 0; i < row.size()-1;i++)
                v[i] = row[i].toDouble();
            if(v.size())
            {
                x.push_back(v);
                index = -1;
                for(int i = 0; i < (int)list.size();i++)
                {
                    if(list[i]==row[row.size()-1])
                    {
                        index = i;
                        break;
                    }
                }
                if(index == -1)
                {
                    index = list.size();
                    list.push_back(row[row.size()-1]);
                }
                y.push_back(index);
            }
        }
        file.close();
    }
}
