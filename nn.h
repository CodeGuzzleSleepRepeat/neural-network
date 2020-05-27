#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include "functions.h"
#include <time.h>

using namespace std;

class func_obj{
public:
    func_obj(){};
    virtual double func(double) = 0;
    virtual double der(double) = 0;
    virtual double func(double, double) = 0;
    virtual double der(double, double) = 0;
};


class sq : public func_obj{
public:
    double func(double);
    double der(double);
    double func(double, double);
    double der(double, double);
};

class sigm_err : public func_obj{
public:
    double func(double);
    double der(double);
    double func(double, double);
    double der(double, double);
};

class sigm : public func_obj{
public:
    double func(double);
    double der(double);
    double func(double, double);
    double der(double, double);
};

class logfm : public func_obj{
public:
    double func(double);
    double der(double);
    double func(double, double);
    double der(double, double);
};

class relu : public func_obj{
public:
    double func(double);
    double der(double);
    double func(double, double);
    double der(double, double);
};

class nn
{
    int num_of_layers;
    vector<vector<double>> weights;
    vector<int> nums_of_neurons;
    vector<vector<double>> connections;
    int type;
    double (*ch_h)(int) = NULL;
    double norm_coef;
    double delete_neuron_lim;
    vector<func_obj *> funcs;
public:
    nn(int nl, vector<int> nns, vector<string> func_name, int u_type, double u_nc = 0, double u_dnl = 0,
       double (*)(int) = NULL);
    void init_weights(int size);
    void init_values(vector<vector<double>> &vec);
    double step(vector<double> &x, double h, double answer, int num, int layer);
    double der_and_step(int layer, int neuron, vector<double> &data, double res, double h);
    double choose_h(int);
    void fit(vector<double> &data, double res, double&, double);
    double count_cur_mistake(vector<double> &data, double res);
    double predict(vector<double> &data);
    double predict(vector<vector<double>> &data, vector<double> &);
    void input_weights(ifstream &stream);
    void output_weights(ofstream &stream);
    void delete_char(int);
    int delete_worst_neuron(vector<vector<double>> &data, vector<double> &res, int, int);
};


