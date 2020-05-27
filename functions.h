#pragma once
#include <vector>
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>
#include <fstream>
#include <time.h>
using namespace std;


void get_data(ifstream &f, vector<vector<double>> &chars, vector<double> &answers, int num);
double metrics(vector<double> arr1, vector<double> arr2, double p);
double spec_scalar_mult(vector<double> &x, vector<double> &weights);
double scalar_mult(vector<double> &x, vector<double> &weights);
void scale1(vector<vector<double>> &chars, int size);
void scale2(vector<double> &ans, double &, double &);
void scale_back_ans(vector<double> &ans, double max, double med);
void shuffle(vector<vector<double>> &chars, int size, vector<double> & answers);
