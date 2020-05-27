#include "functions.h"

void get_data(ifstream &f, vector<vector<double>> &chars, vector<double> &answers, int num){
    string tmp, tmp2;
    vector<double> tmp_vec;
    if (!f.is_open()){
        cout << "Error" << endl;
        return;
    }
    int i = 0, j = 0;
    getline(f, tmp);
    while (f){
        chars.push_back(tmp_vec);
        getline(f, tmp);
        stringstream ss(tmp);
        while(ss){
            getline(ss, tmp2, ',');
            if (tmp2[0] == '.') tmp2 = '0' + tmp2;
            if (j == 0)
                answers.push_back(atof(tmp2.c_str()));
            else chars[i].push_back(atof(tmp2.c_str()));
            j++;
        }
        chars[i][j - 2] = -1;
        j = 0;
        i++;
        if (i == num) break;
    }
}

double metrics(vector<double> arr1, vector<double> arr2, double p){
    double sum = 0;
    int size = arr1.size();
    for (int i = 1; i < size; i++){
        sum += abs(pow(arr1[i] - arr2[i], p));
    }
    sum = pow(sum, 1/p);
    return sum;
}

double spec_scalar_mult(vector<double> &x, vector<double> &weights){
    double sum = 0;
    int size = weights.size(), ans = x[81] < x[163] ? 1 : -1;
    for (int i = 0; i < size; i++){
        if (i == 81 || i == 163) continue;
        sum += x[i] * weights[i];
    }
    return sum * ans;
}

double scalar_mult(vector<double> &x, vector<double> &weights){
    double sum = 0;
    int size = x.size();
    for (int i = 0; i < size; i++){
        sum += x[i] * weights[i];
    }
    return sum;
}


void scale1(vector<vector<double>> &chars, int size){
    vector<double> med(81), max(81);
    int inner_size = chars[0].size();
    for (int k = 0; k < inner_size - 1; k++) med[k] = 0;
    for (int j = 0; j < inner_size - 1; j++){
        for (int i = 0; i < size; i++){
            med[j] += chars[i][j];
        }
        med[j] /= size;
        max[j] = 0;
        for (int i = 0; i < size; i++){
            chars[i][j] -= med[j];
            if (abs(chars[i][j]) > max[j]) max[j] = abs(chars[i][j]);
        }
    }
    for (int j = 0; j < inner_size - 1; j++){
        for (int i = 0; i < size; i++) chars[i][j] = chars[i][j] / max[j];
    }
}

void scale2(vector<double> &ans, double &max, double &med){
    for (auto &el : ans){
        med += el;
    }
    med /= ans.size();
    for (auto &el : ans){
        el -= med;
        if (max < el) max = el;
    }
    for (auto &el : ans) el /= max;
}

void scale_back_ans(vector<double> &ans, double max, double med){
    for (auto &el : ans){
        el *= max;
        el += med;
    }
}

void shuffle(vector<vector<double>> &chars, int size, vector<double> & answers){
    srand(time(NULL));
    int inner_size = chars[0].size(), r;
    double tmp, tmp2;
    for (int i = 0; i < size; i++){
        r = rand() % size;
        tmp2 = answers[i];
        answers[i] = answers[r];
        answers[r] = tmp2;
        for (int j = 0; j < inner_size; j++){
            tmp = chars[i][j];
            chars[i][j] = chars[r][j];
            chars[r][j] = tmp;
        }
    }
}
