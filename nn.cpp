#include "nn.h"
using namespace std;


double sq::func(double x, double y){
    return pow(x - y, 2);
}

double sq::der(double x, double y){
    return 2 * (x - y);
}


double sigm_err::func(double x, double y){
    return exp(-x *y);
}

double sigm_err::der(double x, double y){
    return -y * exp(-x * y);
}

double sigm::func(double x, double y){
    return 1;
}

double sigm::der(double x, double y){
    return 1;
}

double relu::func(double x, double y){
    return 1;
}

double relu::der(double x, double y){
    return 1;
}

double logfm::func(double x, double y){

}

double logfm::der(double x, double y){

}

double sq::func(double x){
    return 1;
}

double sq::der(double x){
    return 1;
}

double sigm_err::func(double x){
    return 1;
}

double sigm_err::der(double x){
    return 1;
}

double sigm::func(double x){
    return 1 / (1 + exp(-x));
}

double sigm::der(double x){
    return exp(-x) / pow(1 + exp(-x), 2);
}

double logfm::func(double x){

}

double logfm::der(double x){

}

double relu::func(double x){
    return x > 0 ? x : 0;
}

double relu::der(double x){
    return x > 0 ? 1 : 0;
}




nn::nn(int nl, vector<int> nns, vector<string> func_name, int u_type, double u_nc, double u_dnl,
       double (*u_func)(int)){
    type = u_type;
    norm_coef = u_nc;
    delete_neuron_lim = u_dnl;
    ch_h = u_func;
    num_of_layers = nl;
    if (nns.size() != num_of_layers){
        cout << "Num of layers is incorrect";
        throw;
    }
    for (int i = 0; i < num_of_layers; i++){
        nums_of_neurons.push_back(nns[i]);
        vector<double> tmp(nums_of_neurons[i]);
        weights.push_back(tmp);
    }
    for (int i = 0; i < num_of_layers; i++){
        if (func_name[i] == "sigmoid")
            funcs.push_back(new sigm());
        if (func_name[i] == "log")
            funcs.push_back(new logfm());
        if (func_name[i] == "relu")
            funcs.push_back(new relu());
        if (func_name[i] == "sq")
            funcs.push_back(new sq());
    }
}

double nn::step(vector<double> &x, double h, double answer, int num, int layer){
    double norm = 0;
    int weights_size;
    int n = 0;
    for (int i = 0; i < layer; i++){
        for (int j = 0; j < nums_of_neurons[i]; j++)
            n++;
    }
    n += num;
    weights_size = weights[n].size();
    for (int i = 0; i < weights_size; i++) norm += weights[n][i] * weights[n][i];
    norm /= (double)weights_size;
    double sclr = funcs[layer]->der(scalar_mult(x, weights[n]));
    for (int i = 0; i < weights_size; i++){
        weights[n][i] -= h * sclr * x[i] * answer;
        if (weights[n][i] != 0)
            weights[n][i] -= norm * norm_coef * abs(weights[n][i]) / weights[n][i];
    }
    return sclr;
}


void nn::init_weights(int size){
    srand(time(NULL));
    int cur = 0, ss;
    double tmp_num;
    weights.clear();
    for (int i = 0; i < num_of_layers; i++){
        for (int j = 0; j < nums_of_neurons[i]; j++){
            vector<double> tmp(i == 0 ? size : nums_of_neurons[i - 1] + 1);
            weights.push_back(tmp);
            ss = weights[cur].size();
            for (int c = 0; c < ss; c++){
                tmp_num = rand() % (4 * ss);
                tmp_num -= 2 * ss;
                tmp_num /= (4 * ss);
                weights[cur][c] = tmp_num;
            }
            cur++;
        }
    }
}



double nn::der_and_step(int layer, int neuron, vector<double> &data, double res, double h){
    int cur = 0;
    for (int j = 0; j < num_of_layers; j++)
        cur += nums_of_neurons[j];
    double der_next = step(data, h, res, neuron, layer);
    return der_next;
}

void nn::init_values(vector<vector<double>> &vec){
    int size = weights[0].size();
    for (int i = 0; i < num_of_layers; i++){
        vector<double> tmp((i == 0 ? size : nums_of_neurons[i - 1] + 1));
        vec.push_back(tmp);
    }
}


double nn::count_cur_mistake(vector<double> &data, double res){
    vector<vector<double>> values;
    this->init_values(values);
    int weights_num = 0;
    for (int i = 0; i < data.size(); i++)
        values[0][i] = data[i];
    for (int i = 1; i < num_of_layers; i++){
        for (int j = 0; j < nums_of_neurons[i - 1]; j++){
            values[i][j] = funcs[i - 1]->func(scalar_mult(data, weights[weights_num]));
            weights_num++;
        }
        values[i][nums_of_neurons[i - 1]] = -1;
    }
    return funcs[num_of_layers - 1]->func(scalar_mult(values[num_of_layers - 1], weights[weights_num]), res);
}

double nn::choose_h(int count){
    if (ch_h != NULL)
        return ch_h(count);
    if (count < 100)
        return 1 / (double)(count + 100);
    if (count <= 1000)
        return 1 / (double)(200);
    if (count > 1000 && count <= 100000){
        int tmp = count / 10000 + 1;
        return 1 / (double)(100 * tmp + count - 1000);
    }
    if (count > 100000 && count < 1000000){
        int tmp = count / 100000 + 1;
        return 1 / (double)(20000 * tmp + count);
    }
    if (count > 4000000)
        return 3. / (2 * count);
    if (count > 3000000)
        return 2. / (1 * count);
    return 2 / (double)(1 * count);
}

void nn::fit(vector<double> &data, double res, double &cur_mistake, double mem_mst){
    static int count = 1;
    double cur_der = 0, h;
    vector<vector<double>> values;
    int weights_num = 0, cc = 0, mem, pred_size;
    init_values(values);
    int num = num_of_layers, cur = weights.size() - 1;
    for (int i = 0; i < data.size(); i++)
        values[0][i] = data[i];
    for (int i = 1; i < num_of_layers; i++){
        for (int j = 0; j < nums_of_neurons[i - 1]; j++){
            values[i][j] = funcs[i - 1]->func(scalar_mult(values[i - 1], weights[weights_num]));
            weights_num++;
        }
        values[i][nums_of_neurons[i - 1]] = -1;
    }
    vector<double> cur_res(cur + 1);
    for (auto &el : cur_res) el = 0;
    cur_res[cur] = funcs[num_of_layers - 1]->der(scalar_mult(values[num_of_layers - 1], weights[cur]), res);

    while (num != 0){
        mem = cc;
        for (int j = 0; j < nums_of_neurons[num - 1]; j++){
            cc = mem;
            h = choose_h(count);
            cur_der = der_and_step(num - 1, j, values[num - 1], cur_res[cur], h);
            if (num > 1){
                for (int i = 0; i < nums_of_neurons[num - 2]; i++){
                    cur_res[cur - cc + j - 1] += cur_der * weights[cur][i];
                    cc++;
                }
            }
            cc--;
            cur--;
        }
        num--;
    }
    count++;
    cur = weights.size() - 1;
    double resss = funcs[num_of_layers - 1]->func(scalar_mult(values[num_of_layers - 1], weights[cur]), res);
    cur_mistake = (1 - mem_mst) * cur_mistake + mem_mst * resss;
}


double nn::predict(vector<double> &data){
    double res;
    vector<vector<double>> values;
    init_values(values);
    int weights_num = 0;
    for (int i = 0; i < data.size(); i++)
        values[0][i] = data[i];
    for (int i = 1; i < num_of_layers; i++){
        for (int j = 0; j < nums_of_neurons[i - 1]; j++){
            values[i][j] = funcs[i - 1]->func(scalar_mult(values[i - 1], weights[weights_num]));
            weights_num++;
        }
        values[i][nums_of_neurons[i - 1]] = -1;
    }
    res = scalar_mult(values[num_of_layers - 1], weights[weights_num]);
    if (type == 1)
        return res;
    if (type == 0)
        return res > 0 ? 1 : -1;
}

double nn::predict(vector<vector<double>> &data, vector<double> &tmp){
    double res = 0;
    int size = data.size();
    for (int i = 0; i < size; i++)
        tmp[i] = predict(data[i]);
    res /= size;
    return res;
}

void nn::input_weights(ifstream &stream){
    for (int i = 0; i < weights.size(); i++){
        for (int j = 0; j < weights[i].size(); j++){
            stream >> weights[i][j];
        }
    }
}

void nn::output_weights(ofstream &stream){
    for (auto &el : weights){
        for (auto &el2 : el){
            stream << el2 << " ";
        }
        stream << endl << endl;
    }
}

void nn::delete_char(int num){
    for (int i = 0; i < nums_of_neurons[0]; i++)
        weights[i][num] = 0;
}

int nn::delete_worst_neuron(vector<vector<double>> &data, vector<double> &res, int layer, int data_size){
    int size = nums_of_neurons[layer], size2 = nums_of_neurons[layer + 1], k = 0;
    vector<double> tmp;
    double mistake = 0, min = 1, min_i, start_mistake = 0;
    for (int i = 0; i < data_size; i++)
        start_mistake += count_cur_mistake(data[i], res[i]);
    start_mistake /= data_size;
    for (int i = 0; i <= layer; i++)
        k += nums_of_neurons[i];

    for (int i = 0; i < size; i++){
        if (weights[k][i] == 0)
            continue;
        for (int j = k; j < size2 + k; j++){
            tmp.push_back(weights[j][i]);
            weights[j][i] = 0;
        }
        for (int i = 0; i < data_size; i++)
            mistake += count_cur_mistake(data[i], res[i]);
        mistake /= data_size;
        if (min > mistake - start_mistake){
            min = mistake - start_mistake;
            min_i = i;
        }
        for (int j = k; j < size2 + k; j++)
            weights[j][i] = tmp[j - k];
        tmp.clear();
        mistake = 0;
    }
    if (min > delete_neuron_lim)
        return -1;
    for (int j = k; j < size2 + k; j++)
        weights[j][min_i] = 0;
    cout << min_i << " " << min << endl;
    return min_i;
}





