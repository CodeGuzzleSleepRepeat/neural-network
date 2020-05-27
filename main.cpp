#include <iostream>
#include <vector>
#include "nn.h"
using namespace std;


double count_mistake(nn clf, vector<vector<double>> &data, vector<double> &answers){
    int size = data.size() * 0.75;
    double res, mistake = 0;
    for (int i = 0; i < size; i++){
       mistake += clf.count_cur_mistake(data[i], answers[i]);
    }
    mistake /= (double)size;
    return mistake;
}

int main(int argc, char *argv[])
{
    double good_mistake = 0.01;
    int max_iter = 10000000, break_point = 100;
    ifstream f("dta_no_ip.csv");
    if (!f.is_open()){
        cout << "Can`t open file" << endl;
        return 0;
    }
    vector<vector<double>> data, test_data;
    vector<double> answers;
    int flag = 0;
    double mem, res, mistake, memory_m = 1, max = 0, med = 0;
    get_data(f, data, answers, 4000);
    f.close();
    int size = data.size() * 0.75, complete_size = data.size();


    vector<double> tmp(data[0].size());
    for (int i = 0; i < complete_size - size; i++)
        test_data.push_back(tmp);
    vector<double> test_answers(complete_size - size), prediction(complete_size - size), prediction2(size);

    shuffle(data, data.size(), answers);
    scale1(data, data.size());
    scale2(answers, max, med);


    for (int i = size; i < complete_size; i++){
        for (int j = 0; j < data[0].size(); j++)
            test_data[i - size][j] = data[i][j];
        test_answers[i - size] = answers[i];
    }
    nn regr(2, {60, 1}, {"sigmoid", "sq"}, 1, 0, 0.0005); //parameter 4 - 1 for regr, 0 for clf
                                                         //parameter 5 - regularization coef
                                                       //parameter 6 - loosing param (put -1 to turn loosing off
                                                        //parameter 7 - function choosing anti-gradient step
                                                        //(depends on int - num of step)
                                                        //activation functions - "sigm", "relu"
                                                        //error funcs - "sq" for min squares, sigm_err for
                                                        //exp(-x) (for classification).
                                                        //further details coming soon
    regr.init_weights(data[0].size());
    mem = 1 / (double)size;
    mistake = count_mistake(regr, data, answers);
    cout << mistake << endl;
    int i = 0;
    double m = 1;
    while (mistake > good_mistake){
        int r = rand() % size;
        regr.fit(data[r], answers[r], mistake, mem);
        if (mistake >= memory_m)
            flag++;
        else flag = 0;
        memory_m = mistake;
        if (flag == break_point)
            break;
        if (i % 10000 == 0){
            m = count_mistake(regr, data, answers);
            cout << i << " " << m << " " << mistake << endl;
        }
        i++;
        if (i > max_iter)
            break;
    }
    m = count_mistake(regr, data, answers);
    cout << m << endl;


    cout << "Mistake on test_data " << count_mistake(regr, test_data, test_answers) << endl;
    cout << "Starting loosing" << endl;
    while (regr.delete_worst_neuron(data, answers, 0, size) >= 0);
    cout << "Mistake after loosing " << count_mistake(regr, test_data, test_answers) << endl;
    regr.predict(test_data, prediction);
    scale_back_ans(test_answers, max, med);
    scale_back_ans(prediction, max, med);
    double result = 0;
    for (int i = 0; i < complete_size - size; i++)
        result += fabs(prediction[i] - test_answers[i]) / test_answers[i];
    cout << "Result " << result / (complete_size - size) << endl;
    ofstream g("file.txt");
    for (int i = 0; i < 1000; i++)
        g << prediction[i] << " " << test_answers[i] << endl;
    g.close();

    ofstream h("weights.txt");
    regr.output_weights(h);
    return 0;
}
