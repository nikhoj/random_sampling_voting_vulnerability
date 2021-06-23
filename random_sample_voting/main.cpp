#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <tuple>
#include <algorithm>
#include <map>
#include "gurobi_c++.h"
#include <sstream>
#include <omp.h>
using namespace std;

const double calc_eps = 1e-2;

template <typename T>

std::string to_string_with_precision(const T a_value, const int n = 6)
{
	std::ostringstream out;
	out.precision(n);
	out << std::fixed << a_value;
	return out.str();
}


double solve(int ell, int n, int m, int B, vector<vector<int>> tau, vector<vector<double>> pdf, vector<vector<double>> cdf, GRBEnv env) {

	GRBVar** x = new GRBVar * [m - 1];
	GRBVar* z = new GRBVar[n + 1];
	GRBVar* y = new GRBVar[n + 1];
	double* coefficient = new double[n + 1];

	for (int i = 0; i < m - 1; i++)
		x[i] = new GRBVar[tau[i].size()];


	try {


		GRBModel model = GRBModel(env);
		model.set(GRB_IntParam_Presolve, 0);
		model.set(GRB_IntParam_OutputFlag, 0);

		for (int i = 0; i < m - 1; i++)
			for (int k = 0; k < tau[i].size(); k++)
				x[i][k] = model.addVar(0, 1, 0, GRB_INTEGER);

		for (int t = 0; t < n + 1; t++) {
			z[t] = model.addVar(-1e3, 0, 0, GRB_CONTINUOUS);
			y[t] = model.addVar(0, 1, 0, GRB_CONTINUOUS);
		}

		GRBLinExpr obj = 0;

		for (int t = 0; t < n + 1; t++) {
			obj += pdf[ell][t] * y[t];
		}



		model.setObjective(obj, GRB_MAXIMIZE);

		string s1 = "FuncPieces=-1";

		//string s2 = "FuncPieceError=0.000001";
		string s2 = "FuncPieceError=" + to_string_with_precision(calc_eps, 12);

		string s3 = "FuncPieceRatio=-1";

		string s = s1 + " " + s2 + " " + s3;

		for (int t = 0; t < n + 1; t++)
			model.addGenConstrExp(z[t], y[t], "", s);

		GRBLinExpr exp = 0;
		for (int i = 0; i < m - 1; i++)
			for (int k = 0; k < tau[i].size(); k++)
				exp += tau[i][tau[i].size() - 1 - k] * x[i][k];
		model.addConstr(exp <= B);


		exp = 0;
		for (int i = 0; i < m - 1; i++)
			for (int k = 0; k < tau[i].size(); k++)
				exp += k * x[i][k];
		model.addConstr(exp == n - ell);



		for (int t = 0; t < n + 1; t++) {
			/*
			GRBLinExpr exp = 0;
			for (int i = 0; i < m - 1; i++)
				for (int k = 0; k < tau[i].size(); k++)
					exp += log(cdf[k][t])* x[i][k];
			model.addConstr(exp == z[t]);
			*/


			GRBLinExpr exp = 0;
			for (int i = 0; i < m - 1; i++) {
				for (int k = 0; k < tau[i].size(); k++)
					coefficient[k] = log(cdf[k][t]);
				exp.addTerms(coefficient, x[i], int(tau[i].size()));
			}
			model.addConstr(exp == z[t]);

		}




		for (int i = 0; i < m - 1; i++) {
			GRBLinExpr exp = 0;
			for (int k = 0; k < tau[i].size(); k++)
				exp += x[i][k];
			model.addConstr(exp == 1);
		}


		model.optimize();



		if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {

			return  model.get(GRB_DoubleAttr_ObjVal);

		}

	}
	catch (GRBException e) {
		cout << "Error code = " << e.getErrorCode() << endl;
		cout << e.getMessage() << endl;
	}
	catch (...) {
		cout << "Exception during optimization" << endl;
	}

	delete[] coefficient;
	delete[] y;
	delete[] z;
	for (int i = 0; i < m - 1; i++)
		delete[] x[i];
	delete[] x;

	return -1;
	////// return;
}

double solve_omp(int n, int m, int B, vector<vector<int>> c, vector<vector<int>> tau, vector<vector<double>> pdf, vector<vector<double>> cdf) {
	vector<double> ans(n + 1);

	for (int j = 0; j <= n; j++)
		ans[j] = -1;


	GRBEnv env = GRBEnv();


	/*
	for (int ell = c[m - 1].size(); ell <= n; ell++) {
		ans[ell] = solve(ell, n, m, B, tau, pdf, cdf, env);
		cout << ell << " " << ans[ell] << endl;
	}
	*/

	int d = 4;

	for (int ell = int(c[m - 1].size()); ell <= n; ell += d) {
		int ell_s = ell;
		int ell_t = min(n, ell + d - 1);

		//omp_set_num_threads(1);
#pragma omp parallel for
		for (int j = ell_s; j <= ell_t; j++) {
			ans[j] = solve(j, n, m, B, tau, pdf, cdf, env);
			cout << j << " " << ans[j] << endl;
		}

		cout << "-------------------" << endl;

		bool end = false;
		for (int j = ell_s; j <= ell_t; j++) {
			if (ans[j] < 0) end = true;
			if (ans[j] >= 1.0 - calc_eps)  end = true;
		}

		if (end)
			break;

	}

	double max = -1;

	for (int ell = int(c[m - 1].size()); ell <= n; ell++)
		if (max < ans[ell])  max = ans[ell];

	return max;
}


tuple< int, int, double, int, vector<vector<int>> > load_data(string file_name) {
	ifstream fin(file_name);

	int n, m, B;
	double p;

	int des_candidate;

	fin >> n >> m >> p >> B;

	fin >> des_candidate;


	vector<vector<int>> c(m);



	for (int i = 0; i < m; i++)
		c[i].clear();

	for (int i = 0; i < n; i++) {
		int v, tmp;
		fin >> v >> tmp;

		////////swap desigante candidate with m-1///////////////////

		if (v == des_candidate)
			v = m - 1;
		else {
			if (v == m - 1)
				v = des_candidate;
		}

		////////////////////////////////////////////////////////////

		c[v].push_back(tmp);
	}

	for (int i = 0; i < m; i++)
		sort(c[i].begin(), c[i].end());

	fin.close();

	return make_tuple(n, m, p, B, c);
}

double calc(int k, int t, double p) {

	double ret = 1.0;

	for (int i = 1; i <= k - t; i++)
		ret *= (t + i) * (1 - p) / i;

	for (int i = 1; i <= t; i++)
		ret *= p;

	return ret;
}

tuple< vector<vector<int>>, vector<vector<double>>, vector<vector<double>>  > init(int n, int m, double p, vector<vector<int>> c) {

	vector<vector<int>> tau(m);
	vector<vector<double>> pdf(n + 1);
	vector<vector<double>> cdf(n + 1);

	for (int i = 0; i < m; i++) {
		tau[i] = vector<int>(c[i].size() + 1);

		tau[i][0] = 0;
		for (int j = 1; j <= c[i].size(); j++)
			tau[i][j] = tau[i][j - 1] + c[i][j - 1];
	}

	for (int i = 0; i < n + 1; i++) {
		pdf[i] = vector<double>(n + 1);
		//	double sum = 0;
		for (int j = 0; j < i + 1; j++) {
			//pdf[i][j] = calc(i, j, p);
			if (j == 0)
				pdf[i][j] = pow(1 - p, i);
			else
				pdf[i][j] = pdf[i][j - 1] * p * (i - j + 1) / ((1 - p) * j);

			//cout << pdf[i][j] << " " << calc(i, j, p) << endl;
			//sum += pdf[i][j];
		}

		for (int j = i + 1; j < n + 1; j++)
			pdf[i][j] = 0;
	}


	for (int i = 0; i < n + 1; i++) {
		cdf[i] = vector<double>(n + 1);
		for (int j = 0; j < n + 1; j++)
			if (j == 0)
				cdf[i][j] = pdf[i][j];
			else
				cdf[i][j] = cdf[i][j - 1] + pdf[i][j];
	}

	return make_tuple(tau, pdf, cdf);

}

void write_ans(string file_name, int index, double opt, double t) {

	ofstream fout(file_name, ios::app);
	fout << setiosflags(ios::fixed);
	fout << setprecision(10);

	fout << index << " " << opt << " " << t << " ";

	fout << endl;

	fout.close();
}


int main(int argc, char* argv[]) {


	string file_prefix = "test";
	int s_index = 0;
	int tot = 1;


	if (argc == 4) {
		file_prefix = argv[1];
		s_index = atoi(argv[2]);
		tot = atoi(argv[3]);
	}


	for (int i = s_index; i < tot; i++) {

		vector<vector<int>> tau;
		vector<vector<double>> pdf;
		vector<vector<double>> cdf;

		int n, m, B;
		double p;
		vector<vector<int>> c;

		cout << file_prefix + "//" + to_string(i) + ".txt" << endl;

		tuple< int, int, double, int, vector<vector<int>> > input_data = load_data(file_prefix + "//" + to_string(i) + ".txt");

		n = get<0>(input_data);
		m = get<1>(input_data);
		p = get<2>(input_data);
		B = get<3>(input_data);
		c = get<4>(input_data);

		cout << n << " " << m << " " << p << " " << B << endl;


		tuple< vector<vector<int>>, vector<vector<double>>, vector<vector<double>>  > data = init(n, m, p, c);

		//cout << "----------" << endl;

		tau = get<0>(data);
		pdf = get<1>(data);
		cdf = get<2>(data);


		//cout << tau[0].size() << " " << tau[1].size() << " " << tau[2].size() << endl;



		size_t t0 = clock();

		double ans = solve_omp(n, m, B, c, tau, pdf, cdf);

		write_ans(file_prefix + "//opt_" + to_string_with_precision(calc_eps, 12) + ".txt", i, ans, (clock() - t0) / 1000.0);

	}

	//system("pause");

	return 0;
}