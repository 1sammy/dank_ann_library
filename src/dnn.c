/* sam's Dank Neural Network library (libdnn)
 *
 * Copyright Sam Popham 2020
 *
 * this file is part of libdnn
 *
 *  libdnn is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <string.h>

#include "dnn_intern.h"

float dnn_act_sigmoid(float x)
{
	return pow(1 + exp(-1 * x), -1);
}

/* swish by default cuz its dank: */
float dnn_act_swish(float x)
{
	return x * dnn_act_sigmoid(x);
}

float dnn_d_act_swish(float x)
{
	return dnn_act_swish(x) + dnn_act_sigmoid(x) * (1 - dnn_act_swish(x));
}

struct dnn_net *dnn_create_network(int num_lays, int *lay_sizes)
{
	struct dnn_net *net;
	int i, j;

	if(!lay_sizes)
		return NULL;
	if(num_lays < 2)
		return NULL;

	net = malloc(sizeof *net);
	net->lays = malloc(sizeof *net->lays * (num_lays - 1));
	/* no need to allocate biases or weights for input "layer" */

	net->num_lays = num_lays;
	net->lay_sizes = malloc(sizeof *net->lay_sizes * num_lays);
	for(i = 0; i < num_lays; ++i)
		net->lay_sizes[i] = lay_sizes[i];

	for(i = 0; i < num_lays - 1; ++i){
		net->lays[i].wm_alloc_handle = malloc(sizeof *net->lays[i].wm_alloc_handle * lay_sizes[i] * lay_sizes[i + 1]);
		net->lays[i].wm = malloc(sizeof *net->lays[i].wm * lay_sizes[i + 1]);
		for(j = 0; j < lay_sizes[i + 1]; ++j)
			net->lays[i].wm[j] = &net->lays[i].wm_alloc_handle[lay_sizes[i] * j];
		net->lays[i].bias = malloc(sizeof *net->lays[i].bias * lay_sizes[i + 1]);
		net->lays[i].actv_func = &dnn_act_swish;
	}

	return net;
}

int dnn_set_act_func(struct dnn_net *net, int lay_num, float (*actv_func)(float x))
{
	if(!net)
		return -1;
	if(lay_num <= 0 || lay_num >= net->num_lays)
		return -1;

	/* input layer is thoeretical for forward propegation,
	 * rlly just the rowsize of internal layer 1, which is
	 * stored in net->lays[0] */
	net->lays[lay_num - 1].actv_func = actv_func;

	return 0;
}

int dnn_set_d_act_func(struct dnn_train *train, int lay_num,
		float (*d_actv_func)(float x))
{
	if(!train)
		return -1;
	if(lay_num < 0 || lay_num >= train->net->num_lays)
		return -1;

	/* input layer actually exists here because we want to
	 * save its activation gradient */
	train->d_lays[lay_num].d_actv_func = d_actv_func;

	/* should be saved to net save files...
	 * for now leave the responsibilty of replicating this
	 * part of state to the user */
	return 0;
}

int dnn_destroy_net(struct dnn_net *net)
{
	int i;

	for(i = 0; i < net->num_lays - 1; ++i){
		free(net->lays[i].bias);
		free(net->lays[i].wm);
		free(net->lays[i].wm_alloc_handle);
	}

	free(net->lay_sizes);
	free(net->lays);
	free(net);

	return 0;
}

/* simple d(cost)/d(output), cost == mse */
float dnn_d_cost_mse(float out, float want)
{
	return 2 * (out - want);
}

struct dnn_train *dnn_create_train(struct dnn_net *net)
{
	int i, j;
	struct dnn_train *train;

	train = malloc(sizeof *train);
	train->net = net;

	train->d_lays = malloc(sizeof *train->d_lays * net->num_lays);

	train->d_lays[0].act = malloc(sizeof *train->d_lays[0].act
			* net->lay_sizes[0]);
	train->d_lays[0].d_act = malloc(sizeof *train->d_lays[0].d_act
			* net->lay_sizes[0]);
	for(i = 1; i < net->num_lays; ++i){
		train->d_lays[i].d_bias = malloc(sizeof *train->d_lays[i].d_bias *
				net->lay_sizes[i]);
		train->d_lays[i].wtd_sum = malloc(sizeof *train->d_lays[i].wtd_sum *
				net->lay_sizes[i]);
		train->d_lays[i].d_wtd_sum = malloc(sizeof *train->d_lays[i].d_wtd_sum *
				net->lay_sizes[i]);
		train->d_lays[i].act = malloc(sizeof *train->d_lays[i].act *
				net->lay_sizes[i]);
		train->d_lays[i].d_act = malloc(sizeof *train->d_lays[i].d_act *
				net->lay_sizes[i]);
		train->d_lays[i].d_wm = malloc(sizeof *train->d_lays[i].d_wm *
				net->lay_sizes[i]);
		train->d_lays[i].d_wm_alloc_handle = malloc(sizeof
				*train->d_lays[i].d_wm_alloc_handle *
				net->lay_sizes[i - 1] * net->lay_sizes[i]);
		for(j = 0; j < net->lay_sizes[i]; ++j)
			train->d_lays[i].d_wm[j] = &train->d_lays[i].d_wm_alloc_handle[j * net->lay_sizes[i - 1]];
		train->d_lays[i].d_actv_func = &dnn_d_act_swish;
	}

	train->d_cost = &dnn_d_cost_mse;

	return train;
}

int dnn_set_d_cost_func(struct dnn_train *train,
		float (*d_cost_func)(float out, float want))
{
	if(!train || !d_cost_func)
		return -1;
	train->d_cost = d_cost_func;
	return 0;
}

int dnn_destroy_train(struct dnn_train *train)
{
	int i;

	free(train->d_lays[0].act);
	free(train->d_lays[0].d_act);
	for(i = 1; i < train->net->num_lays; ++i){
		free(train->d_lays[i].d_bias);
		free(train->d_lays[i].wtd_sum);
		free(train->d_lays[i].d_wtd_sum);
		free(train->d_lays[i].act);
		free(train->d_lays[i].d_act);
		free(train->d_lays[i].d_wm);
		free(train->d_lays[i].d_wm_alloc_handle);
	}

	free(train->d_lays);
	free(train);

	return 0;
}

float normal_probability(float x)
{
	return pow(sqrt(2 * M_PI), -1) * exp(-1/2 * pow(x, 2));
}

float *xavier_data(int n_cols, int n_rows)
{
	int i;
	float rand_n;
	float *data;

	/* calloc() to keep clang-tidy happy */
	data = calloc(sizeof *data, n_cols * n_rows);
	if(!data)
		return NULL;

	srand(time(NULL));

	for(i = 0; i < n_cols * n_rows; ++i){
		/* generate a random float [0,1) */
		rand_n = rand() / (RAND_MAX + 1.0);
		/* compare its normal occurance probability to a random probability */
		while(normal_probability(rand() / (RAND_MAX + 1.0)) > normal_probability(rand_n))
			rand_n = rand() / (RAND_MAX + 1.0);
		if(rand() % 2 == 0){
			/* target dataset:
			 * data[i] ~ U[-1/sqrt(n), 1/sqrt(n)],
			 * n = row_size(wm) */
			rand_n *= -1;
		}
		rand_n /= sqrt(n_cols);
		data[i] = rand_n;
	}

	return data;
}

/* Xavier initialization, cuz its dank */
int dnn_init_net(struct dnn_net *net)
{
	int i, j, k;
	float *xavier_wts;

	if(!net)
		return -1;

	for(i = 0; i < net->num_lays - 1; ++i){
		xavier_wts = xavier_data(net->lay_sizes[i], net->lay_sizes[i + 1]);
		if(!xavier_wts)
			return -1;

		for(j = 0; j < net->lay_sizes[i + 1]; ++j){
			net->lays[i].bias[j] = 0;

			for(k = 0; k < net->lay_sizes[i]; ++k)
				net->lays[i].wm[j][k] = xavier_wts[j * net->lay_sizes[i] + k];
		}
		free(xavier_wts);
	}

	return 0;
}

int dnn_save_net(struct dnn_net *net, const char *filename)
{
	int i, j;
	FILE *fp;
	char *data_access;
	static float float_magicnum = 9;

	fp = fopen(filename, "w");
	if(!fp)
		return -1;

	/* float_magicnum, validates savefile compatibility
	 * plus can encode shit in it too version information etc */
	for(i = 0; i < (int)sizeof float_magicnum; ++i)
		fputc(((char *)&float_magicnum)[i], fp);

	/* number of layers (must be < 256layers / net) */
	fputc((char)net->num_lays, fp);

	/* layer_sizes (2 bytes so < 65536nodes / layer) */
	for(i = 0; i < net->num_lays; ++i)
		for(j = 0; j < 2; ++j)
			fputc((char)net->lay_sizes[i] >> (8 * j), fp);

	/* write parameters, enough informantion to decode and load this
	 * data is now stored in the header bytes */
	for(i = 0; i < net->num_lays - 1; ++i){
		data_access = (char *)net->lays[i].bias;
		for(j = 0; j < net->lay_sizes[i + 1] * (int)sizeof *net->lays[i].bias; ++j)
			fputc(data_access[j], fp);

		data_access = (char *)net->lays[i].wm_alloc_handle;
		for(j = 0; j < net->lay_sizes[i] * net->lay_sizes[i + 1] * (int)sizeof *net->lays[i].wm_alloc_handle; ++j)
			fputc(data_access[j], fp);
	}

	fputc(EOF, fp);
	fclose(fp);

	return 0;
}

struct dnn_net *dnn_load_net(const char *filename)
{
	int i, j;
	int num_lays;
	float float_magicnum;
	int *lay_sizes;
	char *data_access;
	struct dnn_net *net;
	FILE *fp;

	fp = fopen(filename, "r");
	if(!fp)
		return NULL;

	/* chock float_magicnum to validate network save file
	 * compatibility */
	for(i = 0; i < (int)sizeof float_magicnum; ++i)
		((char *)&float_magicnum)[i] = fgetc(fp);
	if(float_magicnum != 9)
		return NULL;

	num_lays = fgetc(fp);

	lay_sizes = malloc(sizeof *lay_sizes * num_lays);
	for(i = 0; i < num_lays; ++i){
		lay_sizes[i] = 0;
		for(j = 0; j < 2; ++j)
			lay_sizes[i] |= fgetc(fp) << (8 * j);
	}

	net = dnn_create_network(num_lays, lay_sizes);

	for(i = 0; i < num_lays - 1; ++i){
		data_access = (char *)net->lays[i].bias;
		for(j = 0; j < lay_sizes[i + 1] * (int)sizeof float_magicnum; ++j)
			data_access[j] = fgetc(fp);

		data_access = (char *)net->lays[i].wm_alloc_handle;
		for(j = 0; j < lay_sizes[i] * lay_sizes[i + 1] * (int)sizeof float_magicnum; ++j)
			data_access[j] = fgetc(fp);
	}

	free(lay_sizes);
	fclose(fp);

	return net;
}

int dnn_train(float *inp, float *want, struct dnn_train *train)
{
	int i, j, k;

	for(i = 0; i < train->net->lay_sizes[0]; ++i)
		train->d_lays[0].act[i] = inp[i];

	/* forward pass, saving useful parameters */
	for(i = 1; i < train->net->num_lays; ++i){
		for(j = 0; j < train->net->lay_sizes[i]; ++j){
			train->d_lays[i].wtd_sum[j] = 0;
			for(k = 0; k < train->net->lay_sizes[i - 1]; ++k)
				train->d_lays[i].wtd_sum[j] +=
					train->net->lays[i - 1].wm[j][k] *
					train->d_lays[i - 1].act[k];
			train->d_lays[i].wtd_sum[j] += train->net->lays[i - 1].bias[j];
			train->d_lays[i].act[j] = train->net->lays[i - 1].actv_func(train->d_lays[i].wtd_sum[j]);
		}
	}
	for(i = 0; i < train->net->lay_sizes[train->net->num_lays - 1]; ++i)
		train->d_lays[train->net->num_lays - 1].d_act[i] = train->d_cost(train->d_lays[train->net->num_lays - 1].act[i], want[i]);
	/* backpropegationnnnnnnnn baby */
	for(i = train->net->num_lays - 1; i > 0; --i){
		for(j = 0; j < train->net->lay_sizes[i]; ++j){
			train->d_lays[i].d_wtd_sum[j] = train->d_lays[i].d_actv_func(train->d_lays[i].wtd_sum[j]) * train->d_lays[i].d_act[j];
			train->d_lays[i].d_bias[j] = train->d_lays[i].d_wtd_sum[j];
		}
		for(j = 0; j < train->net->lay_sizes[i]; ++j)
			for(k = 0; k < train->net->lay_sizes[i - 1]; ++k)
				train->d_lays[i].d_wm[j][k] = train->d_lays[i].d_wtd_sum[j] * train->d_lays[i - 1].act[k];
		for(k = 0; k < train->net->lay_sizes[i - 1]; ++k)
			for(j = 0; j < train->net->lay_sizes[i]; ++j)
				train->d_lays[i - 1].d_act[k] = train->d_lays[i].d_wtd_sum[j] * train->net->lays[i - 1].wm[j][k];
	}
	return 0;
}

float *get_input_gradient(struct dnn_train *train)
{
	int i;
	float *inp_grad;

	inp_grad = malloc(sizeof *inp_grad * train->net->lay_sizes[0]);
	for(i = 0; i < train->net->lay_sizes[0]; ++i)
		inp_grad[i] = train->d_lays[0].d_act[i];

	return inp_grad;
}

int dnn_apply(struct dnn_train **train, int n_train, float train_aggr)
{
	int i, j, k, l;

	for(i = 0; i < n_train; ++i){
		for(j = 1; j < train[i]->net->num_lays; ++j){
			for(k = 0; k < train[i]->net->lay_sizes[j]; ++k){
				for(l = 0; l < train[i]->net->lay_sizes[j - 1]; ++l)
					train[i]->net->lays[j - 1].wm[k][l] += -1 * train_aggr / (float)n_train * train[i]->d_lays[j].d_wm[k][l];
				train[i]->net->lays[j - 1].bias[k] += -1 * train_aggr / (float)n_train * train[i]->d_lays[j].d_bias[k];
			}
		}
	}

	return 0;
}

float *dnn_test(struct dnn_net *net, float *inp)
{
	int i, j, k;
	float *output;
	float **act;
	float *acts;
	int sum_lay_sizes;

	if(!net || !inp)
		return NULL;

	/* allocate temp activation storage */
	sum_lay_sizes = 0;
	for(i = 0; i < net->num_lays; ++i)
		sum_lay_sizes += net->lay_sizes[i];

	/* make clang-tidy happy */
	if(sum_lay_sizes == 0)
		return NULL;

	act = malloc(sizeof *act * net->num_lays);
	acts = malloc(sizeof *acts * sum_lay_sizes);
	output = malloc(sizeof *output * net->lay_sizes[net->num_lays - 1]);

	sum_lay_sizes = 0;
	for(i = 0; i < net->num_lays; ++i){
		act[i] = &acts[sum_lay_sizes];
		sum_lay_sizes += net->lay_sizes[i];
	}

	for(i = 0; i < net->lay_sizes[0]; ++i)
		act[0][i] = inp[i];

	/* its alive! */
	for(i = 0; i < net->num_lays - 1; ++i){
		for(j = 0; j < net->lay_sizes[i + 1]; ++j){
			act[i + 1][j] = 0;
			for(k = 0; k < net->lay_sizes[i]; ++k)
				act[i + 1][j] += net->lays[i].wm[j][k] * act[i][k];
			act[i + 1][j] = net->lays[i].actv_func(act[i + 1][j] +
					net->lays[i].bias[j]);
		}
	}

	for(i = 0; i < net->lay_sizes[net->num_lays - 1]; ++i)
		output[i] = act[net->num_lays - 1][i];

	free(act);
	free(acts);

	return output;
}
