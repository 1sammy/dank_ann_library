/* sam's Dank Neural Network library (libdanknn)
 *
 * Copyright Sam Popham 2020
 *
 * this file is part of libdanknn
 *
 *  libdanknn is free software: you can redistribute it and/or modify
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
#ifndef DNN_INTERN
#define DNN_INTERN

struct dnn_layer{
	float **wm;
	float *wm_alloc_handle;
	float *bias;
	float (*actv_func)(float inp);
};

struct dnn_d_layer{
	float **d_wm;
	float *d_wm_alloc_handle;
	float *d_bias;
	float (*d_actv_func)(float inp);

	float *wtd_sum;
	float *d_wtd_sum;

	float *act;
	float *d_act;
};

struct dnn_train{
	struct dnn_net *net;
	struct dnn_d_layer *d_lays;
	float (*d_cost)(float out, float want);
};

struct dnn_net{
	int num_lays;
	int *lay_sizes;
	struct dnn_layer *lays;
};

/* activation functions */
float dnn_act_sigmoid(float x);
float dnn_act_swish(float x);
float dnn_d_act_swish(float x);
float dnn_d_cost_mse(float out, float want);

/* dnn_type creation */
struct dnn_net *dnn_create_network(int num_lays, int *lay_sizes);
struct dnn_train *dnn_create_train(struct dnn_net *net);

/* set internal function pointers */
int dnn_set_act_func(struct dnn_net *net, int lay_num, float (*actv_func)(float x));
int dnn_set_d_act_func(struct dnn_train *train, int lay_num,
		float (*d_actv_func)(float x));
int dnn_set_d_cost_func(struct dnn_train *train,
		float (*d_cost_func)(float out, float want));

/* network initialization */
float normal_probability(float x);
float *xavier_data(int n_cols, int n_rows);
int dnn_init_net(struct dnn_net *net);

/* network save/load */
int dnn_save_net(struct dnn_net *net, const char *filename);
struct dnn_net *dnn_load_net(const char *filename);

/* network training and execution */
int dnn_train(float *inp, float *want, struct dnn_train *train);
int dnn_apply(struct dnn_train **train, int n_train, float train_aggr);

float *get_input_gradient(struct dnn_train *train);

float *dnn_test(struct dnn_net *net, float *inp);

/* cleanup functions */
int dnn_destroy_net(struct dnn_net *net);
int dnn_destroy_train(struct dnn_train *train);

#endif /* DNN_INTERN */
