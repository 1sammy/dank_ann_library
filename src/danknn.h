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

#ifndef DNN
#define DNN

#ifdef __cplusplus
extern "C"
{
#ifndef __cplusplus
} /* fool autoindent */
#endif
#endif

/*
 * all functions exposed by this header are prefixed with dnn_
 * they provide functionality to:
 *
 * 1)	create fully-connected feedforward neural networks
 * 2)	initialize the networks
 * 3)	train and test the networks
 * 4)	save and load networks from files
 * 5)	connect two or more networks together
 *
 * functions returning pointer types return NULL on failure
 * functions returning int return -1 on failure
 *
 */

	/* dnn_type creation */

struct dnn_net *dnn_create_network(int num_lays, int *lay_sizes);
/* dnn_create_network() returns an uninitialized network handle
 * with number of layers num_lays and each layer size given by a
 * repsective entry in lay_sizes[num_lays] */
struct dnn_train *dnn_create_train(struct dnn_net *net);
/* dnn_create_train() returns a training object handle which must be
 * passed to the training functions, and may only be used for training
 * the dnn_net for which it was created */

int dnn_init_net(struct dnn_net *net);
/* dnn_init_net() randomly initializes all weights and biases in the
 * network net using xavier initialization, all biases = 0 and all
 * weights normally distributed on the range [-1/sqrt(n), 1/sqrt(n)],
 * n == number of nodes in the weight's layer */

	/* set internal function pointers */

int dnn_set_act_func(struct dnn_net *net, int lay_num, 
		float (*actv_func)(float x));
/* dnn_set_act_func() sets the activation function of layer lay_num in network
 * net to the function pointed to by actv_func this cannot be set for the
 * input layer (lay_num == 0), and when training networks, a derivitave
 * function should be supplied and set with dnn_set_d_act_func() in order for
 * training results to be non-garbage 
 * swish is used by default if this is not called for each layer in a network */
int dnn_set_d_act_func(struct dnn_train *train, int lay_num,
		float (*d_actv_func)(float x));
/* dnn_set_d_act_func() sets the derivitave activation function for a layer in 
 * train, similarily to dnn_set_act_func()
 * training objects default to d/dx(swish(x)) */
int dnn_set_d_cost_func(struct dnn_train *train,
		float (*d_cost_func)(float out, float want));
/* sets the derivitave cost function (target essential optimination) for a
 * training object, where d_cost_func defines the derivitave of cost for
 * a given out and want of a training example
 * defaults to d/dx(mean_squared_error(x)) */

	/* network save/load */

int dnn_save_net(struct dnn_net *net, const char *filename);
/* dnn_save_net() saves all internal paramaters of net to the file provided
 * by filename, so that the network may be loaded in a later context or
 * new process by dnn_load_net()*/
struct dnn_net *dnn_load_net(const char *filename);
/* dnn_net() returns an initialized network read from the parameters saved to
 * filename */

	/* network training and execution */

int dnn_train(float *inp, float *want, struct dnn_train *train);
/* dnn_train takes a float input vector, a desired output vector, and overwrites
 * the internal cost gradient parameters in train with those calculated for this
 * particular training example */
int dnn_apply(struct dnn_train **train, int n_train, float train_aggr);
/* dnn_apply() takes an array of train objects, of length n_train, and updates 
 * the internal parameters of the network associated with the training objects
 * according to the average cost gradient of the training objects */
float *dnn_test(struct dnn_net *net, float *inp);
/* dnn_test() returns an output float vector for the forward pass of input vector
 * inp through network net */
float *get_input_gradient(struct dnn_train *train);
/* returns the input gradient with repsect to cost from train,
 * useful for providing the negative of this to another network that
 * fed the inputs of train's training example to play min/max games */

	/* cleanup functions */

int dnn_destroy_net(struct dnn_net *net);
/* frees memory owned by net, do not attempt to use net after calling this on it */
int dnn_destroy_train(struct dnn_train *train);
/* frees memory owned by train, ^^ */

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* DNN */
