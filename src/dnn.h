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
 * 1)	create neural networks of layer number < 255
 * 	and layer size < 65535 (size is arbitrarily limited to
 * 	system memory avaliability, however networks exceeding
 * 	these limits will not save or load correctly)
 * 2)	initialize networks
 * 3)	train the networks on examples provided by input and
 * 	output float arrays
 * 4)	feed forward inputs to return network guesses
 * 5)	save and load networks to and from horrifically custom
 * 	file formats (any systems sharing network save files
 * 	must have the same floating point implementation; this
 * 	is likely to be true on modern systems implementing
 * 	IEEE754
 * 6)	destroy objects returned by dnn_create_ functions and
 * 	free their memory back to system
 * all functions returning pointer types will return NULL on
 * error, int functions returning 0 on success and -1 on error
 *
 * oh btw its full of bugs so have fun!
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

int dnn_destroy_net(struct dnn_net *net);
/* frees memory owned by net, do not attempt to use net after calling this on it */
int dnn_destroy_train(struct dnn_train *train);
/* frees memory owned by train, ^^ */

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* DNN */
