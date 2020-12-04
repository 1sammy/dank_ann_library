/* an example program using libdnn (sam's Dank Neural Network library)
 * to train a network to recognize handwritten digits in the MNIST
 * database 
 *
 * Copyright Sam Popham, 2020
 *
 *  This program is free software: you can redistribute it and/or modify
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
#include <limits.h>
#include <time.h>

#include "../dnn.h"

#define TRAIN_DATA	"MNIST/train-images-idx3-ubyte"
#define TRAIN_LABEL	"MNIST/train-labels-idx1-ubyte"

#define TEST_DATA	"MNIST/t10k-images-idx3-ubyte"
#define TEST_LABEL	"MNIST/t10k-labels-idx1-ubyte"

struct dataset
{
	// metadata
	unsigned char data_mn[4], label_mn[4];
	int *data_size, *label_size;

	// data
	unsigned char *label;
	float *data_alloc_handle;
	float **data;
};

struct dataset *load_dataset(const char *datafile, const char *labelfile)
{
	int i, a;
	int rand_sel;
	struct dataset *ds;
	FILE *data_fp, *label_fp;
	
	ds = malloc(sizeof *ds);

	// load MNIST training data

	// "the IDX file format is a simple format for vectors and
	// multidimensional matrices of various numerical types."
	data_fp = fopen(datafile, "r");
	label_fp = fopen(labelfile, "r");
	if(!data_fp || !label_fp){
		perror("data/labelfile");
		return NULL;
	}
	// "The magic number is an integer (MSB first)."
	for(i = 0; i < 4; ++i){
		ds->data_mn[i] = fgetc(data_fp);
		ds->label_mn[i] = fgetc(label_fp);
	}

	// "The first 2 bytes are always 0."
	for(i = 0; i < 2; ++i){
		if(ds->data_mn[i] || ds->label_mn[i]){
			puts("file format != IDX");
			return NULL;
		}
	}
	// "The third byte codes the type of the data:"
	if(ds->data_mn[2] != 8 || ds->label_mn[2] != 8){
		puts("stored data not ubyte");
		return NULL;
	}
	// "The 4-th byte codes the number of dimensions of the
	// vector/matrix:"
	if(ds->label_mn[3] != 1){
		puts("unexpected data dimensions");
		return NULL;
	}
	// although this is about halfway to working with
	// an arbitrary number of data/label dimensions
	ds->data_size = malloc(sizeof *ds->data_size * ds->data_mn[3]);
	ds->label_size = malloc(sizeof *ds->label_size * ds->label_mn[3]);

	// read dimension sizes

	// "The sizes in each dimension are 4-byte integers (MSB
	// first, high endian, like in most non-Intel processors)."
	for(i = 0; i < ds->data_mn[3]; ++i){
		ds->data_size[i] = 0;
		for(a = 0; a < 4; ++a)
			ds->data_size[i] |= fgetc(data_fp) << (24 - 8*a);
	}
	for(i = 0; i < ds->label_mn[3]; ++i){
		ds->label_size[i] = 0;
		for(a = 0; a < 4; ++a)
			ds->label_size[i] |= fgetc(label_fp) << (24 - 8*a);
	}

	// sizes in dimension 0 should be equal
	if(ds->data_size[0] != ds->label_size[0]){
		puts("unequal number of data to labels");
		return NULL;
	}

	// flatten remaining dimensions
	for(i = 2; i < ds->data_mn[3]; ++i)
		ds->data_size[1] *= ds->data_size[i];

	// now ready to allocate memory for the data
	ds->label = malloc(sizeof(*ds->label) * ds->label_size[0]);
	// allocating contigous memory for multidim array of data
	ds->data_alloc_handle = malloc(sizeof(*ds->data_alloc_handle) *
			ds->data_size[0] *
			ds->data_size[1]);
	ds->data = malloc(sizeof(*ds->data) *
			ds->data_size[0]);

	for(i = 0; i < ds->data_size[0]; ++i)
		ds->data[i] = &ds->data_alloc_handle[i * ds->data_size[1]];

	// read data into memory
	for(i = 0; i < ds->label_size[0]; ++i)
		ds->label[i] = fgetc(label_fp);
	for(i = 0; i < ds->data_size[0]; ++i)
		for(a = 0; a < ds->data_size[1]; ++a)
				ds->data[i][a] = (float)fgetc(data_fp) / UCHAR_MAX;
	// / UCHAR_MAX **

	fclose(data_fp);
	fclose(label_fp);

	// print a random image from the dataset
	// for fun and verification
	srand(time(NULL));
	rand_sel = rand() % ds->data_size[0];

	for(i = 0; i < 28 * 28; ++i){
		if(ds->data[rand_sel][i] > 0.5)
			putchar('X');
		else
			putchar(' ');
		if(i % 28 == 0)
			putchar('\n');
	}
	printf("label: %d\n", ds->label[rand_sel]);

	return ds;
}

int destroy_dataset(struct dataset *ds)
{
	free(ds->label);
	free(ds->data_alloc_handle);
	free(ds->data);
	free(ds->data_size);
	free(ds->label_size);
	free(ds);

	return 0;
}

int main()
{
	int i, a;
	int rand_sel;

	struct dataset *train_data, *test_data;

	struct dnn_net *net;
	struct dnn_train *train;
	float *output;
	float want[10];

	int guess;
	float max_output;
	float accuracy;

	static int some_sizes[4] = {28 * 28, 16, 16, 10};

	train_data = load_dataset(TRAIN_DATA, TRAIN_LABEL);

	net = dnn_create_network(4, some_sizes);
	dnn_init_net(net);

	train = dnn_create_train(net);

	printf("sweeping over the training database once...\n");
	for(i = 0; i < train_data->data_size[0]; ++i){
		for(a = 0; a < 10; ++a)
			want[a] = 0;
		want[train_data->label[i]] = 1;
		dnn_train(train_data->data[i], want, train);
		dnn_apply(&train, 1, 0.03);
	}

	destroy_dataset(train_data);

	test_data = load_dataset(TEST_DATA, TEST_LABEL);

	guess = 0;
	accuracy = 0;
	printf("testing on the testing database...\n");
	for(i = 0; i < test_data->data_size[0]; ++i){
		rand_sel = rand() % test_data->data_size[0];
		output = dnn_test(net, test_data->data[rand_sel]);
		for(a = 0; a < 10; ++a)
			printf("%1.3f ", output[a]);
		putchar('\n');
		printf("labelled value: %i\n", test_data->label[rand_sel]);

		// determining accuracy
		max_output = 0;
		for(a = 0; a < 10; ++a){
			if(output[a] > max_output){
				max_output = output[a];
				guess = a;
			}
		}
		printf("network_guess: %i\n", guess);
		if(guess == test_data->label[rand_sel])
			accuracy += 1.0/test_data->data_size[0];

		free(output);
	}

	destroy_dataset(test_data);

	printf("accuracy: %f\n", accuracy);
	dnn_save_net(net, "dank.net");

	dnn_destroy_train(train);
	dnn_destroy_net(net);

	return 0;
}
