#include "inference.h"

void conv2d_load_inference(int type, FILE *file, in_out *in)
{
    //print_in_out(*in);
    in_out out = {0, 0, 0, NULL};
    int conv_w = 0;
    if (!read_weight(type, &conv_w, sizeof(int), 1, file))
    {
        debug("Read weights error!");
    }
    int conv_h = 0;
    if (!read_weight(type, &conv_h, sizeof(int), 1, file))
    {
        debug("Read weights error!");
    }
    int in_c = 0;
    if (!read_weight(type, &in_c, sizeof(int), 1, file))
    {
        debug("Read weights error!");
    }
    int out_c = 0;
    if (!read_weight(type, &out_c, sizeof(int), 1, file))
    {
        debug("Read weights error!");
    }
    int stride = 0;
    if (!read_weight(type, &stride, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }
    if (!read_weight(type, &stride, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }
    int padding = 0; //{"valid":1, "same":2}
    if (!read_weight(type, &padding, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }
    int use_bias = 0; //{"valid":1, "same":2}
    if (!read_weight(type, &use_bias, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }
    debug("use_bias: %d", use_bias);
    debug("conv_w:%d, conv_h:%d, in_c:%d, out_c:%d, stride:%d, padding:%d",
          conv_w, conv_h, in_c, out_c, stride, padding);

    int pad_w = 0;
    int pad_h = 0;
    if (padding == 2) // same
    {
        out.w = in->w;
        out.h = in->h;
        out.c = out_c;
        debug("%d, %d, %d", out.w, out.h, out.c);
        out.data = calloc(out.w * out.h * out.c, sizeof(float));
        debug();
        if (conv_w % 2 == 1)
        {
            pad_w = conv_w / 2;
            pad_h = conv_h / 2;
        }
        //debug("pad:%d", pad);
    }
    else if (padding == 1) // valid
    {
        out.w = in->w;
        out.h = in->h;
        out.c = out_c;
        out.data = calloc(out.w * out.h * out.c, sizeof(float));
    }
    else
    {
        debug("Read conv2d padding error!");
        return;
    }

    float *weights = calloc(conv_w * conv_h * in_c, sizeof(float));
    float *bias = NULL;
    if (use_bias == 1)
    {
        bias = calloc(1, sizeof(float));
    }

    for (int c_out = 0; c_out < out.c; c_out++)
    {
        for (int i = 0; i < conv_w * conv_h * in_c; i++)
        {
            if (!read_weight(type, weights + i, sizeof(float), 1, file))
            {
                debug("Read conv2d weights error!");
            }
        }
        if (use_bias == 1)
        {
            if (!read_weight(type, bias, sizeof(float), 1, file))
            {
                debug("Read conv2d bias error!");
            }
        }
        /*  
        debug("weights:---------------");
        for (int i = 0; i < conv_w * conv_h * in_c; i++)
        {
            printf("%f ", weights[i]);
        }
        printf("\n");
        printf("%f\n", *bias);
        debug("weights end---------------");
*/
        for (int h = 0; h < out.h; h++)
        {
            for (int w = 0; w < out.w; w++)
            {
                out.data[c_out * out.w * out.h + h * out.w + w] = 0;
                for (int c_in = 0; c_in < in->c; c_in++)
                {
                    for (int y = 0; y < conv_h; y++)
                    {
                        int in_y = h + y - pad_h;
                        if (in_y < 0 || in_y >= in->h)
                            continue;
                        for (int x = 0; x < conv_w; x++)
                        {
                            int in_x = w + x - pad_w;
                            if (in_x < 0 || in_x >= in->w)
                                continue;
                            out.data[c_out * out.w * out.h + h * out.w + w] +=
                                in->data[c_in * in->w * in->h + in_y * in->w + in_x] *
                                weights[c_in * conv_w * conv_h + y * conv_w + x];
                        }
                    }
                }
                if (use_bias == 1)
                {
                    out.data[c_out * out.w * out.h + h * out.w + w] += bias[0];
                }
            }
        }
    }

    free(weights);
    free(bias);
    free_in_out(in);
    in->c = out.c;
    in->w = out.w;
    in->h = out.h;
    in->data = out.data;
    out.data = NULL;
    //print_in_out(*in);
}

void depthwiseconv2d_load_inference(int type, FILE *file, in_out *in)
{
    in_out out = {0, 0, 0, NULL};
    int conv_w = 0;
    if (!read_weight(type, &conv_w, sizeof(int), 1, file))
    {
        debug("Read weights error!");
    }
    int conv_h = 0;
    if (!read_weight(type, &conv_h, sizeof(int), 1, file))
    {
        debug("Read weights error!");
    }
    int in_c = 0;
    if (!read_weight(type, &in_c, sizeof(int), 1, file))
    {
        debug("Read weights error!");
    }
    int out_c = 0;
    if (!read_weight(type, &out_c, sizeof(int), 1, file))
    {
        debug("Read weights error!");
    }
    int kernel = 0;
    if (!read_weight(type, &kernel, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }
    if (!read_weight(type, &kernel, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }
    int stride = 0;
    if (!read_weight(type, &stride, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }
    if (!read_weight(type, &stride, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }
    int padding = 0; //{"valid":1, "same":2}
    if (!read_weight(type, &padding, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }
    int depth_multiplier = 0;
    if (!read_weight(type, &depth_multiplier, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }
    int use_bias = 0;
    if (!read_weight(type, &use_bias, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }
    debug("use_bias: %d", use_bias);
    debug("conv_w:%d, conv_h:%d, in_c:%d, out_c:%d, kernel:%d, stride:%d, padding:%d, depth_multiplier:%d",
          conv_w, conv_h, in_c, out_c, kernel, stride, padding, depth_multiplier);

    int pad_w = 0;
    int pad_h = 0;
    if (padding == 2) // same
    {
        out.w = in->w;
        out.h = in->h;
        out.c = in->c;
        out.data = calloc(out.w * out.h * out.c, sizeof(float));
        if (conv_w % 2 == 1)
        {
            pad_w = conv_w / 2;
            pad_h = conv_h / 2;
        }
        //debug("pad:%d", pad);
    }
    else if (padding == 1) // valid
    {
        out.w = in->w;
        out.h = in->h;
        out.c = in->c;
        out.data = calloc(out.w * out.h * out.c, sizeof(float));
    }
    else
    {
        debug("Read conv2d padding error!");
        return;
    }

    float *weights = calloc(conv_w * conv_h * in_c, sizeof(float));
    float *bias = NULL;
    if (use_bias == 1)
    {
        bias = calloc(1, sizeof(float));
    }

    for (int c_out = 0; c_out < out.c; c_out++)
    {
        for (int i = 0; i < conv_w * conv_h; i++)
        {
            if (!read_weight(type, weights + i, sizeof(float), 1, file))
            {
                debug("Read conv2d weights error!");
            }
        }
        if (use_bias == 1)
        {
            if (!read_weight(type, bias, sizeof(float), 1, file))
            {
                debug("Read conv2d bias error!");
            }
        }
        /*  
        for(int i = 0; i < conv_w*conv_h*in_c; i++)
        {
            printf("%f ", weights[i]);
        }
        printf("\n");
        printf("%f\n", *bias);
        */

        for (int h = 0; h < out.h; h++)
        {
            for (int w = 0; w < out.w; w++)
            {
                out.data[c_out * out.w * out.h + h * out.w + w] = 0;
                for (int y = 0; y < conv_h; y++)
                {
                    int in_y = h + y - pad_h;
                    if (in_y < 0 || in_y >= in->h)
                        continue;
                    for (int x = 0; x < conv_w; x++)
                    {
                        int in_x = w + x - pad_w;
                        if (in_x < 0 || in_x >= in->w)
                            continue;
                        out.data[c_out * out.w * out.h + h * out.w + w] +=
                            in->data[c_out * in->w * in->h + in_y * in->w + in_x] *
                            weights[c_out * conv_w * conv_h + y * conv_w + x];
                    }
                }
                if (use_bias == 1)
                {
                    out.data[c_out * out.w * out.h + h * out.w + w] += bias[0];
                }
            }
        }
    }

    free(weights);
    free(bias);
    free_in_out(in);
    in->c = out.c;
    in->w = out.w;
    in->h = out.h;
    in->data = out.data;
    out.data = NULL;
}

void bn_load_inference(int type, FILE *file, in_out *in)
{
    //print_in_out(*in);
    float gamma = 1;
    float beta = 0;
    float mean = 0;
    float variance = 1;

    for (int c = 0; c < in->c; c++)
    {
        if (!read_weight(type, &gamma, sizeof(float), 1, file))
        {
            debug("Read BN weights error!");
        }
        if (!read_weight(type, &beta, sizeof(float), 1, file))
        {
            debug("Read BN weights error!");
        }
        if (!read_weight(type, &mean, sizeof(float), 1, file))
        {
            debug("Read BN weights error!");
        }
        if (!read_weight(type, &variance, sizeof(float), 1, file))
        {
            debug("Read BN weights error!");
        }

        //debug("%f, %f, %f, %f ", gamma, beta, mean, variance);
        for (int h = 0; h < in->h; h++)
        {
            for (int w = 0; w < in->w; w++)
            {

                in->data[c * in->w * in->h + h * in->w + w] =
                    (in->data[c * in->w * in->h + h * in->w + w] - mean) /
                        sqrtf(variance + 0.001) * gamma +
                    beta;
            }
        }
    }
    //print_in_out(*in);
}

void activation_load_inference(int type, FILE *file, in_out *in)
{
    int activation_type = 0;
    if (!read_weight(type, &activation_type, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }
    //debug("activation type: %d", type);
    if (activation_type == 1) // relu
    {
        for (int c = 0; c < in->c; c++)
        {
            for (int h = 0; h < in->h; h++)
            {
                for (int w = 0; w < in->w; w++)
                {
                    in->data[c * in->w * in->h + h * in->w + w] =
                        in->data[c * in->w * in->h + h * in->w + w] > 0 ? in->data[c * in->w * in->h + h * in->w + w] : 0;
                }
            }
        }
        //print_in_out(*in);
    }
    else if (activation_type == 2) // softmax
    {
        float sum = 0;
        float largest = -FLT_MAX;
        for (int i = 0; i < in->w; ++i)
        {
            if (in->data[i] > largest)
            {
                largest = in->data[i];
            }
        }
        for (int i = 0; i < in->w; ++i)
        {
            float e = expf(in->data[i] - largest);
            sum += e;
            in->data[i] = e;
        }
        for (int i = 0; i < in->w; ++i)
        {
            in->data[i] /= sum;
        }
        //print_in_out(*in);
    }
}

void relu_load_inference(int type, FILE *file, in_out *in)
{
    float threshold = 0;
    if (!read_weight(type, &threshold, sizeof(float), 1, file))
    {
        debug("Read weights error!");
    }
    debug("ReLU threshold: %f", threshold);

    float max_value = 0;
    if (!read_weight(type, &max_value, sizeof(float), 1, file))
    {
        debug("Read weights error!");
    }
    debug("ReLU max_value: %f", max_value);

    float alpha = 0;
    if (!read_weight(type, &alpha, sizeof(float), 1, file))
    {
        debug("Read weights error!");
    }
    debug("leakyrelu alpha: %f", alpha);

    for (int c = 0; c < in->c; c++)
    {
        for (int h = 0; h < in->h; h++)
        {
            for (int w = 0; w < in->w; w++)
            {
                in->data[c * in->w * in->h + h * in->w + w] =
                    in->data[c * in->w * in->h + h * in->w + w] >= max_value ? max_value : in->data[c * in->w * in->h + h * in->w + w];
                in->data[c * in->w * in->h + h * in->w + w] =
                    in->data[c * in->w * in->h + h * in->w + w] <= threshold ? alpha * (in->data[c * in->w * in->h + h * in->w + w] - threshold) : in->data[c * in->w * in->h + h * in->w + w];
            }
        }
    }
}

void leakyrelu_load_inference(int type, FILE *file, in_out *in)
{
    float alpha = 0;
    if (!read_weight(type, &alpha, sizeof(float), 1, file))
    {
        debug("Read weights error!");
    }
    debug("leakyrelu alpha: %f", alpha);
    for (int c = 0; c < in->c; c++)
    {
        for (int h = 0; h < in->h; h++)
        {
            for (int w = 0; w < in->w; w++)
            {
                in->data[c * in->w * in->h + h * in->w + w] =
                    in->data[c * in->w * in->h + h * in->w + w] > 0 ? in->data[c * in->w * in->h + h * in->w + w] : in->data[c * in->w * in->h + h * in->w + w] * alpha;
            }
        }
    }
}

void maxpooling_load_inference(int type, FILE *file, in_out *in)
{
    in_out out = {0, 0, 0, NULL};
    int pool_w = 0;
    if (!read_weight(type, &pool_w, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }
    int pool_h = 0;
    if (!read_weight(type, &pool_h, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }
    int pooling_type = 0; //{"valid":1, "same":2}
    if (!read_weight(type, &pooling_type, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }
    int stride = 0;
    debug("pool_w:%d, pool_h:%d, type:%d", pool_w, pool_h, pooling_type);

    if (pooling_type == 1) // valid
    {
        stride = pool_w;
        out.w = in->w / pool_w;
        out.h = in->h / pool_h;
        out.c = in->c;
        out.data = calloc(out.w * out.h * out.c, sizeof(float));
        debug("out:%d x %d x %d", out.w, out.h, out.c);
        for (int c_out = 0; c_out < out.c; c_out++)
        {
            for (int h = 0; h < out.h; h++)
            {
                for (int w = 0; w < out.w; w++)
                {
                    float max = -FLT_MAX;
                    for (int y = 0; y < pool_h; y++)
                    {
                        int index_h = h * stride + y;
                        for (int x = 0; x < pool_w; x++)
                        {
                            int index_w = w * stride + x;
                            max =
                                in->data[c_out * in->w * in->h + index_h * in->w + index_w] > max
                                    ? in->data[c_out * in->w * in->h + index_h * in->w + index_w]
                                    : max;
                        }
                    }
                    out.data[c_out * out.w * out.h + h * out.w + w] = max;
                }
            }
        }
    }
    else if (pooling_type == 2) // same
    {
        stride = 1;
        out.w = in->w;
        out.h = in->h;
        out.c = in->c;
        out.data = calloc(out.w * out.h * out.c, sizeof(float));
        debug("out:%d x %d x %d", out.w, out.h, out.c);
        for (int c_out = 0; c_out < out.c; c_out++)
        {
            for (int h = 0; h < out.h; h++)
            {
                for (int w = 0; w < out.w; w++)
                {
                    float max = -FLT_MAX;
                    for (int y = 0; y < pool_h; y++)
                    {
                        int index_h = h * stride + y;
                        index_h = index_h > (in->h - 1) ? (in->h - 1) : index_h;
                        for (int x = 0; x < pool_w; x++)
                        {
                            int index_w = w * stride + x;
                            index_w = index_w > (in->w - 1) ? (in->w - 1) : index_w;
                            max =
                                in->data[c_out * in->w * in->h + index_h * in->w + index_w] > max
                                    ? in->data[c_out * in->w * in->h + index_h * in->w + index_w]
                                    : max;
                        }
                    }
                    out.data[c_out * out.w * out.h + h * out.w + w] = max;
                }
            }
        }
    }
    else
    {
        debug("Maxpooling type error!");
        return;
    }
    free_in_out(in);
    in->c = out.c;
    in->w = out.w;
    in->h = out.h;
    in->data = out.data;
    out.data = NULL;

    //print_in_out(*in);
}

void zeropadding2d_load_inference(int type, FILE *file, in_out *in)
{
    in_out out = {0, 0, 0, NULL};
    int top_pad = 0;
    if (!read_weight(type, &top_pad, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }
    int bottom_pad = 0;
    if (!read_weight(type, &bottom_pad, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }
    int left_pad = 0;
    if (!read_weight(type, &left_pad, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }
    int right_pad = 0;
    if (!read_weight(type, &right_pad, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }

    out.w = in->w + left_pad + right_pad;
    out.h = in->h + top_pad + bottom_pad;
    out.c = in->c;
    out.data = calloc(out.w * out.h * out.c, sizeof(float));
    debug("out:%d x %d x %d", out.w, out.h, out.c);
    for (int c_out = 0; c_out < out.c; c_out++)
    {
        for (int h = top_pad; h < out.h - bottom_pad; h++)
        {
            for (int w = left_pad; w < out.w - right_pad; w++)
            {

                out.data[c_out * out.w * out.h + h * out.w + w] =
                    in->data[c_out * in->w * in->h + (h - top_pad) * out.w + w - left_pad];
            }
        }
    }
    free_in_out(in);
    in->c = out.c;
    in->w = out.w;
    in->h = out.h;
    in->data = out.data;
    out.data = NULL;
}

void flatten_load_inference(int type, FILE *file, in_out *in)
{
    in_out out = {0, 0, 0, NULL};
    out.w = in->w;
    out.h = in->h;
    out.c = in->c;
    out.data = calloc(in->w * in->h * in->c, sizeof(float));
    for (int c = 0; c < out.c; c++)
    {
        for (int h = 0; h < out.h; h++)
        {
            for (int w = 0; w < out.w; w++)
            {
                out.data[h * out.c * out.h + w * out.c + c] =
                    in->data[c * out.w * out.h + h * out.w + w];
            }
        }
    }
    free_in_out(in);
    in->w = out.c * out.w * out.h;
    in->h = 1;
    in->c = 1;
    in->data = out.data;
    out.data = NULL;
}

void dense_load_inference(int type, FILE *file, in_out *in)
{
    in_out out = {0, 0, 0, NULL};
    int in_w = 0;
    if (!read_weight(type, &in_w, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }
    int out_w = 0;
    if (!read_weight(type, &out_w, sizeof(char), 1, file))
    {
        debug("Read weights error!");
    }
    debug("in_w:%d, out_w:%d", in_w, out_w);
    out.w = out_w;
    out.h = 1;
    out.c = 1;
    out.data = calloc(out.w * out.h * out.c, sizeof(float));

    float *weights = calloc(in->w * in->h * in->c, sizeof(float));
    float *bias = calloc(1, sizeof(float));

    for (int w = 0; w < out.w; w++)
    {
        for (int i = 0; i < in->w; i++)
        {
            if (!read_weight(type, weights + i, sizeof(float), 1, file))
            {
                debug("Read Dense weights error!");
            }
        }
        if (!read_weight(type, bias, sizeof(float), 1, file))
        {
            debug("Read Dense bias error!");
        }
        /*  
        for(int i = 0; i < in->w*in->h; i++)
        {
            printf("%f ", weights[i]);
        }
        printf("\n");
        printf("%f\n", *bias);
        */
        out.data[w] = 0;
        for (int l = 0; l < in->w; l++)
        {
            out.data[w] += in->data[l] * weights[l];
        }
        out.data[w] += bias[0];
    }
    free(weights);
    free(bias);
    free_in_out(in);
    in->c = out.c;
    in->w = out.w;
    in->h = out.h;
    in->data = out.data;
    out.data = NULL;
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

void yolo_v2(in_out *in, int resize_w, int resize_h)
{
    float anchors[] = {1.05, 5.13, 1.10, 3.60, 1.16, 5.14, 1.24, 4.22, 1.34, 5.24};
    float obj_threshold = 0.3;
    float mns_threshold = 0.3;
    int grid_h = in->h;         //10
    int grid_w = in->w;         //3
    int nb_box = 5;             // #anchors
    int paras = in->c / nb_box; //15
    debug("grid_h:%d, grid_w:%d, nb_box:%d", grid_h, grid_w, nb_box);
    // sigmoid for the 5th of each b_box
    for (int e = 0; e < nb_box; e++)
    {
        for (int j = 0; j < in->h; j++)
        {
            for (int k = 0; k < in->w; k++)
            {
                in->data[e * paras * grid_h * grid_w + 4 * grid_h * grid_w + j * grid_w + k] =
                    sigmoid(in->data[e * paras * grid_h * grid_w + 4 * grid_h * grid_w + j * grid_w + k]);
            }
        }
    }
    // softmax for 6th to 15th of each box, x 4th element and filter by obj_theshold
    for (int e = 0; e < nb_box; e++)
    {
        for (int j = 0; j < in->h; j++)
        {
            for (int k = 0; k < in->w; k++)
            {
                float sum = 0;
                float largest = -FLT_MAX;
                float smallest = FLT_MAX;
                for (int i = 5; i < 15; i++)
                {
                    float tmp = in->data[e * paras * grid_h * grid_w + i * grid_h * grid_w + j * grid_w + k];
                    if (tmp > largest)
                    {
                        largest = tmp;
                    }
                    if (tmp < smallest)
                    {
                        smallest = tmp;
                    }
                }
                for (int i = 5; i < 15; i++)
                {
                    float tmp = in->data[e * paras * grid_h * grid_w + i * grid_h * grid_w + j * grid_w + k];
                    tmp -= largest;
                    if (smallest < -100)
                    {
                        tmp = tmp / smallest * -100;
                    }
                    tmp = expf(tmp);
                    sum += tmp;
                    in->data[e * paras * grid_h * grid_w + i * grid_h * grid_w + j * grid_w + k] = tmp;
                }
                for (int i = 5; i < 15; i++)
                {
                    in->data[e * paras * grid_h * grid_w + i * grid_h * grid_w + j * grid_w + k] /= sum;
                    in->data[e * paras * grid_h * grid_w + i * grid_h * grid_w + j * grid_w + k] *=
                        in->data[e * paras * grid_h * grid_w + 4 * grid_h * grid_w + j * grid_w + k];
                    in->data[e * paras * grid_h * grid_w + i * grid_h * grid_w + j * grid_w + k] =
                        in->data[e * paras * grid_h * grid_w + i * grid_h * grid_w + j * grid_w + k] > obj_threshold ? in->data[e * paras * grid_h * grid_w + i * grid_h * grid_w + j * grid_w + k] : 0;
                }
            }
        }
    }
    // get b_box
    b_box *box_list[10];
    int index_box = 0;
    for (int e = 0; e < nb_box; e++)
    {
        for (int j = 0; j < grid_h; j++)
        {
            for (int k = 0; k < grid_w; k++)
            {
                float sum = 0;
                for (int i = 5; i < 15; i++)
                {
                    sum += in->data[e * paras * grid_h * grid_w + i * grid_h * grid_w + j * grid_w + k];
                }
                if (sum > 0)
                {
                    float x = in->data[e * paras * grid_h * grid_w + 0 * grid_h * grid_w + j * grid_w + k];
                    float y = in->data[e * paras * grid_h * grid_w + 1 * grid_h * grid_w + j * grid_w + k];
                    float w = in->data[e * paras * grid_h * grid_w + 2 * grid_h * grid_w + j * grid_w + k];
                    float h = in->data[e * paras * grid_h * grid_w + 3 * grid_h * grid_w + j * grid_w + k];
                    x = ((float)k + sigmoid(x)) / (float)grid_w;
                    y = ((float)j + sigmoid(y)) / (float)grid_h;
                    w = anchors[2 * e + 0] * exp(w) / (float)grid_w;
                    h = anchors[2 * e + 1] * exp(h) / (float)grid_h;
                    float confidence = in->data[e * paras * grid_h * grid_w + 4 * grid_h * grid_w + j * grid_w + k];
                    //debug("%f, %f, %f, %f, %f", x, y, w, h, confidence);
                    box_list[index_box] = calloc(1, sizeof(b_box));
                    box_list[index_box]->xmin = x - w / 2.0f;
                    box_list[index_box]->ymin = y - h / 2.0f;
                    box_list[index_box]->xmax = x + w / 2.0f;
                    box_list[index_box]->ymax = y + h / 2.0f;
                    box_list[index_box]->confidence = confidence;
                    float max = -FLT_MAX;
                    for (int i = 5; i < 15; i++)
                    {
                        if (max < in->data[e * paras * grid_h * grid_w + i * grid_h * grid_w + j * grid_w + k])
                        {
                            box_list[index_box]->label = i - 5;
                            box_list[index_box]->score = in->data[e * paras * grid_h * grid_w + i * grid_h * grid_w + j * grid_w + k];
                            max = in->data[e * paras * grid_h * grid_w + i * grid_h * grid_w + j * grid_w + k];
                        }
                    }
                    index_box++;
                    if (index_box >= 10)
                    {
                        debug("Error: Not enough buffer for b_box!");
                        break;
                    }
                }
            }
        }
    }
    debug("index_box: %d", index_box);
    // debug
    for (int i = 0; i < index_box; i++)
    {
        debug("%d, %f, %d, %d, %d, %d",
              box_list[i]->label, box_list[i]->score,
              (int)(box_list[i]->xmin * resize_w),
              (int)(box_list[i]->xmax * resize_w),
              (int)(box_list[i]->ymin * resize_h),
              (int)(box_list[i]->ymax * resize_h));
        free(box_list[i]);
    }
}

int read_weight(int type, void *buf, size_t size, size_t n, FILE *fp)
{
    if (type == 0)
    {
        return fread(buf, size, n, fp);
    }
    else if (type == 1)
    {
        uint32_t add = WEIGHT_ADDR_BASE + address_shift;
        //debug("buf:%p, %d, size:%d, add:%d", add, *(char*)add, size, address_shift);
        if (size == 1)
        {
            *(uint8_t *)buf = *(uint8_t *)add;
            address_shift += 1;
            return 1;
        }
        else if (size == 2)
        {
            *(uint16_t *)buf = *(uint16_t *)add;
            address_shift += 2;
            return 2;
        }
        else if (size == 4)
        {
            *(uint32_t *)buf = *(uint32_t *)add;
            address_shift += 4;
            return 4;
        }
        else
        {
            return 0;
        }
    }
    else
    {
        return 0;
    }
}

in_out *uInference(int type, in_out *im, char *model_name)
{
    in_out *in = im;
    FILE *file;

    if (type == 0)
    {
        debug("model_name: %s", model_name);
        file = fopen(model_name, "rb");
        if (file == 0)
        {
            debug("Couldn't open file: %s", model_name);
            return in;
        }
    }
    else if (type == 1)
    {
    }
    else
    {
        debug("Error: type Error!");
        return in;
    }

    int layer = 0;
    int flag_end = 0;
    while (flag_end == 0)
    {
        uint8_t layer_type = 0;
        if (!read_weight(type, &layer_type, sizeof(uint8_t), 1, file))
        {
            debug("Read weights error!");
        }
        debug("layer type: %d", layer_type);

        switch (layer_type)
        {
        case 0: //End
            flag_end = 1;
            address_shift = 0;
            debug("Read weights finished!");
            break;
        case 1: //Conv2D
            debug("Layer Conv2D");
            debug("in: %d x %d x %d", in->w, in->h, in->c);
            conv2d_load_inference(type, file, in);
            break;
        case 2: //BatchNormalization
            debug("Layer BatchNormalization");
            debug("in: %d x %d x %d", in->w, in->h, in->c);
            bn_load_inference(type, file, in);
            break;
        case 3: //Activation
            debug("Layer Activation");
            debug("in: %d x %d x %d", in->w, in->h, in->c);
            activation_load_inference(type, file, in);
            break;
        case 4: //MaxPooling2D
            debug("Layer MaxPooling2D");
            debug("in: %d x %d x %d", in->w, in->h, in->c);
            maxpooling_load_inference(type, file, in);
            break;
        case 5: //Flatten
            debug("Layer Flatten");
            debug("in: %d x %d x %d", in->w, in->h, in->c);
            flatten_load_inference(type, file, in);
            break;
        case 6: //Dense
            debug("Layer Dense");
            debug("in: %d x %d x %d", in->w, in->h, in->c);
            dense_load_inference(type, file, in);
            break;
        case 7: //InputLayer
            debug("Layer InputLayer");

            break;
        case 8: // Model
            debug("Layer Model");

            break;
        case 9: // Reshape
            debug("Layer Reshape");

            break;
        case 10: // Lambda
            debug("Layer Lambda");

            break;
        case 11: // LeakyReLU
            debug("Layer LeakyReLU");
            leakyrelu_load_inference(type, file, in);
            break;
        case 12: // DepthwiseConv2D
            debug("Layer DepthwiseConv2D");
            depthwiseconv2d_load_inference(type, file, in);
            break;
        case 13: // ZeroPadding2D
            debug("Layer ZeroPadding2D");
            zeropadding2d_load_inference(type, file, in);
            break;
        case 14: // ReLU
            debug("Layer ReLU");
            relu_load_inference(type, file, in);
            break;
        default:
            debug("layer_type: %d not recognized!", layer_type);
            fclose(file);
            free_in_out(in);
            return in;
        }

        /*
        if (layer == 33)
        {
            save_in_out(*in);
            debug("shape:w=%d,h=%d,c=%d", in->w, in->h, in->c);
            debug("layer %d, type %d saved", layer, layer_type);
        }
        */
        layer++;
    }
    debug("Number of layers: %d", layer);
    fclose(file);

    return in;
}
