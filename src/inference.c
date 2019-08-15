#include "inference.h"

void conv2d_load_inference(FILE *file, in_out *in, in_out *out)
{
    int conv_w = fgetc(file);
    int conv_h = fgetc(file);
    int in_c = fgetc(file);
    int out_c = fgetc(file);
    int stride = fgetc(file);
    int type = fgetc(file); //{"valid":1, "same":2}
    int pad = 0;
    //debug("conv_w:%d, conv_h:%d, in_c:%d, out_c:%d, stride:%d, type:%d", 
    //    conv_w, conv_h, in_c, out_c, stride, type);
    
    if(type == 2) // same
    {
        out->w = in->w;
        out->h = in->h;
        out->c = out_c;
        out->data = calloc(out->w*out->h*out->c, sizeof(float));
        pad = conv_w / 2;
        debug("pad:%d", pad);
    } 
    else if(type == 1) // valid
    {
        out->w = in->w;
        out->h = in->h;
        out->c = out_c;
        //out->data = calloc(out->w*out->h*out->c, sizeof(float));
    }
    else
    {
        debug("Read conv2d type error!");
        exit(0);
    }
  
    float* weights = calloc(conv_w*conv_h*in_c, sizeof(float));
    float* bias = calloc(1, sizeof(float));
    
    for(int c_out = 0; c_out < out->c; c_out++)
    {
        for(int i = 0; i < conv_w*conv_h*in_c; i++)
        {
            if(!fread(weights+i,sizeof(float),1,file))
            {
                debug("Read conv2d weights error!");
            }
        }
        if(!fread(bias,sizeof(float),1,file))
        {
            debug("Read conv2d bias error!");
        }
        /*  
        for(int i = 0; i < conv_w*conv_h*in_c; i++)
        {
            printf("%f ", weights[i]);
        }
        printf("\n");
        printf("%f\n", *bias);
        */

        for(int h = 0; h < out->h; h++)
        {
            for(int w = 0; w < out->w; w++)
            {
                out->data[c_out*out->w*out->h + h*out->w + w] = 0;
                for(int c_in = 0; c_in < in->c; c_in++)
                {
                    for(int y = 0; y < conv_h; y++)
                    {
                        int in_y = h + y - pad; 
                        if(in_y < 0 || in_y >= out->h) continue;
                        for(int x = 0; x < conv_w; x++)
                        {
                            int in_x = w + x - pad;
                            if(in_x < 0 || in_x >= out->w) continue;
                            out->data[c_out*out->w*out->h + h*out->w + w]
                            += in->data[c_in*in->w*in->h + in_y*in->w + in_x] 
                                * weights[c_in*conv_w*conv_h + y*conv_w + x];
                        }
                    }
                }
                out->data[c_out*out->w*out->h + h*out->w + w] += bias[0];
            }
        }
    }
    //print_image(*in);
    //print_image(*out);
    //debug("%f", out->data[2*out->w*out->h + 5*out->w + 4]);

    free(weights);
    free(bias);
    free_in_out(in);
}

void bn_load_inference(FILE *file, in_out *in)
{
    float gamma = 1;
    float beta = 0;
    float mean = 0;
    float variance = 1;
    
    for(int c = 0; c < in->c; c++)
    {
        if(!fread(&gamma,sizeof(float),1,file))
        {
            debug("Read BN weights error!");
        }
        if(!fread(&beta,sizeof(float),1,file))
        {
            debug("Read BN weights error!");
        }
        if(!fread(&mean,sizeof(float),1,file))
        {
            debug("Read BN weights error!");
        }
        if(!fread(&variance,sizeof(float),1,file))
        {
            debug("Read BN weights error!");
        }
        //printf("%f, %f, %f, %f \n", gamma, beta, mean, variance);
        for(int h = 0; h < in->h; h++)
        {
            for(int w = 0; w < in->w; w++)
            {
                
                in->data[c*in->w*in->h + h*in->w + w] = 
                (in->data[c*in->w*in->h + h*in->w + w] - mean) / sqrtf(variance + 0.001)
                * gamma + beta;
            }
        }
    }

    //print_image(*in);
    //debug("%f", in->data[2*in->w*in->h + 5*in->w + 4]);
}

void activation_load_inference(FILE *file, in_out *in)
{
    int type = fgetc(file);
    //debug("activation type: %d", type);
    if(type == 1) // relu
    {
        for(int c = 0; c < in->c; c++)
        {
            for(int h = 0; h < in->h; h++)
            {
                for(int w = 0; w < in->w; w++)
                {
                    in->data[c*in->w*in->h + h*in->w + w] = 
                    in->data[c*in->w*in->h + h*in->w + w] > 0 ?
                    in->data[c*in->w*in->h + h*in->w + w] : 0;
                }
            }
        }
    }
    else if(type == 2) // softmax
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
        print_image(*in);
    }

    //print_image(*in);
    //debug("%f", in->data[5*in->w*in->h + 5*in->w + 4]);
}

void maxpooling_load_inference(FILE *file, in_out *in, in_out *out)
{
    int pool_w = fgetc(file);
    int pool_h = fgetc(file);
    int type = fgetc(file); //{"valid":1, "same":2}
    int stride = 0;
    debug("pool_w:%d, pool_h:%d, type:%d", pool_w, pool_h, type);
    
    if(type == 1) // valid
    {
        stride = pool_w;
        out->w = in->w / pool_w;
        out->h = in->h / pool_h;
        out->c = in->c;
        out->data = calloc(out->w*out->h*out->c, sizeof(float));
        debug("out:%d x %d x %d", out->w, out->h, out->c);
    } 
    else
    {
        debug("Read maxpooling type error!");
        exit(0);
    }
  
    for(int c_out = 0; c_out < out->c; c_out++)
    {
        for(int h = 0; h < out->h; h++)
        {
            for(int w = 0; w < out->w; w++)
            {
                float max = -FLT_MAX;
                for(int y = 0; y < pool_h; y++)
                {
                    int index_h = h*stride + y;
                    for(int x = 0; x < pool_w; x++)
                    {                        
                        int index_w = w*stride + x;
                        max = 
                        in->data[c_out*in->w*in->h + index_h*in->w + index_w] > max ?
                        in->data[c_out*in->w*in->h + index_h*in->w + index_w] : max;
                    }
                }
                out->data[c_out*out->w*out->h + h*out->w + w] = max;
            }
        }
        
    }

    //print_image(*out);
    //debug("%f", out->data[2*out->w*out->h + 1*out->w + 1]);

    free_in_out(in);
}


void flatten_load_inference(FILE *file, in_out *in)
{
    in_out out = {0, 0, 0, NULL};
    out.w = in->w;
    out.h = in->h;
    out.c = in->c;
    out.data = calloc(in->w*in->h*in->c, sizeof(float));
    for(int c = 0; c < out.c; c++)
    {
        for(int h = 0; h < out.h; h++)
        {
            for(int w = 0; w < out.w; w++)
            {
               out.data[h*out.c*out.h + w*out.c + c] =  
               in->data[c*out.w*out.h + h*out.w + w];
            } 
        }
    }
    free_in_out(in);
    in->w = out.c*out.w*out.h;
    in->h = 1; 
    in->c = 1;
    in->data = out.data;
    out.data = NULL;
    //print_image(*in);
}


void dense_load_inference(FILE *file, in_out *in, in_out *out)
{
    int in_w = fgetc(file);
    int out_w = fgetc(file);
    debug("in_w:%d, out_w:%d", in_w, out_w);
    out->w = out_w;
    out->h = 1;
    out->c = 1;
    out->data = calloc(out->w*out->h*out->c, sizeof(float));

    float* weights = calloc(in->w*in->h*in->c, sizeof(float));
    float* bias = calloc(1, sizeof(float));
    
    for(int w = 0; w < out->w; w++)
    {
        for(int i = 0; i < in->w; i++)
        {
            if(!fread(weights+i,sizeof(float),1,file))
            {
                debug("Read Dense weights error!");
            }
        }
        if(!fread(bias,sizeof(float),1,file))
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
        out->data[w] = 0;
        for(int l = 0; l < in->w; l++)
        {
            out->data[w] += in->data[l] * weights[l];
        }
        out->data[w] += bias[0];
    }

    //print_image(*in);
    //print_image(*out);

    free(weights);
    free(bias);
    free_in_out(in);
}