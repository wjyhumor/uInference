
import argparse
import os
import re
import struct
import numpy as np

argparser = argparse.ArgumentParser(description='Transform weights to binary')

argparser.add_argument(
    '-input',
    help='path of model weights input')

argparser.add_argument(
    '-output',
    help='name of model weights output')


def _main_(args):
    input_path = args.input
    output_name = args.output
    weights = []

    with open(input_path, mode='r') as f_in:
        while 1:
            line = f_in.readline()
            if not line:
                break
            if re.search('0x', line):
                l = line.strip('\n').split(',')
                l = list(filter(None, l))
                # print(l)
                for i in range(len(l)):
                    item = int(l[i].strip(' '), 16)
                    weights.append(item)
    # print(weights)
    print("Save weights of length: " + str(len(weights)) + " to " + output_name)

    with open(output_name, mode='w+b') as f_out:
        for item in weights:
            f_out.write(struct.pack('B', item))
    f_out.close()


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
