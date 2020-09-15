# coding: utf-8

import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scales', help='scales', type=int, default=5)
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    args = parse()
    print(args.scales)
