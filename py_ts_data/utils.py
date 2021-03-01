import os
import json
import numpy as np

def parse_line(line):
    parts = line.rstrip().split(":")
    data = parts[0]
    label = parts[1]

    variables = data.split(";")
    ts = []
    for var in variables:
        ts.append([float(x) for x in var.split(" ")])
    return np.array(ts), label


def parse_variable_length_file(filepath, info):
    data = []
    labels = []
    max_len = float('-inf')
    nvars = info["n_variables"]
    with open(filepath) as f:
        for line in f:
            ts, label = parse_line(line)
            labels.append(label)
            assert nvars == ts.shape[0]
            if ts.shape[1] > max_len:
                max_len = ts.shape[1]
            data.append(ts)

    final_dataset = []
    for ts in data:
        final = np.empty((nvars, max_len))
        final[:] = np.nan
        final[:ts.shape[0], :ts.shape[1]] = ts
        final_dataset.append(final)

    return np.array(final_dataset), np.array(labels)

def parse_fixed_length_file(filepath, info):
    data = []
    labels = []
    nvars = info["n_variables"]
    with open(filepath) as f:
        for line in f:
            ts, label = parse_line(line)
            assert nvars == ts.shape[0]
            data.append(ts)
            labels.append(label)
    return np.array(data), np.array(labels)

def parse_file(filepath, info):
    data = []
    labels = []
    max_len = float('-inf')
    nvars = info["n_variables"]
    with open(filepath) as f:
        for line in f:
            ts, label = parse_line(line)
            labels.append(label)
            assert nvars == ts.shape[0]
            if ts.shape[1] > max_len:
                max_len = ts.shape[1]
            data.append(ts)

    final_dataset = data
    if info["n_timestamps"] == -1:
        final_dataset = []
        for ts in data:
            final = np.empty((nvars, max_len))
            final[:] = np.nan
            final[:ts.shape[0], :ts.shape[1]] = ts
            final_dataset.append(final)
    
    return np.array(data), np.array(labels)



