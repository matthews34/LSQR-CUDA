import re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class Result():
    def __init__(self, m: int, n: int, elapsed_time: float, it_max_exceeded: bool, it_number: int, final_residual: float, x: np.array):
        self.m = m
        self.n = n
        self.elapsed_time = elapsed_time
        self.it_max_exceeded = it_max_exceeded
        self.final_residual = final_residual
        self.x = x

class Comparison():
    comparison_list = []
    def __init__(self, cpu_result, gpu_result):
        self.cpu_result = cpu_result
        self.gpu_result = gpu_result
        if cpu_result.n != gpu_result.n:
            exit(-1)
        if cpu_result.m != gpu_result.m:
            exit(-1)
        self.n = cpu_result.n
        self.m = cpu_result.m
        self.time_diff = float(cpu_result.elapsed_time) - float(gpu_result.elapsed_time)
        self.x_error = abs(cpu_result.x - gpu_result.x)


def read_result(file_path: str) -> Result:
    file = open(file_path, 'r')

    line = file.readline()
    match = re.search(r'n = ([0-9]+)', line)
    if match != None:
        n = match.group(1)
    else:
        print("No value for n")
        exit(-1)
    match = re.search(r'm = ([0-9]+)', line)
    if match != None:
        m = match.group(1)
    else:
        print("No value for m")
        exit(-1)
    
    line = file.readline()
    # match = re.search(r'initial residual = ([0-9]+)', line)
    # if match != None:
    #     initial_residual = match.group(1)
    # else:
    #     exit(-1)

    line = file.readline()
    match = re.search(r'finished after ([0-9]+) iterations', line)
    it_max_exceeded = False
    it_number = 0
    if match != None:
        it_number = match.group(1)
    else:
        match = re.search(r'it_max exceeded', line)
        if match != None:
            it_max_exceeded = True
        else:
            print("No information on iterations: '{}'".format(file_path))
            exit(-1)
    
    line = file.readline()
    match = re.search(r'\[s\]: ([0-9]+\.[0-9]+)', line)
    if match != None: 
        elapsed_time = match.group(1)
    else:
        print("No information for elapsed time")
        exit(-1)

    line = file.readline()
    match = re.search(r'residual = ([0-9]+\.[0-9]+)', line)
    if match != None: 
        final_residual = match.group(1)
    else:
        print("No information for final_residual")
        exit(-1)

    line = file.readline()
    x = re.findall(r'-*[0-9]+\.[0-9]+', line)
    if x == None: 
        print("No values for x")
        exit(-1)

    file.close()
    return Result(m, n, elapsed_time, it_max_exceeded, it_number, final_residual, np.array(x, dtype=float))

def read_all_results(folder_path: str):
    read_list = []
    for file in os.listdir(folder_path):
        file_path = folder_path + "/" + file
        if file_path[:-4] in read_list:
            continue
        if file_path[-3:] == "cpu":
            cpu_result = read_result(file_path)
            gpu_result = read_result(file_path[:-3] + "gpu")
            read_list.append(file_path[:-4])
            Comparison.comparison_list.append(Comparison(cpu_result, gpu_result))
        elif file_path[-3:] == "gpu":
            continue
        else:
            print("Invalid filename")
            exit(-1)
        
if __name__ == "__main__":
    read_all_results("results")
    cpu_times = [c.cpu_result.elapsed_time for c in Comparison.comparison_list]
    gpu_times = [c.gpu_result.elapsed_time for c in Comparison.comparison_list]
    time_diff = [c.time_diff for c in Comparison.comparison_list]
    size_list = [int(c.m) * int(c.n) for c in Comparison.comparison_list]
    average_error_list = [np.average(c.x_error) for c in Comparison.comparison_list]
    std_error_list = [np.std(c.x_error) for c in Comparison.comparison_list]

    # time_diff
    plt.scatter(size_list, time_diff)
    plt.title("Time difference between CPU and GPU implementations")
    plt.xlabel("Matrix size (m x n)")
    plt.xlim([0, None])
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.savefig("time_diff.png")
    plt.clf()

    # cpu times
    plt.scatter(size_list, cpu_times)
    plt.title("CPU runtime")
    plt.xlabel("Matrix size (m x n)")
    plt.xlim([0, None])
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.savefig("cpu_time.png")
    plt.clf()

    # gpu times
    plt.scatter(size_list, gpu_times)
    plt.title("GPU runtime")
    plt.xlabel("Matrix size (m x n)")
    plt.xlim([0, None])
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.savefig("gpu_time.png")
    plt.clf()

    # average error
    plt.scatter(size_list, average_error_list)
    plt.title("Average difference")
    plt.xlabel("Matrix size (m x n)")
    plt.xlim([0, None])
    plt.grid(True)
    plt.savefig("avg_error.png")
    plt.clf()
    
    # error standard deviation 
    plt.scatter(size_list, std_error_list)
    plt.title("Standard deviation of errors")
    plt.xlabel("Matrix size (m x n)")
    plt.ylabel("Standard deviation")
    plt.xlim([0, None])
    plt.grid(True)
    plt.savefig("std_error.png")
    plt.clf()