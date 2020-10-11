import numpy as np
import os

from tempfile import mkstemp
from shutil import move
from os import fdopen, rename
import shutil
from pathlib import Path


def make_dataset_export_hfnet_yaml(file_path, dataset_name):
    fh, abs_path = mkstemp()
    with fdopen(fh, 'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                line = line.replace("aachen", dataset_name, 100)
                new_file.write(line)
    # Move new file
    move(abs_path, os.path.dirname(file_path))
    rename(os.path.join(os.path.dirname(file_path), abs_path.split("/")[-1]),
           os.path.join(os.path.dirname(file_path), "hfnet_export_" + dataset_name + "_db.yaml"))


def make_dataset_export_experiment_yaml(file_path, dataset_name, exper_name):
    fh, abs_path = mkstemp()
    with fdopen(fh, 'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                line = line.replace("medium4", dataset_name, 100)
                new_file.write(line)
    # Move new file
    move(abs_path, os.path.dirname(file_path))
    rename(os.path.join(os.path.dirname(file_path), abs_path.split("/")[-1]),
           os.path.join(os.path.dirname(file_path), exper_name + "_export_" + dataset_name + "_db.yaml"))


def make_dataset_evaluate_script(file_path, dataset_name):
    fh, abs_path = mkstemp()
    with fdopen(fh, 'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                if not "Sf.db" in line:
                    line = line.replace("medium4", dataset_name, 100)
                new_file.write(line)
    # Move new file
    move(abs_path, os.path.dirname(file_path))
    rename(os.path.join(os.path.dirname(file_path), abs_path.split("/")[-1]),
           os.path.join(os.path.dirname(file_path), "evaluate_" + dataset_name + ".py"))


def make_dataset_class(file_path, dataset_name):
    fh, abs_path = mkstemp()
    with fdopen(fh, 'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                line = line.replace("medium4", dataset_name, 100)
                new_file.write(line)
    # Move new file
    move(abs_path, os.path.dirname(file_path))
    rename(os.path.join(os.path.dirname(file_path), abs_path.split("/")[-1]),
           os.path.join(os.path.dirname(file_path), dataset_name + ".py"))


def walklevel(path, depth=1):
    """It works just like os.walk, but you can pass it a level parameter
       that indicates how deep the recursion will go.
       If depth is -1 (or less than 0), the full depth is walked.
    """
    # if depth is negative, just walk
    if depth < 0:
        for root, dirs, files in os.walk(path):
            yield root, dirs, files

    # path.count works because is a file has a "/" it will show up in the list
    # as a ":"
    path = path.rstrip(os.path.sep)
    num_sep = path.count(os.path.sep)
    for root, dirs, files in os.walk(path):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + depth <= num_sep_this:
            del dirs[:]


def main():
    current_path = "/home/ara/Documents/Internship"
    sub_folder = "OpenSfM/data/"
    hfnet_path = "/home/ara/PycharmProjects/hfnet"
    for dirpath, dirnames, filenames in walklevel(os.path.join(current_path, sub_folder)):
        print(f"Running on {dirpath}")

        dataset_name = dirpath.split(os.path.sep)[-1]
        make_dataset_export_hfnet_yaml(f"{hfnet_path}/hfnet/configs/hfnet_export_aachen_db.yaml",
                                       dataset_name)
        make_dataset_class(f"{hfnet_path}/hfnet/datasets/medium4.py", dataset_name)
        os.system(
            f"python3 hfnet/export_predictions.py {hfnet_path}/hfnet/configs/hfnet_export_" + dataset_name + "_db.yaml --exper_name hfnet --keys keypoints,scores,local_descriptor_map,global_descriptor " + dataset_name)
        make_dataset_evaluate_script(f"{hfnet_path}/hfnet/evaluate_medium4.py", dataset_name)
        os.system(
            f"python3 {hfnet_path}/hfnet/evaluate_" + dataset_name + f".py {os.path.join(current_path, sub_folder, dataset_name)}" + "/colmap_models/db_triangulated eval_recons_hfnet --local_method hfnet --global_method hfnet --build_db --queries query --export_poses")


if __name__ == '__main__':
    main()
