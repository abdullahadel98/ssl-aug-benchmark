#!/usr/bin/env bash

mkdir datasets
cd datasets
# Download describable textures dataset
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xf dtd-r1.0.1.tar.gz
rm dtd-r1.0.1.tar.gz

mkdir mvtec
cd mvtec
# Download MVTec anomaly detection dataset
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz
tar -xf mvtec_anomaly_detection.tar.xz
rm mvtec_anomaly_detection.tar.xz


# download cifar-c dataset
wget https://zenodo.org/records/3555552/files/CIFAR-100-C.tar
tar -xf CIFAR-100-C.tar
rm CIFAR-100-C.tar