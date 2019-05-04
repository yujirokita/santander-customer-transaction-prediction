#!/bin/bash

mkdir input
mkdir output
mkdir data
mkdir script/log
echo "datetime,script,duration,local_score,public_LB,private_LB" > script/results.csv

kaggle competitions download -q -c santander-customer-transaction-prediction -p input