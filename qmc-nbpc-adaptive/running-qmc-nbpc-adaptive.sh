#!/bin/bash

for k in 10.0 12.0 #10.0 20.0 30.0
do
    python run-qmc-nbpc-adaptive.py $k
done
