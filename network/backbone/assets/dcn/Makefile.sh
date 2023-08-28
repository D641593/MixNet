#!/bin/bash
rm  *.so 
python3 setup.py build_ext --inplace
rm -rf ./build


