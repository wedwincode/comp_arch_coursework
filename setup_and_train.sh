#!/bin/bash

git clone https://github.com/wedwincode/comp_arch_coursework.git
cd comp_arch_coursework

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

nohup python main.py > train.log 2>&1 &