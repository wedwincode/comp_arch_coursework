#!/bin/bash

git clone https://github.com/wedwincode/comp_arch_coursework.git
cd comp_arch_coursework

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

nohup python main.py > train.log 2>&1 &