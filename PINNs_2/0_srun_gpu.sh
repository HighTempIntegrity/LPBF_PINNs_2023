#!/bin/bash
#SBATCH -n 4
#SBATCH --time=24:00:00
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem-per-cpu=10000
#SBACTH --tmp=4000
#SBATCH -J 'PINN'
#SBATCH --mail-type=BEGIN,FAIL,END
python PINNS2.py
python PINNS2.py
python PINNS2.py
python PINNS2.py
python PINNS2.py
python PINNS2.py
python PINNS2.py
python PINNS2.py
python PINNS2.py
python PINNS2.py
python PINNS2.py
python PINNS2.py
python PINNS2.py
python PINNS2.py
python PINNS2.py
