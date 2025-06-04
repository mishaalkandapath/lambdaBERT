#!/bin/bash

FILENAME=$1

cd $SLURM_TMPDIR

virtualenv env -p python3.10
module purge
source env/bin/activate

pip install -r /home/mishaalk/scratch/requirements.txt
deactivate

mkdir data
cp /home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/data/dataset_splits.pkl data/dataset_splits.pkl
cp /home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/data/input_sentences.csv data/input_sentences.csv

cp /home/mishaalk/scratch/$FILENAME $FILENAME
if [ [$1 == "simplestlambda.tgz"] ]; then
    tar -zxf $FILENAME --strip-components=2 lambdaBERT/data/ -C data/
else
    tar -zxf $FILENAME --strip-components=1 data/ -C data/
fi
