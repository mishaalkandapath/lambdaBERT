#!/bin/sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=bash
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=v100l:2
#SBATCH --mem=128G

#SBATCH --time=9:0:0
#SBATCH --account=def-gpenn

if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename>"
    echo "Example: $0 myfile.py"
    exit 1
fi

./compute_can_setup.sh $1
module purge 
module load opencv cuda gcc scipy-stack 
source ~/py10/bin/activate

#python models.py --discrete --model_path /home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/lambdabertmodel_linear_last/best_linear_last.ckpt --save_dir /home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/discrete_linear_last/ --batch_size 20 --finetune_discrete #--model_is_discrete #--bert_is_last
python models.py --save_dir "/home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/linear_last_var_reg/" --model_path "/home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/linear_last_var_reg/train_r6_1_varrg.ckpt" --batch_size 150 --bert_is_last --custom_transformer
# python models.py --save_dir "/home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/linear_last/" --model_path /home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/linear_last/train_r1_lin_last.ckpt --batch_size 20 --bert_is_last #--custom_transformer
# python models.py --shuffled_mode --t_force --save_dir "/home/mishaalk/scratch/lambdaModelsNoTforce/" --batch_size 30
# python models.py --shuffled_mode --save_dir "/home/mishaalk/scratch/lambdaPosModel/" --model_path /home/mishaalk/scratch/bestpost.ckpt
# rem,ember not T Force is ur model w discrete

# PREVIOUSLT  --cpus-per-task=1

#salloc --time=1:30:0 --ntasks=12 --gres=gpu:v100l:2 --mem=128G --nodes=1 --account=def-gpenn
