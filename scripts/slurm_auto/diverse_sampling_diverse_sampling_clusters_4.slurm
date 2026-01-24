#!/bin/bash
#SBATCH --account=def-cfwelch
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=5:59:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=hendek12@mcmaster.ca
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=logs/slurm/diverse_sampling_clusters_4.out

module load python/3.11
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load cudnn/9.2.1.18
module load arrow/18.1.0

source env/bin/activate
python src/ft_bert_no_verdicts.py --use_authors='false' --author_encoder='average' --loss_type='focal' --num_epochs=10 --sbert_model='sentence-transformers/all-distilroberta-v1' --bert_tok='sentence-transformers/all-distilroberta-v1' --sbert_dim=768 --user_dim=768 --model_name='sbert' --split_type='sit' --situation='text' --authors_embedding_path='data/final_embeddings/diverse_sampling_clusters.pkl' --plot_title='diverse_sampling_clusters Embedding Diverse' --path_to_data='data/' --social_norm='true' --log_file='diverse_sampling_clusters_4'
