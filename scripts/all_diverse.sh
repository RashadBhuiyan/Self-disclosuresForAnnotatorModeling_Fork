#!/bin/bash

# This script generates and queues slurm job files for running diverse tests on the model. You can comment out the jobs you don't want to run.

# Each job config line: embedding_file|output_name
declare -a job_configs=(
    "data/final_embeddings/diverse_sampling_all.pkl|diverse_sampling_all"
    "data/final_embeddings/diverse_sampling_manual.pkl|diverse_sampling_manual"
    "data/final_embeddings/diverse_sampling_clusters.pkl|diverse_sampling_clusters"

)

for i in {1..5}; do

    for job in "${job_configs[@]}"; do
        IFS="|" read -r embedding_file output_name <<< "$job"

        

        slurm_file="scripts/slurm_auto/diverse_sampling_${output_name}_${i}.slurm"
        cat > "$slurm_file" <<EOL
#!/bin/bash
#SBATCH --account=def-cfwelch
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=5:59:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=hendek12@mcmaster.ca
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=logs/slurm/${output_name}_${i}.out

module load python/3.11
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load cudnn/9.2.1.18
module load arrow/18.1.0

source env/bin/activate
python src/ft_bert_no_verdicts.py \
--use_authors='false' \
--author_encoder='average' \
--loss_type='focal' \
--num_epochs=10 \
--sbert_model='sentence-transformers/all-distilroberta-v1' \
--bert_tok='sentence-transformers/all-distilroberta-v1' \
--sbert_dim=768 \
--user_dim=768 \
--model_name='sbert' \
--split_type='sit' \
--situation='text' \
--authors_embedding_path='${embedding_file}' \
--plot_title='${output_name} Embedding Diverse' \
--path_to_data='data/' \
--social_norm='true' \
--log_file='${output_name}_${i}'
EOL

    sbatch "$slurm_file"
    done

done
