##############################################################################################################################

source venv/bin/activate

# name="M1_seq"

# python main.py -n "./results/AAAI/$name" -j --min_variance 0.01 -t 10000000
 
# python experiments.py -n "./results/AAAI/${name}_last" -j --output_file "./results/AAAI/${name}_last" --min_variance 0.01 

# python experiments.py -n "./results/AAAI/${name}_last" -j --output_file "./results/AAAI/${name}_last" --min_variance 0.01 

# ##############################################################################################################################

# name="M1_hum_10l_seq_variant"

# python main.py -n "./results/AAAI/$name" -j --min_variance 0.01 -h --hf './human_data/human/pickle/human_10l_speed_formatted.pickle' -t 10000000

# python experiments.py -n "./results/AAAI/${name}" -j --output_file "./results/AAAI/${name}" --min_variance 0.01 

# python experiments.py -n "./results/AAAI/${name}_last" -j --output_file "./results/AAAI/${name}_last" --min_variance 0.01 

# ##############################################################################################################################

name="inf_BC"

python behavioural_clonning.py 

python main.py -n "./results/AAAI/$name" -j --variance 0.2  --min_variance 0.01 --lr 0.0005 -t 3000000

python experiments.py -n "./results/AAAI/${name}" -j --output_file "./results/AAAI/${name}" --min_variance 0.01 

python experiments.py -n "./results/AAAI/${name}_last" -j --output_file "./results/AAAI/${name}_last" --min_variance 0.01 

deactivate
