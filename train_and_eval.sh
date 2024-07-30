##############################################################################################################################

name="M1_var_01"

python main.py -n "./results/AAAI/$name" -j --min_variance 0.01

python experiments.py -n "./results/AAAI/$name" -j --output_file "./results/AAAI/${name}.txt" --min_variance 0.01

python experiments.py -n "./results/AAAI/${name}_last" -j --output_file "./results/AAAI/${name}_last.txt" --min_variance 0.01

##############################################################################################################################

name="BotBeh_var_01"

python main.py -n "./results/AAAI/$name" -j --min_variance 0.01 -h --hf './human_data/inferno_formatted.pickle'

python experiments.py -n "./results/AAAI/$name" -j --output_file "./results/AAAI/${name}.txt" --min_variance 0.01

python experiments.py -n "./results/AAAI/${name}_last" -j --output_file "./results/AAAI/${name}_last.txt" --min_variance 0.01

##############################################################################################################################

name="HuBeh_var_01"

python main.py -n "./results/AAAI/$name" -j --min_variance 0.01 -h --hf './human_data/human_formatted.pickle'

python experiments.py -n "./results/AAAI/$name" -j --output_file "./results/AAAI/${name}.txt" --min_variance 0.01

python experiments.py -n "./results/AAAI/${name}_last" -j --output_file "./results/AAAI/${name}_last.txt" --min_variance 0.01

##############################################################################################################################

name="BotBeh"

python main.py -n "./results/AAAI/$name" -j -h --hf './human_data/inferno_formatted.pickle'

python experiments.py -n "./results/AAAI/$name" -j --output_file "./results/AAAI/${name}.txt"

python experiments.py -n "./results/AAAI/${name}_last" -j --output_file "./results/AAAI/${name}_last.txt"

##############################################################################################################################

name="HuBeh"

python main.py -n "./results/AAAI/$name" -j -h --hf './human_data/human_formatted.pickle'

python experiments.py -n "./results/AAAI/$name" -j --output_file "./results/AAAI/${name}.txt"

python experiments.py -n "./results/AAAI/${name}_last" -j --output_file "./results/AAAI/${name}_last.txt" 