# ##############################################################

# name="ppo_HuC"

# python main.py -n "./tfm/$name" -j -f 5 --focus

# python experiments.py -n "./tfm/$name" -j --output_file "./tfm/${name}.txt" -f 5 --focus

# python experiments.py -n "./tfm/${name}_last" -j --output_file "./tfm/${name}_last.txt" -f 5 --focus

# ##############################################################

# name="ppo_HuBe"

# python main.py -n "./tfm/$name" -j -h --hf "./human_data/formatted_fs1.pickle"

# python experiments.py -n "./tfm/$name" -j --output_file "./tfm/${name}.txt" 

# python experiments.py -n "./tfm/${name}_last" -j --output_file "./tfm/${name}_last.txt" 

# ##############################################################

name="ppo_HuBeC"

python main.py -n "./tfm/$name" -j -f 5 -h --focus --hf "./human_data/formatted_fs5.pickle"

python experiments.py -n "./tfm/$name" -j --output_file "./tfm/${name}.txt" -f 5 --focus

python experiments.py -n "./tfm/${name}_last" -j --output_file "./tfm/${name}_last.txt" -f 5 --focus
