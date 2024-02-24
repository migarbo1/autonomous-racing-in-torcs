# echo 'launching train 3 frame stacking'
# python main.py 3
# python experiments.py

# mv weights/ppo_actor.pth weights/ppo_actor_fs_3.pth
# mv weights/ppo_critic.pth weights/ppo_critic_fs_3.pth
# mv training_data.pickle training_data_fs_3.pickle
# mv results.txt results/results_fs_3.txt

# echo 'launching train 5 frame stacking'
# python main.py 5
# python experiments.py

# mv weights/ppo_actor.pth weights/ppo_actor_fs_5.pth
# mv weights/ppo_critic.pth weights/ppo_critic_fs_5.pth
# mv training_data.pickle training_data_fs_5.pickle
# mv results.txt results/results_fs_5.txt

# echo 'launching train baseline'
# python main.py 1
# python experiments.py

# mv weights/ppo_actor.pth weights/ppo_actor_baseline.pth
# mv weights/ppo_critic.pth weights/ppo_critic_baseline.pth
# mv training_data.pickle training_data_baseline.pickle
# mv results.txt results/results_baseline.txt


echo 'launching eval with 3 frame stacking'
mv weights/ppo_actor_fs_3.pth weights/ppo_actor.pth
mv weights/ppo_critic_fs_3.pth weights/ppo_critic.pth
python experiments.py 3
mv weights/ppo_actor.pth weights/ppo_actor_fs_3.pth
mv weights/ppo_critic.pth weights/ppo_critic_fs_3.pth
mv results.txt results/results_fs_3.txt

echo 'launching eval for baseline'
mv weights/ppo_actor_baseline.pth weights/ppo_actor.pth
mv weights/ppo_critic_baseline.pth weights/ppo_critic.pth
python experiments.py 1
mv weights/ppo_actor.pth weights/ppo_actor_baseline.pth
mv weights/ppo_critic.pth weights/ppo_critic_baseline.pth
mv results.txt results/results_baseline.txt
