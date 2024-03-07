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

echo 'launching train baseline with focus and human data'
#python main.py 5 True
#echo 'main completed'
python experiments.py 5 True

mv weights/ppo_actor.pth weights/ppo_actor_focus_fs5_HD_new.pth
mv weights/ppo_critic.pth weights/ppo_critic_focus_fs5_HD_new.pth
mv training_data.pickle training_data_focus_fs5_HD_new.pickle
mv results.txt results/results_focus_fs5_HD_new.txt

cp weights/ppo_actor_last.pth weights/ppo_actor.pth
cp weights/ppo_critic_last.pth weights/ppo_critic.pth
python experiments.py 5 True
mv results.txt results/results_focus_fs5_HD_new_last.txt


# echo 'launching train baseline with focus and human data'
# python main.py 1 True
# echo 'main completed'
# python experiments.py 1 True

# mv weights/ppo_actor.pth weights/ppo_actor_baseline_focus_HD.pth
# mv weights/ppo_critic.pth weights/ppo_critic_baseline_focus_HD.pth
# mv training_data.pickle training_data_baseline_focus_HD.pickle
# mv results.txt results/results_baseline_focus_HD.txt

# echo 'launching eval with 3 frame stacking'
# mv weights/ppo_actor_fs_3.pth weights/ppo_actor.pth
# mv weights/ppo_critic_fs_3.pth weights/ppo_critic.pth
# python experiments.py 3
# mv weights/ppo_actor.pth weights/ppo_actor_fs_3.pth
# mv weights/ppo_critic.pth weights/ppo_critic_fs_3.pth
# mv results.txt results/results_fs_3.txt

# echo 'launching eval for baseline focus + human data'
# mv weights/ppo_actor_baseline_focus_HD.pth weights/ppo_actor.pth
# mv weights/ppo_critic_baseline_focus_HD.pth weights/ppo_critic.pth
# python experiments.py 1 True
# mv weights/ppo_actor.pth weights/ppo_actor_baseline_focus_HD.pth
# mv weights/ppo_critic.pth weights/ppo_critic_baseline_focus_HD.pth
# mv results.txt results/results_baseline_focus_HD.txt

# echo 'launching eval for baseline: only focus'
# mv weights/ppo_actor_baseline_focus.pth weights/ppo_actor.pth
# mv weights/ppo_critic_baseline_focus.pth weights/ppo_critic.pth
# python experiments.py 1 False
# mv weights/ppo_actor.pth weights/ppo_actor_baseline_focus.pth
# mv weights/ppo_critic.pth weights/ppo_critic_baseline_focus.pth
# mv results.txt results/results_baseline_focus.txt

# echo 'train focused framestacking with 3 frames'
# #python main.py 3
# python experiments.py 3

# mv weights/ppo_actor.pth weights/ppo_actor_focus_fs3.pth
# mv weights/ppo_critic.pth weights/ppo_critic_focus_fs3.pth
# mv training_data.pickle training_data_focus_fs3.pickle
# mv results.txt results/results_focus_fs3.txt
