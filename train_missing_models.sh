git checkout f7180d1
mv torcs_python_client/torcs_env.py torcs_python_client/torcs_env_o.py
mv torcs_python_client/torcs_env_noRewReg.py torcs_python_client/torcs_env.py
python main.py
mv training_data.pickle training_data_RewReg.pickle
mv weights/ppo_actor.pth weights/ppo_actor_RewReg.pth
mv weights/ppo_critic.pth weights/ppo_critic_RewReg.pth

git checkout e455ff3 
mv torcs_python_client/torcs_env.py torcs_python_client/torcs_env_noRewReg.py
mv torcs_python_client/torcs_env_frame_stacking_3.py torcs_python_client/torcs_env.py
python main.py
mv training_data.pickle training_data_frame_stacking_3.pickle
mv weights/ppo_actor.pth weights/ppo_actor_frame_stacking_3.pth
mv weights/ppo_critic.pth weights/ppo_critic_frame_stacking_3.pth

mv torcs_python_client/torcs_env_o.py torcs_python_client/torcs_env.py
git checkout master