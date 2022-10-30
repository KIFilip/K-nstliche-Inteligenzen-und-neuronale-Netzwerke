import os
import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy



# vars
environment = "CartPole-v1"
log_path = os.path.join("Training", "Logs")
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_model')
training_length = 20000
training_episodes = 20
# create environment
env = gym.make(environment)
model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_path)
# train model
model.learn(total_timesteps=training_length, progress_bar=False)
# save trained model
model.save(PPO_path)
del model
saved_model = PPO.load(PPO_path, env=env)
# evaluate saved model
evaluation = evaluate_policy(saved_model, env, n_eval_episodes=10, render=False)
print(evaluation)
# test model
for episode in range(0, training_episodes):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action, states = saved_model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
    print(episode, score)
env.close
