import time
import os
import warnings

# Suppress Gym deprecation warnings
warnings.filterwarnings("ignore", message="Gym has been unmaintained")
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
import gym
import numpy as np 
from torch.utils.tensorboard import SummaryWriter 
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3_torch import Agent

if __name__ == '__main__':

    if not os.path.exists('tmp/td3'):
        os.makedirs('tmp/td3')
    
    env_name = "Door"

    env = suite.make(
        env_name,
        robots=['Panda'],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=True,
        use_camera_obs=False,
        horizon=300,
        render_camera="frontview",
        has_offscreen_renderer=True,
        reward_shaping=True,
        control_freq=20,
    )

    
    env = GymWrapper(env)


    actor_learning_rate = 0.0003
    critic_learning_rate = 0.0003
    batch_size = 256
    layer_1_size = 256
    layer_2_size = 128

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, env=env, input_dims=env.observation_space.shape, tau=0.005, gamma=0.99, update_actor_interval=2, warmup=1000, n_actions=env.action_space.shape[0], layer1_size=layer_1_size, layer2_size=layer_2_size, batch_size=batch_size)


    n_games = 3
    best_score = 0
    episode_identifier = f"1 - actor_learning_rate: {actor_learning_rate}, critic_learning_rate: {critic_learning_rate}, layer_1_size: {layer_1_size}, layer_2_size: {layer_2_size}"
    
            
    agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0 
        
        while not done:
            action = agent.choose_action(observation, validation=True )
            next_observation, reward, done, info = env.step(action)
            env.render()
            time.sleep(0.03)
            score += reward
            observation = next_observation
        
            
        print(f'Episode: {i}, Score: {score}')
