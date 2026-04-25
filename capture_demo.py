import imageio
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3_torch import Agent
import os

if __name__ == '__main__':
    # Create assets directory
    if not os.path.exists('assets'):
        os.makedirs('assets')
    
    # Copy the training graph (Hardcoded path from user upload)
    src_graph = "C:/Users/omkar/.gemini/antigravity/brain/76a8a41f-1b47-4eec-842b-82bd64abf900/uploaded_image_1766981168998.png"
    dst_graph = "assets/training_graph.png"
    import shutil
    try:
        shutil.copy(src_graph, dst_graph)
        print("Copied training graph to assets/training_graph.png")
    except Exception as e:
        print(f"Could not copy graph: {e}")

    env_name = "Door"

    # Define the environment with OFFSCREEN rendering enabled for video capture
    env = suite.make(
        env_name,
        robots=['Panda'],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=False,             # Disable on-screen
        has_offscreen_renderer=True,    # Enable off-screen
        use_camera_obs=False,           # Disable camera observations (agent doesn't need pixels)
        use_object_obs=True,
        render_camera="frontview",      # Camera to record
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)

    # Hyperparameters
    actor_learning_rate = 0.0003
    critic_learning_rate = 0.0003
    batch_size = 256
    layer_1_size = 256
    layer_2_size = 128

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, env=env, input_dims=env.observation_space.shape, tau=0.005, gamma=0.99, update_actor_interval=2, warmup=1000, n_actions=env.action_space.shape[0], layer1_size=layer_1_size, layer2_size=layer_2_size, batch_size=batch_size)

    print("Loading trained models...")
    agent.load_models()

    print("Capturing video...")
    frames = []
    
    observation = env.reset()
    done = False
    
    # Run for one episode (Trimmed to 3 seconds = 60 frames @ 20fps)
    for i in range(60): 
        # Pure policy (validation=True)
        action = agent.choose_action(observation, validation=True)
        next_observation, reward, done, info = env.step(action)
        
        frame = env.sim.render(
            camera_name="frontview", 
            width=512, 
            height=512, 
            depth=False
        )
        frame = np.flipud(frame)
        frames.append(frame)

        observation = next_observation
        if done:
            break

    # Write frames twice to repeat the action
    video_writer = imageio.get_writer('assets/demo.gif', fps=20, loop=0)
    for _ in range(2):
        for frame in frames:
            video_writer.append_data(frame)

    video_writer.close()
    print("Video saved to assets/demo.gif")
