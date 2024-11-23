from stable_baselines3 import DQN
from dino_env import DinoGameEnv  # Import the DinoGameEnv class

env = DinoGameEnv()

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save and test
model.save("dino_rl_model")
obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
