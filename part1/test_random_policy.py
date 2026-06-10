import gymnasium as gym

def main():
    render = False

    if render:
        env = gym.make('Hopper-v4', render_mode='human')
    else:
        env = gym.make('Hopper-v4', render_mode='rgb_array')
    print('State space:', env.observation_space)  
    print('Action space:', env.action_space)  

    model = env.unwrapped.model

    print("\n--- MuJoCo model info ---")
    print("Body names:", [model.body(i).name for i in range(model.nbody)])
    print("Body masses:", model.body_mass)
    print("Number of DoFs:", model.nv)
    print("DoFs per body:", model.body_dofnum)
    print("Number of actuators:", model.nu)

    n_episodes = 50

    for ep in range(n_episodes):  
        done = False
        state, info = env.reset()  

        while not done:  
            action = env.action_space.sample()  

            state, reward, terminated, truncated, _ = env.step(action)  
            done = terminated or truncated

            if render:
                env.render()


if __name__ == '__main__':
    main()