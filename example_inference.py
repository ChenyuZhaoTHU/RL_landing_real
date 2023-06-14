import torch
from rocket import Rocket
from policy import ActorCritic
import os
import glob
import matplotlib.pyplot as plt
# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    task = 'landing'  # 'hover' or 'landing'
    max_steps = 800
    ckpt_dir = glob.glob(os.path.join(task+'_ckpt', '*.pt'))[-1]  # last ckpt

    env = Rocket(task=task, max_steps=max_steps)
    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    if os.path.exists(ckpt_dir):
        checkpoint = torch.load(ckpt_dir)
        net.load_state_dict(checkpoint['model_G_state_dict'])

    state = env.reset()

    z = []
    z_d = []
    t = []
    for step_id in range(max_steps):
        action, log_prob, value = net.get_action(state)
        state, reward, done, _ = env.step(action)
        print(state[1])
        t.append(step_id)
        z.append(state[1]*100)
        z_d.append(env.z_d)
        import time
        time.sleep(0.1)
        env.render(window_name='test')
        if env.already_crash:
            break
    plt.figure()
    plt.plot(t, z)
    plt.plot(t, z_d)
    plt.xlabel('time')
    plt.xlabel('height')
    plt.legend()
    plt.show()
    # plt.savefig('suc_1.png')
