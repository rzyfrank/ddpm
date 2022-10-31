import argparse
import random


def obtain_args():
    parser = argparse.ArgumentParser(description='DDPM')
    parser.add_argument('--rand_seed', type=int)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--print_freq', type=int)
    parser.add_argument('--save_freq', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--dim', type=int)
    parser.add_argument('--timesteps', type=int)
    parser.add_argument('--beta_schedule', type=str)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--eval_freq', type=int)
    parser.add_argument('--channels', type=int)
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--learning_rate', type=float)



    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 10000)

    return args
