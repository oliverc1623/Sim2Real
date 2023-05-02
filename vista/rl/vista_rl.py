import argparse
# from learn import Memory, Learner
# from learn_lstm import Learner
from learn import Learner
# from learn_reward_upgrade import Learner
import matplotlib.pyplot as plt

def main(args):
    print(args.neuralnetwork)
    learner = Learner(args.neuralnetwork, args.learning_rate, args.episodes, args.clip, args.animate)
    learner.learn()
    # learner.save()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="VISTA Deep Reinforcement Learner")
    parser.add_argument("-nn", "--neuralnetwork", required=True)
    parser.add_argument("-lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("-e", "--episodes", default=500, type=int)
    parser.add_argument("-c", "--clip", default=2, type=int)
    parser.add_argument("-a", "--animate", default=False, type=bool)
    args = parser.parse_args()
    main(args)

