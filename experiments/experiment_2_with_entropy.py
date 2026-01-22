import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.Policy_Gradient import (
    train_reinforce,
    evaluate_policy,
    plot_average_score,
    plot_evaluation_scores
)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("=" * 80)
    print("Experiment 2: REINFORCE (With Entropy)")
    print("=" * 80)
    print(f"Using device: {device}")
    print(f"Parameters: entropy_coef=5e-4")
    print("=" * 80)
    policy_net, scores = train_reinforce(
        num_episodes=2000,
        lr=1e-3,
        gamma=0.99,
        entropy_coef=5e-4,
        baseline_momentum=0.05,
        max_steps_per_episode=2000
    )
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    print("\nGenerating training plot...")
    plot_average_score(scores, window=100, smoothing_factor=0.9, save_path=os.path.join(results_dir, "experiment_2_training.png"))
    print("\nRunning evaluation...")
    eval_scores, eval_best_tile = evaluate_policy(policy_net, num_episodes=50, device=device)
    print("\nGenerating evaluation plot...")
    plot_evaluation_scores(eval_scores, window=10, smoothing_factor=0.9, save_path=os.path.join(results_dir, "experiment_2_evaluation.png"))
    print("\n" + "=" * 80)
    print("Experiment 2 Results:")
    print("=" * 80)
    print(f"Average evaluation score: {sum(eval_scores)/len(eval_scores):.1f}")
    print(f"Best evaluation score: {max(eval_scores)}")
    print("=" * 80)

