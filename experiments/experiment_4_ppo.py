import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.Policy_Gradient import train_ppo_advanced, evaluate_policy, plot_average_score, plot_evaluation_scores

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("=" * 80)
    print("Experiment 4: PPO (Proximal Policy Optimization)")
    print("=" * 80)
    print(f"Using device: {device}")
    print()
    
    policy_net, value_net, training_scores, eval_results = train_ppo_advanced(
        num_episodes=5000,
        gamma=0.99,
        lr=3e-4,
        print_every=50,
        eval_every=500,
        device=device,
        entropy_coef=0.01,
        value_coef=0.5,
        max_steps_per_episode=3000,
        clip_epsilon=0.2,
        ppo_epochs=4,
        batch_size=None,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        use_large_network=False,
        use_lstm=False,
        checkpoint_dir=os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'ppo'),
        enable_burst=False,
    )
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Training Complete! Starting Evaluation...")
    print("=" * 80)
    plot_average_score(
        training_scores, 
        window=100, 
        smoothing_factor=0.9, 
        save_path=os.path.join(results_dir, "experiment_4_ppo_training.png")
    )
    eval_scores, eval_best_tile = evaluate_policy(policy_net, num_episodes=50, device=device)
    plot_evaluation_scores(
        eval_scores, 
        window=10, 
        smoothing_factor=0.9, 
        save_path=os.path.join(results_dir, "experiment_4_ppo_evaluation.png")
    )
    
    print("\n" + "=" * 80)
    print("Experiment 4 Complete!")
    print("=" * 80)
    print(f"\nFinal Results:")
    print(f"  - Training episodes: 5000")
    print(f"  - Evaluation episodes: 50")
    print(f"  - Average evaluation score: {sum(eval_scores)/len(eval_scores):.1f}")
    print(f"  - Best evaluation score: {max(eval_scores)}")
    print(f"  - Best tile reached: {eval_best_tile}")
    
    if eval_best_tile >= 2048:
        print("\nSUCCESS! Reached the 2048 tile!")
    elif eval_best_tile >= 1024:
        print(f"\nClose! Reached {eval_best_tile}, but not quite 2048 yet.")
        print("Try increasing num_episodes or adjusting hyperparameters.")
    else:
        print(f"\nProgress: Reached {eval_best_tile} tile.")
        print("Continue training or adjust hyperparameters to reach 2048.")
    
    print("\n" + "=" * 80)

