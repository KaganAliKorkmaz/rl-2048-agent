# experiment_7_reinforce_lstm.py
# REINFORCE (Monte Carlo Policy Gradient) with LSTM + Baseline

import torch
from Policy_Gradient import train_reinforce_with_baseline_lstm, evaluate_policy_advanced
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    policy_net, value_net, training_scores, eval_results, training_metrics = train_reinforce_with_baseline_lstm(
        num_episodes=10000,
        gamma=0.99,
        lr=1e-4,
        print_every=10,
        eval_every=1000,
        device=device,
        entropy_coef=0.01,
        value_coef=0.5,
        max_steps_per_episode=3000,
        max_grad_norm=1.0,
        checkpoint_dir="checkpoints_reinforce_lstm",
        truncated_bptt=20,
        curriculum_stages=[
            (0, 3000, 128),
            (3000, 7000, 256),
            (7000, None, None),
        ],
    )
    
    print("\nFinal Evaluation Results:")
    final_eval_scores, final_best_tile, final_1024_count, final_2048_count = evaluate_policy_advanced(
        policy_net, num_episodes=100, device=device, use_lstm=True
    )
    final_avg_score = sum(final_eval_scores) / len(final_eval_scores)
    final_best_score = max(final_eval_scores)
    
    print(f"Average Score: {final_avg_score:.1f}")
    print(f"Best Score: {final_best_score}")
    print(f"Best Tile: {final_best_tile}")
    print(f"1024 Count: {final_1024_count}/100")
    print(f"2048 Count: {final_2048_count}/100")
    
    if len(training_scores) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(training_scores, alpha=0.3, label='Raw Scores', color='blue')
        
        window = 100
        if len(training_scores) >= window:
            smoothed = []
            for i in range(len(training_scores)):
                start = max(0, i - window + 1)
                smoothed.append(sum(training_scores[start:i+1]) / (i - start + 1))
            plt.plot(smoothed, label=f'Smoothed (window={window})', color='red', linewidth=2)
        
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('REINFORCE LSTM - Training Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig('reinforce_lstm_training_curve.png', dpi=150, bbox_inches='tight')
        print("Training curve saved: reinforce_lstm_training_curve.png")
        plt.close()
    
    if len(training_metrics) > 0:
        episodes = [m['episode'] for m in training_metrics]
        hidden_norms = [m['hidden_norm'] for m in training_metrics]
        entropies = [m['entropy'] for m in training_metrics]
        policy_losses = [m['policy_loss'] for m in training_metrics]
        value_losses = [m['value_loss'] for m in training_metrics]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(hidden_norms, alpha=0.7, color='purple')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('LSTM Hidden Norm')
        axes[0, 0].set_title('LSTM Hidden State Norm')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(entropies, alpha=0.7, color='green')
        axes[0, 1].axhline(y=0.1, color='red', linestyle='--', label='Threshold')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Policy Entropy')
        axes[0, 1].set_title('Policy Entropy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(policy_losses, alpha=0.7, color='blue')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Policy Loss')
        axes[1, 0].set_title('Policy Loss')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(value_losses, alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Value Loss')
        axes[1, 1].set_title('Value Loss')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('reinforce_lstm_metrics.png', dpi=150, bbox_inches='tight')
        print("LSTM metrics saved: reinforce_lstm_metrics.png")
        plt.close()
    
    if len(eval_results) > 0:
        episodes = [r['episode'] for r in eval_results]
        avg_scores = [r['avg_score'] for r in eval_results]
        best_scores = [r['best_score'] for r in eval_results]
        best_tiles = [r['best_tile'] for r in eval_results]
        count_1024 = [r['1024_count'] for r in eval_results]
        count_2048 = [r['2048_count'] for r in eval_results]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(episodes, avg_scores, marker='o', label='Avg Score', color='blue')
        axes[0, 0].plot(episodes, best_scores, marker='s', label='Best Score', color='red')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Evaluation Scores')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(episodes, best_tiles, marker='o', color='green')
        axes[0, 1].axhline(y=1024, color='orange', linestyle='--', label='1024 Target')
        axes[0, 1].axhline(y=2048, color='red', linestyle='--', label='2048 Target')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Best Tile')
        axes[0, 1].set_title('Best Tile Reached')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(episodes, count_1024, marker='o', label='1024 Count', color='orange')
        axes[1, 0].plot(episodes, count_2048, marker='s', label='2048 Count', color='red')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Tile Achievement Counts')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(episodes, avg_scores, marker='o', label='Avg Score', alpha=0.7)
        axes[1, 1].plot(episodes, best_tiles, marker='s', label='Best Tile', alpha=0.7)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Combined View')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('reinforce_lstm_evaluation_curve.png', dpi=150, bbox_inches='tight')
        print("Evaluation curve saved: reinforce_lstm_evaluation_curve.png")
        plt.close()
    
    print("\nTraining complete")
