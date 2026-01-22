# experiment_6_reinforce_advanced.py
# REINFORCE (Monte Carlo Policy Gradient) with Baseline and Action Masking

import torch
from Policy_Gradient import train_reinforce_with_baseline, evaluate_policy_advanced
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("=" * 80)
    print("REINFORCE (Monte Carlo Policy Gradient) with Action Masking")
    print("=" * 80)
    print(f"Device: {device}")
    print("\nÖzellikler:")
    print("  ✅ Monte Carlo Returns (G_t = sum gamma^(k-t) * r_k)")
    print("  ✅ Baseline (Value Network) - variance reduction")
    print("  ✅ Action Masking - illegal moves prevented")
    print("  ✅ Curriculum Learning:")
    print("     * Stage 1 (0-3000): max_spawn_tile=128")
    print("     * Stage 2 (3000-7000): max_spawn_tile=256")
    print("     * Stage 3 (7000+): Normal 2048")
    print("  ✅ Reward Structure:")
    print("     * Base: merge_value/2048 + empty_cells*0.01")
    print("     * Terminal: game_over=-1.0, max_tile>=1024=+1.0")
    print("  ✅ Network: 20→256→256→4 (Feedforward only, no LSTM)")
    print("  ✅ Optimizer: Adam(lr=1e-4), grad_clip=0.5")
    print("  ✅ Evaluation: Greedy (argmax), no sampling")
    print("=" * 80)
    
  
    policy_net, value_net, training_scores, eval_results = train_reinforce_with_baseline(
        num_episodes=10000,        
        gamma=0.99,                
        lr=1e-4,                   
        print_every=100,
        eval_every=1000,           
        device=device,
        entropy_coef=0.01,         
        value_coef=0.5,            
        max_steps_per_episode=3000,
        max_grad_norm=0.5,         
        checkpoint_dir="checkpoints_reinforce_advanced",
        curriculum_stages=[
            (0, 3000, 128),        
            (3000, 7000, 256),      
            (7000, None, None),    
        ],
    )
    print("\n" + "=" * 80)
    print("FINAL EVALUATION (100 episodes, greedy)")
    print("=" * 80)
    final_eval_scores, final_best_tile, final_1024_count, final_2048_count = evaluate_policy_advanced(
        policy_net, num_episodes=100, device=device, use_lstm=False
    )
    final_avg_score = sum(final_eval_scores) / len(final_eval_scores)
    final_best_score = max(final_eval_scores)
    
    print(f"Final Average Score: {final_avg_score:.1f}")
    print(f"Final Best Score: {final_best_score}")
    print(f"Final Best Tile: {final_best_tile}")
    print(f"1024 Count: {final_1024_count}/100")
    print(f"2048 Count: {final_2048_count}/100")
    print("=" * 80)
    
  
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
        plt.title('REINFORCE Advanced - Training Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig('reinforce_advanced_training_curve.png', dpi=150, bbox_inches='tight')
        print("\n✅ Training curve saved: reinforce_advanced_training_curve.png")
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
        axes[1, 0].set_ylabel('Count (out of 100)')
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
        plt.savefig('reinforce_advanced_evaluation_curve.png', dpi=150, bbox_inches='tight')
        print("✅ Evaluation curve saved: reinforce_advanced_evaluation_curve.png")
        plt.close()
    
    print("\n✅ Training and evaluation complete!")

