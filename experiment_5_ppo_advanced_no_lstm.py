# PPO Advanced

import torch
import json
from Policy_Gradient import (
    train_ppo_advanced,
    evaluate_policy_advanced,
    plot_average_score,
    plot_evaluation_scores
)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("=" * 80)
    print("PPO ADVANCED (NO LSTM) - PPO with GAE(λ)")
    print("=" * 80)
    print(f"Using device: {device}")
    print()
    print("Özellikler:")
    print("  ✅ PPO with Feedforward Networks (PolicyNet256, ValueNet256)")
    print("  ✅ GAE(λ) advantage estimation (λ=0.97, gamma=0.99)")
    print("  ✅ Action masking compatible")
    print("  ✅ Milestone rewards (512/1024/2048)")
    print("=" * 80)
    print()
    
  
    policy_net, value_net, training_scores, eval_results = train_ppo_advanced(
        num_episodes=10000,
        gamma=0.99,
        lr=5e-5,
        print_every=100,
        eval_every=1000,
        device=device,
        entropy_coef=0.01,
        value_coef=0.5,
        max_steps_per_episode=3000,
        clip_epsilon=0.3,
        ppo_epochs=3,
        batch_size=256,
        gae_lambda=0.97,
        max_grad_norm=1.0,
        use_large_network=True,
        use_lstm=False,
        lstm_hidden=256,
        checkpoint_dir="checkpoints_ppo_no_lstm",
        enable_burst=True,
        burst_interval=1000,
        burst_duration=200,
        burst_entropy_multiplier=2.0,
        burst_clip_epsilon=None,
    )
    
    print("\n" + "=" * 80)
    print("Training Complete! Generating Final Evaluation and Reports...")
    print("=" * 80)
    print("\nRunning final evaluation (100 episodes)...")
    final_eval_scores, final_best_tile, final_1024_count, final_2048_count = evaluate_policy_advanced(
        policy_net, num_episodes=100, device=device, use_lstm=False
    )
    print("\nGenerating training plot...")
    plot_average_score(
        training_scores, 
        window=200, 
        smoothing_factor=0.9, 
        save_path="experiment_5_ppo_advanced_no_lstm_training.png"
    )
    if eval_results:
        eval_scores_list = [r['avg_score'] for r in eval_results]
        eval_episodes = [r['episode'] for r in eval_results]
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(eval_episodes, eval_scores_list, 'o-', label='Avg Evaluation Score')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('PPO Advanced (NO LSTM) - Evaluation Average Score')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        eval_tiles = [r['best_tile'] for r in eval_results]
        plt.plot(eval_episodes, eval_tiles, 's-', color='orange', label='Best Tile')
        plt.axhline(y=512, color='g', linestyle='--', alpha=0.5, label='512')
        plt.axhline(y=1024, color='b', linestyle='--', alpha=0.5, label='1024')
        plt.axhline(y=2048, color='r', linestyle='--', alpha=0.5, label='2048')
        plt.xlabel('Episode')
        plt.ylabel('Best Tile')
        plt.title('PPO Advanced (NO LSTM) - Best Tile Reached')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("experiment_5_ppo_advanced_no_lstm_evaluation.png", bbox_inches="tight")
        print("Saved evaluation plot to experiment_5_ppo_advanced_no_lstm_evaluation.png")
        plt.close()
    
  
    plot_evaluation_scores(
        final_eval_scores, 
        window=10, 
        smoothing_factor=0.9, 
        save_path="experiment_5_ppo_advanced_no_lstm_final_eval.png"
    )
    
  
    print("\n" + "=" * 80)
    print("FINAL RESULTS REPORT (NO LSTM)")
    print("=" * 80)
    print(f"\nTraining:")
    print(f"  - Total episodes: 10000")
    print(f"  - Best training score: {max(training_scores)}")
    print(f"  - Final 200-episode average: {sum(training_scores[-200:])/len(training_scores[-200:]):.1f}")
    
    print(f"\nFinal Evaluation (100 episodes):")
    print(f"  - Average score: {sum(final_eval_scores)/len(final_eval_scores):.1f}")
    print(f"  - Best score: {max(final_eval_scores)}")
    print(f"  - Best tile: {final_best_tile}")
    print(f"  - 1024 tiles reached: {final_1024_count}/100 ({final_1024_count}%)")
    print(f"  - 2048 tiles reached: {final_2048_count}/100 ({final_2048_count}%)")
    
    if eval_results:
        print(f"\nPeriodic Evaluations:")
        for result in eval_results:
            print(f"  Episode {result['episode']:5d}: "
                  f"Avg={result['avg_score']:6.1f}, "
                  f"Best={result['best_score']:6.0f}, "
                  f"Tile={result['best_tile']:4.0f}, "
                  f"1024={result.get('1024_count', 0):2d}, "
                  f"2048={result.get('2048_count', 0):2d}")
    
  
    report = {
        'training': {
            'total_episodes': 10000,
            'best_score': max(training_scores),
            'final_avg_score': sum(training_scores[-200:])/len(training_scores[-200:]),
        },
        'final_evaluation': {
            'num_episodes': 100,
            'avg_score': sum(final_eval_scores)/len(final_eval_scores),
            'best_score': max(final_eval_scores),
            'best_tile': final_best_tile,
            'tiles_1024_count': final_1024_count,
            'tiles_2048_count': final_2048_count,
        },
        'periodic_evaluations': eval_results,
        'architecture': 'Feedforward (NO LSTM)',
        'network': 'PolicyNet256 + ValueNet256'
    }
    
    with open('experiment_5_ppo_advanced_no_lstm_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Saved detailed report to experiment_5_ppo_advanced_no_lstm_report.json")
    
  
    print("\n" + "=" * 80)
    if final_best_tile >= 2048:
        print("🎉🎉🎉 SUCCESS! Reached the 2048 tile! 🎉🎉🎉")
    elif final_best_tile >= 1024:
        print(f"✅ Good progress! Reached {final_best_tile} tile.")
        print("   Continue training or adjust hyperparameters to reach 2048.")
    else:
        print(f"📈 Progress: Reached {final_best_tile} tile.")
        print("   Consider longer training or hyperparameter tuning.")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("COMPARISON NOTE:")
    print("=" * 80)
    print("This experiment uses Feedforward Networks (NO LSTM).")
    print("Compare results with experiment_5_ppo_advanced.py (LSTM version)")
    print("to see the impact of LSTM on 2048 game performance.")
    print("=" * 80)

