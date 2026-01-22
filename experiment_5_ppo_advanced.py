# PPO Advanced

import torch
import json
from Policy_Gradient import (
    train_ppo_lstm,
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
    print("PPO ADVANCED - PPO-LSTM with GAE(λ) and KL Guard")
    print("=" * 80)
    print(f"Using device: {device}")
    print()
    print("Özellikler:")
    print("  ✅ PPO-LSTM with LayerNormLSTMCell (hidden state stabilization)")
    print("  ✅ GAE(λ) advantage estimation (λ=0.95, gamma=0.99)")
    print("  ✅ Sequence-aware minibatch sampling (recurrent-aware)")
    print("  ✅ KL guard (early stop if KL > 0.01)")
    print("  ✅ Action masking compatible")
    print("  ✅ Curriculum learning (stages 1/2/3)")
    print("  ✅ Milestone rewards (512/1024/2048)")
    print("  ✅ Value clipping (value_clip_epsilon=0.2)")
    print("  ✅ Advantage clipping (-5.0, 5.0)")
    print("=" * 80)
    print()
    policy_net, value_net, training_scores, eval_results = train_ppo_lstm(
        num_episodes=10000,
        gamma=0.99,
        lam=0.97,
        lr=5e-5,
        ppo_epochs=3,
        clip_epsilon=0.3,
        value_clip_epsilon=0.2,
        vf_coef=0.4,
        ent_coef=0.01,
        max_grad_norm=1.0,
        rollout_len=512,
        seq_len=20,
        minibatch_size=128,
        print_every=100,
        eval_every=1000,
        device=device,
        max_steps_per_episode=3000,
        checkpoint_dir="checkpoints_ppo_lstm",
            curriculum_stages=[
                (0, 4000, 128),    
                (4000, 8000, 256), 
                (8000, None, None),
            ],
        seed=42,
    )
    
    print("\n" + "=" * 80)
    print("Training Complete! Generating Final Evaluation and Reports...")
    print("=" * 80)
    print("\nRunning final evaluation (100 episodes)...")
    final_eval_scores, final_best_tile, final_1024_count, final_2048_count = evaluate_policy_advanced(
        policy_net, num_episodes=100, device=device, use_lstm=True
    )
    print("\nGenerating training plot...")
    plot_average_score(
        training_scores, 
        window=200, 
        smoothing_factor=0.9, 
        save_path="experiment_5_ppo_advanced_training.png"
    )
    if eval_results:
        eval_scores_list = [r['avg_score'] for r in eval_results]
        eval_episodes = [r['episode'] for r in eval_results]
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(eval_episodes, eval_scores_list, 'o-', label='Avg Evaluation Score')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('PPO Advanced - Evaluation Average Score')
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
        plt.title('PPO Advanced - Best Tile Reached')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("experiment_5_ppo_advanced_evaluation.png", bbox_inches="tight")
        print("Saved evaluation plot to experiment_5_ppo_advanced_evaluation.png")
        plt.close()
    
  
    plot_evaluation_scores(
        final_eval_scores, 
        window=10, 
        smoothing_factor=0.9, 
        save_path="experiment_5_ppo_advanced_final_eval.png"
    )
    
  
    print("\n" + "=" * 80)
    print("FINAL RESULTS REPORT")
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
        'periodic_evaluations': eval_results
    }
    
    with open('experiment_5_ppo_advanced_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Saved detailed report to experiment_5_ppo_advanced_report.json")
    
  
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
