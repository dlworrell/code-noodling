Project: code-noodling

Prime generation + physical dice simulation for tabletop/RPG workflows.

Features
	•	Multi-GPU CUDA sieve (cuda_sieve_mgpu) — segmented, odd-only, √N base primes on CPU, optimized for Tesla K80 (sm_37), JSON output.
	•	Unbiased CPU dice engine (dice_cpu) — any NdM+K, no modulo bias, optional prime-seeded determinism, JSON/CSV logs, chi-square.
	•	PhysX dice simulator (physx_dice_multi) — fully physical D6/D8/D12/D20, auto hardware tuning, optional dice chute geometry (--chute), PhysX Visual Debugger streaming (--pvd), prime seeding, JSON/CSV logs, chi-square.
