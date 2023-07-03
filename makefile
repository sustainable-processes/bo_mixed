mobo_experiment_noise_0:
	python main.py --repeats 20 --num-initial-experiments 25 --max-iterations 20 --noise-level 0.0
mobo_experiment_noise_5:
	python main.py --repeats 20 --num-initial-experiments 25 --max-iterations 20 --noise-level 5.0
mobo_experiments: mobo_experiment_noise_0 mobo_experiment_noise_5