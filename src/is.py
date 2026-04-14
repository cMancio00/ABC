from simulator import GenerativeProcess, BinomialMixtureSimulator
from utils.statistics import calculate_frequency

N_SAMPLES = 100

process = GenerativeProcess(theta1=0.2, theta2=0.7, rate=0.1)
obs = process.mixture.sample((N_SAMPLES,))

simulator = BinomialMixtureSimulator()
sim = simulator.generate(0.2, 0.7, 0.1, times=N_SAMPLES)

# proposal = torch.distributions.beta

print(calculate_frequency(obs))
print(calculate_frequency(sim))
