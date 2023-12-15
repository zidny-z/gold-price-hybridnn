import subprocess

hidden = 7
learning_rate = .1
num_particles_range = [10,20,30,40,50,60,70,80,90,100]
w = .5


for num_particles in num_particles_range:
     print("Num particles: ", num_particles)
     subprocess.run(["python", "train_nnpso_particle.py", str(hidden), str(learning_rate), str(num_particles), str(w)])
          