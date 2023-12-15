import subprocess

hidden = 7
learning_rates = [.5,.6,.7,.8,.9,1]
w_range = [.4,.5,.6]
num_particles_range = [50,60]
for learning_rate in learning_rates:
     for w in w_range:
          for num_particles in num_particles_range:
               print("Num particles: ", num_particles)
               print("w: ", w)
               subprocess.run(["python", "train_bpnn_nnpso.py", str(hidden), str(learning_rate), str(num_particles), str(w)])
