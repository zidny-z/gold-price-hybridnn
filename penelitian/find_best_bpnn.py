import subprocess

hidden_range = [2,3,4,5,6,7,8]
learning_rate_range = [0.5,0.6,0.7,0.8,0.9,1]
for hidden_size in hidden_range:
     for learning_rate in learning_rate_range:
          print("Hidden size: ", hidden_size)
          print("Learning rate: ", learning_rate)
          subprocess.run(["python", "train_bpnn.py", str(hidden_size), str(learning_rate)])

          