import subprocess
from itertools import product
import os

# Define the range of p and q values
p_values = [0.25, 0.5, 1, 2, 4]
q_values = [0.25, 0.5, 1, 2, 4]

# Path to the input and output
input_path = "graph/economy_collaborators_no_US_with_weights.edgelist"  # Match your actual path
output_dir = "emb/no_US_experiments/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all combinations of p and q
for p, q in product(p_values, q_values):
    output_path = f"{output_dir}country_collab_no_US_p_{p}_q_{q}.emd"
    
    # Build the command
    command = [
        "python", "src/main.py",
        "--input", input_path,
        "--output", output_path,
        "--weighted",
        "--directed",
        "--p", str(p),
        "--q", str(q)
    ]
    
    # Run the command
    print(f"Running: p={p}, q={q}")
    try:
        subprocess.run(command, check=True)
        print(f"✓ Success: p={p}, q={q}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: p={p}, q={q} - Error: {e}")
        break  # Stop on first error to avoid spam

print("Experiments completed!")