import subprocess
import sys

def start_model_process(model_path, env_path):
    python_path = '/root/miniconda3/envs/AgentFormer/bin/python'
    return subprocess.Popen(
        [python_path, model_path],# ["conda", "run", "-n", env_path, "python", model_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        encoding="utf-8",
    )
def communicate_with_model(process, input_data):
    process.stdin.write("Hello there" + "\n")
    process.stdin.flush()
    output_data = process.stdout.readline().strip()
    return output_data

if __name__ == "__main__":
    model1_path = "test_comm_sub.py"
    model1_env = 'AgentFormer'
    # Start model1 process
    model1_process = start_model_process(model1_path, model1_env)
    start_up_message = model1_process.stdout.readline().strip()
    print(f"start_up_message: {start_up_message}")

    # Communicate with model1
    input_data1 = "test input example"
    output_data1 = communicate_with_model(model1_process, input_data1)
    print("Model1 output:", output_data1)

    # Close the processes
    model1_process.stdin.close()
    model1_process.wait()