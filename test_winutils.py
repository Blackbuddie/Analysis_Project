import os
import subprocess

# Get HADOOP_HOME
hadoop_home = os.environ.get('HADOOP_HOME')
print(f"HADOOP_HOME: {hadoop_home}")

if not hadoop_home:
    print("[ERROR] HADOOP_HOME environment variable is not set.")
    exit(1)

winutils_path = os.path.join(hadoop_home, 'bin', 'winutils.exe')
print(f"winutils.exe path: {winutils_path}")

if not os.path.isfile(winutils_path):
    print(f"[ERROR] winutils.exe not found at {winutils_path}")
    exit(1)

try:
    print("Running: winutils.exe ls C:\\")
    result = subprocess.run([winutils_path, 'ls', 'C:\\'], capture_output=True, text=True)
    print("Return code:", result.returncode)
    print("Output:")
    print(result.stdout)
    if result.stderr:
        print("[STDERR]:")
        print(result.stderr)
    if result.returncode != 0:
        print("[ERROR] winutils.exe did not run successfully.")
    else:
        print("winutils.exe ran successfully!")
except Exception as e:
    print(f"[ERROR] Exception while running winutils.exe: {e}") 