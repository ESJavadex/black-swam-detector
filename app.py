from flask import Flask, render_template, request, redirect, url_for, flash
import subprocess
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flash messages

@app.route('/', methods=['GET'])
def index():
    result = None
    error = None
    # Always run the simulation for today
    cmd = ['python', 'main.py']
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        result = proc.stdout
        if proc.returncode != 0:
            error = proc.stderr or 'Simulation failed.'
    except Exception as e:
        error = str(e)
    # Check if the plot exists
    plot_path = os.path.join('static', 'latest_plot.html')
    plot_exists = os.path.exists(plot_path)
    return render_template('index.html', result=result, error=error, plot_exists=plot_exists, plot_path=plot_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
