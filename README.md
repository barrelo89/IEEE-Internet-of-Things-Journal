# Small Profits adn Quick Returns: A Practical Social Welfare Maximizing Incentive Mechanisms for Deadline-sensitive Tasks in Crowdsourcing
This is python code for the paper work published in IEEE Internet of Things Journal. You can access to the paper through this [link](https://ieeexplore.ieee.org/document/8897639).

## Prerequisites
- Language: Python 
- Required Packages: numpy, pandas, matplotlib, munkres 
- To install the required packages, type the following command:
1) Python 2
```
pip install numpy pandas matplotlib munkres
```
2) Python 3
```
pip3 install numpy pandas matplotlib munkres
```

## Running the code
- Class 'task': task.py, Class 'provider' or 'worker': provider.py, and Class 'platform': platform.py
1) Visualization of the behavior of task
![Alt text](/media/duin/SAMSUNG1/Research/IEEE Internet of Things Journal/Python Code/figure/task_value.pdf?raw=true "Title")
```
python3 task_figure.py
```
2) Visualization of the punctual behavior of worker 
```
python3 provider_figure.py
```
3) Simulation for optimal task-worker matching 
```
python3 optimal_run.py
```
4) Visualization of the results of 3) 
```
python3 optimal_visualization.py
```
5) Simulation for one time competition between our work and the benchmark 
```
python3 capacity_run.py
```
6) Visualization of the results from 5) 
```
python3 capacity_visualization.py
```
7) Simulation for reselection process which consider the movement of workers and tasks between two platforms 
```
python3 reselection.py
```
8) Visualization of the results from 7) 
```
python3 reselection_visualization.py
```
9) Simulation for multiple reselection process which converges to balance point 
```
python3 reselection_multiple_round.py
```
10) Visualization of the results from 9) 
```
python3 reselection_multiple_round_visualization.py
```
11) Simulation for parameter setting
```
python3 parameter_run.py
```
12) Visualization of the results from 11) 
```
python3 parameter_visualization.py
```

