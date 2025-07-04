# IoT Telemetry failure and anomaly detection


# 1. Introduction

The dataset selected for this work is the Smart Manufacturing IoT Cloud Monitoring Dataset
- https://www.kaggle.com/datasets/ziya07/smart-manufacturing-iot-cloud-monitoring-dataset

We used this dataset as such machines are similar to any PC components. It contains features such as temperature, vibration, humidity, pressure, and energy consumption, which can represent the real features of a PC component:
- Temperature: 
    - Represents the thermal state of PC components like CPU and GPU. 
    - Elevated, fluctuating, or sustained high temps often hint at cooling interface degradation or increased workload—just as in IoT systems, abnormal temperature readings indicate equipment stress or failure 

- Vibration: 
    - In PCs, fan vibration or hard-drive spin irregularities act like mechanical failure indicators—mirroring industrial settings where vibration spikes reveal mechanical faults in motors or structures.

- Humidity: 
    - While PCs are kept in controlled environments, ambient humidity still affects internal corrosion risk and electrical stability—just like environmental IoT sensor systems highlight humidity’s impact on electronic device reliability.

- Pressure: 
    - Though PCs don’t have internal pressure sensors, this can stand in for PSU voltage fluctuations or airflow pressure changes. 
    - In industrial IoT, pressure sensors track fluid or airflow—changes often signal blockages or system degradation.

- Energy consumption: 
    - Maps directly to PC power draw, reflecting CPU/GPU workloads or inefficiencies—exist in IoT energy‐monitoring systems where spikes can signal abnormal component behavior .


# 2. Explorary Data Analysis
- Load data
- Analyze and plot data over time and show anomalies
    ```
    1_main_EDA.ipynb
    ```

# 3. Feature Engineering
- Rolling windows for mean/diff/trend
- Scale features

# 4. Model Training
- Train LightGBM autoencoder or classifier
- Show feature importance, evaluation metrics

# 5. Optimization
- Reduce number of features from 5 to 2

# 6. Inference Demo
- Create streaming loop over telemetry samples

# 7. Save Artifacts
- Save models, summary tables, plots


# Edge Telemetry Anomaly Detection with LightGBM

**Steps:**
1. Load SMART telemetry
2. Feature scaling
3. Train compact LightGBM autoencoder
4. Evaluate on holdout set
5. Export to ONNX & quantize
6. Benchmark size & latency
7. Demonstrate streaming inference