# 1. Introduction
- Describe objective & dataset

# 2. Data Ingestion & EDA
- Load CSV(s)
- Plot SMART stats over time, show anomalies

# 3. Feature Engineering
- Rolling windows for mean/diff/trend
- Scale features

# 4. Model Training
- Train LightGBM autoencoder or classifier
- Show feature importance, evaluation metrics

# 5. Optimization
- Export to ONNX
- Quantize to int8
- Measure model size & inference speed

# 6. Inference Demo
- Create streaming loop over telemetry samples
- Load optimized model, flag anomalies live

# 7. Save Artifacts
- Save models, summary tables, plots
