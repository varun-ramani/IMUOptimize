# IMUOptimize: A Data-Driven Approach to Optimal IMU Placement for Human Pose Estimation with Transformer Architecture

This paper presents a novel approach for predicting human
poses using IMU data, diverging from previous studies such
as DIP-IMU, IMUPoser, and TransPose, which use up to
6 IMUs in conjunction with bidirectional RNNs. We in-
troduce two main innovations: a data-driven strategy for
optimal IMU placement and a transformer-based model ar-
chitecture for time series analysis. Our findings indicate
that our approach not only outperforms traditional 6 IMU-
based biRNN models but also that the transformer architec-
ture significantly enhances pose reconstruction from data ob-
tained from 24 IMU locations, with equivalent performance
to biRNNs when using only 6 IMUs. The enhanced accuracy
provided by our optimally chosen locations, when coupled
with the parallelizability and performance of transformers,
provides significant improvements to the field of IMU-based
pose estimation.