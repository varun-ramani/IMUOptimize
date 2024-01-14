# IMUFind
A data driven approach to IMU-based pose estimation.

## What?
We want to solve the task of human pose estimation by using IMUs - by placing
devices that combine accelerometers and gyroscopes on various points on the
body, we want to run machine learning on the collected data and try to predict
the rotation of all the user's joints.

## Core Advancement
IMUs are expensive! We don't want to overdo them - so we need to try and
minimize their numhber. We'd like to try and get the most "bang for our buck" -
i.e. try and figure out where the most optimal IMU locations would be on the
body.