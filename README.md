# Convex Hull Classification with Optimization Techniques

This repository contains solutions to the **Convex Hull Classification Problem** from the ECE599/AI539 Convex Optimization course (Homework 3). The project involves implementing and comparing various optimization techniques for Support Vector Machines (SVMs) using Convex Hull formulations.

---

## Overview

This repository provides implementations for:
1. **Convex Hull (C-Hull) and Reduced Convex Hull (Reduced C-Hull)** classifiers.
2. Optimization methods:
   - **Projected Gradient Descent**
   - **Nesterov’s Accelerated Gradient**
   - **Alternating Direction Method of Multipliers (ADMM)**
3. Visualization of classifiers and error reporting.
4. Comparative analysis of optimization methods through iteration vs. objective value and time vs. objective value plots.
5. A detailed report explaining the methodology, results, and insights is available in the `report` directory. Access it [here](./report/report3_Woonki_Kim.pdf).

---

## Methodology

### Optimization Techniques Implemented:
1. **Projected Gradient Descent**:
   - Simple iterative optimization method with projection onto constraints.
2. **Nesterov’s Accelerated Gradient**:
   - Faster convergence by incorporating momentum terms.
   - Reference: Beck and Teboulle's "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems."
3. **ADMM (Alternating Direction Method of Multipliers)**:
   - Solves optimization problems by splitting variables and updating iteratively.
   - Reference: Boyd et al.'s "Distributed Optimization and Statistical Learning via ADMM."

### Dataset:
- **Separable Data**: Train and test datasets for linearly separable classes.
- **Overlapping Data**: Train and test datasets for overlapping classes.

---

## Results

### Visualizations:
- Training and testing data visualized with classifiers.
- Comparative metrics include:
  - Classification error on testing datasets.
  - Iteration vs. Objective Value plot.
  - Time vs. Objective Value plot.

### Error Reporting:
- Classification errors reported for all methods and datasets.
