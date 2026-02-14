# AviMovieRecommender

A simple **movie recommendation system** using **collaborative filtering (SVD)** with a lightweight **Flask** interface.

## Features
- Mode 1: Recommend for an **existing userId** (from ratings.csv)
- Mode 2: Recommend for a **new user** based on favorite movie titles (one per line)
- Includes basic evaluation metric (RMSE) shown on the homepage

## Setup
1. Create a virtual environment (recommended)
2. Install deps:
   ```bash
   pip install -r requirements.txt
