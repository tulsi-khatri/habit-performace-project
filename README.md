**Habit Performance Analyser**

Overview:
I built this project to understand how daily habits affect overall performance using data and machine learning.

Instead of just tracking habits, I wanted to analyze patterns and predict a performance score based on factors like sleep, study time, screen usage, mood, and exercise. The idea was to apply ML on a simple real-life problem and compare how different models perform.


-> Tech Stack:
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  



-> Features:
- Generates a custom dataset for habit tracking  
- Applies multiple ML models for prediction  
- Compares Linear Regression and Random Forest  
- Uses cross-validation for better evaluation  
- Hyperparameter tuning using GridSearchCV  
- Visualizes predictions and feature importance  
- Saves trained model for future use  



-> Machine Learning Approach
In this project, I used two models:

- **Linear Regression** as a baseline model  
- **Random Forest Regressor** for improved performance  

I also implemented:
- Train-test split for evaluation  
- Cross-validation to check model stability  
- GridSearchCV for hyperparameter tuning  

Finally, I compared both models using R² score and Mean Squared Error to evaluate performance.


-> Visualizations:
- Actual vs Predicted performance graph:
  <img width="1792" height="1120" alt="Screenshot 2026-04-19 at 12 00 32 AM" src="https://github.com/user-attachments/assets/0e2fbfbf-e1c5-484d-a51b-fb25c5711468" />
  This plot compares actual performance with model predictions.
  Points closer to the diagonal indicate better prediction accuracy , showing that the model captures the overall trend well.
 

- Feature importance graph:
-  <img width="1792" height="1120" alt="Screenshot 2026-04-19 at 12 00 47 AM" src="https://github.com/user-attachments/assets/814cc9ca-1e3c-4c23-947e-1807e577db8e" />
This graph shows the contribution of each habit to performance prediction.
Study hours and sleep have higher importance , indicating they play a major role in determining performance. 


-> Dataset:
I created a synthetic dataset using NumPy to simulate real-life habit patterns.

 
-> Features:
- Sleep hours  
- Study hours  
- Screen time  
- Mood score  
- Exercise

  

-> Target:
- Performance score (based on a custom formula with some randomness)


-> Requirements:
- pandas  
- numpy  
- matplotlib  
- scikit-learn  
- joblib  



-> Installation:
Make sure Python is installed, then run:

pip3 install pandas numpy matplotlib scikit-learn joblib



-> How to Run:
1. Run the Python script:
python3 habit-performance.py

2. The program will:
- Generate dataset  
- Train ML models  
- Compare performance  
- Show graphs  
- Save the model as `habit_model.pkl`  



-> What I Learned:
- How to build a complete ML pipeline from scratch  
- Difference between linear and ensemble models  
- Importance of model evaluation and cross-validation  
- Basics of hyperparameter tuning  
- How to interpret feature importance  



-> Future Improvements:
- Use real-world dataset instead of synthetic data  
- Try advanced models like XGBoost  
- Build a simple user interface  
- Deploy as a web application  


 Author
Tulsi Khatri
BTech Student | Aspiring AI/ML Engineer
