# Climate Analyzer Project

This project analyzes climate data, implements predictive algorithms, and visualizes results.

# HOW TO RUN:
In your projects root directory
1. Create a virtual environment
   - python3 -m venv .venv
2. Start the virtual environment
   - On Mac / Linux: source .venv/bin/activate
   - On Windows: .venv\Scripts\activate
3. Install the dependencies
   - pip install -r requirements.txt
4. Run the project
   - python3 src/main.py

## USAGE
After the project is ran, the user will be prompted with six options:
# Predict Humidity in Tallahassee, Fl
   If this option is selected, the program trains our machine learning model and the user will be asked to provide the following values:
      - Air temperature
      - Dew point
      - Precipitation (inches)
      - Windspeed
    With these values our algorithm calculates and outputs the predicted humidity percentage.

# Predict Average Monthly Temperature in Tallahassee, Fl
   If this option is selected, the program uses machine learning to predict and output the average monthly temperature for each month in 2026.

# Cluster Monthly Temperatures from Tallahasssee, Chicago & NYC
   ...

# Detect Temperature Anomalies in Tallahassee, Fl
   ...

# Graph Predicted Humidity vs Real Humidity in Tallahassee, Fl
   ...

# Exit Program
   The user enters "exit" to close the program.

(Project takes a second to run the first time after installing dependencies)
