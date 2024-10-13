# DecisionLens: AI-Powered Business Strategy Assistant

DecisionLens is an innovative AI-powered application designed to assist businesses in making data-driven strategic decisions. It combines market analysis, business strategy assistance, and data strategy simulation to provide comprehensive insights for decision-makers.

Access Link here 👉 https://decisionlens.streamlit.app/
## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

1. **User Authentication**: Secure login and signup functionality using Firebase.
2. **Market Analysis**: Analyze commodity and company stock data with interactive visualizations.
3. **Business Strategy Assistant**: AI-powered query system for strategic business insights.
4. **Data Strategy Simulator**: Upload and analyze business data, connect to databases, and get AI-generated strategic recommendations.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7+
- pip (Python package manager)
- A Firebase account and project
- A Groq API key
- MySQL database (for the Data Strategy Simulator feature)

## Installation

1. Clone the repository:
    git clone https://github.com/yourusername/decisionlens.git
    cd decisionlens
2. Create a virtual environment (optional but recommended):
    python -m venv venv
    source venv/bin/activate  # On Windows, use venv\Scripts\activate
3. Install the required packages:
    pip install -r requirements.txt
4. Set up your Firebase configuration:
- Create a Firebase project at [Firebase Console](https://console.firebase.google.com/)
- Generate a new private key for your service account
- Save the JSON file as `prithvi-45d3f-firebase-adminsdk-o4c77-62e1d077aa.json` in the project root directory

5. Set up your Groq API key:
- Sign up for a Groq account and obtain an API key
- You'll enter this key in the application's sidebar when running the app

## Usage

To run the DecisionLens application:

1. Ensure you're in the project directory and your virtual environment is activated (if you're using one).

2. Run the Streamlit app:
    streamlit run app.py
3. Open your web browser and go to `http://localhost:8501` (or the URL provided in the terminal).

4. Log in or sign up to access the DecisionLens features.

## Configuration

### Firebase Configuration

Update the `firebase_config` dictionary in `app.py` with your Firebase project details:

```python
firebase_config = {
 "apiKey": "YOUR_API_KEY",
 "authDomain": "YOUR_AUTH_DOMAIN",
 "projectId": "YOUR_PROJECT_ID",
 "storageBucket": "YOUR_STORAGE_BUCKET",
 "messagingSenderId": "YOUR_MESSAGING_SENDER_ID",
 "appId": "YOUR_APP_ID",
 "databaseURL": "YOUR_DATABASE_URL"
}
```
Project Structure:
```
decisionlens/
│
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── prithvi-45d3f-firebase-adminsdk-o4c77-62e1d077aa.json  # Firebase admin SDK key
```
License
This project is licensed under the MIT License - see the LICENSE.md file for details.
Copy
This README provides a comprehensive guide for users and potential contributors to understand, install, and use your DecisionLens project. You may want to customize it further based on specific details of your project implementation or any additional features you add.

Remember to create the mentioned CONTRIBUTING.md and LICENSE.md files if you haven't already,
