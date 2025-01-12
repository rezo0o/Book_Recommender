# Book Recommender System

A web application that provides personalized book recommendations based on user input using collaborative filtering techniques.

## Live Demo
[Try the app here](https://bookrecommender-check.streamlit.app/)

## Features
- Book search functionality
- Personalized recommendations based on book similarity
- User-friendly interface built with Streamlit
- Integration with Kaggle dataset for comprehensive book information

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Kaggle API

## Installation and Local Setup

1. Clone the repository
```bash
git clone https://github.com/your-username/Book_Recommender.git
cd Book_Recommender
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up Kaggle credentials
- Create a Kaggle account if you don't have one
- Download your Kaggle API credentials (kaggle.json)
- Place the kaggle.json file in the appropriate directory

4. Run the app
```bash
streamlit run app.py
```

## Configuration
Create a `.streamlit/secrets.toml` file with your Kaggle credentials:
```toml
[kaggle]
username = "YOUR_USERNAME"
key = "YOUR_KEY"
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
