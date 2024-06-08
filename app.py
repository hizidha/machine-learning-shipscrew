from flask import Flask, json, request, render_template
from model import getRecommendation

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommended_candidates', methods=["POST"])
def recommend_candidates():
    data = request.json
    
    recommendations = getRecommendation(data)
    recommendations_json = recommendations.to_dict(orient='records')
    # print(recommendations_json)

    return json.dumps(recommendations_json)

if __name__ == '__main__':
    app.run(debug=True, port=5000)