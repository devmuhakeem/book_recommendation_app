from flask import Flask, request, jsonify, render_template
from book_recommender import recommend_books

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = request.args.get("user_id", type=int)
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    recommendations = recommend_books(user_id)
    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(debug=True)
