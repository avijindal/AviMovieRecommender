import os
from flask import Flask, render_template, request
from recommender import MovieRecommender

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize recommender
rec = MovieRecommender(BASE_DIR)
rec.load_data()
rmse = rec.train()


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", rmse=rmse)


@app.route("/recommend", methods=["POST"])
def recommend():
    mode = request.form.get("mode", "user")
    top_n = int(request.form.get("top_n", "10"))

    try:
        if mode == "user":
            user_id = int(request.form.get("user_id", "1"))
            user_name = request.form.get("user_name", "").strip()

            results = rec.recommend_for_existing_user(user_id=user_id, top_n=top_n)

            if user_name:
                title = f"Recommendations for {user_name}"
            else:
                title = f"Recommendations for Profile ID {user_id}"

            metric_label = "Predicted Rating"

        else:
            favorites_raw = request.form.get("favorites", "")
            favorites = [x.strip() for x in favorites_raw.split("\n") if x.strip()]

            results = rec.recommend_from_favorites(favorites, top_n=top_n)
            title = "Recommendations Based on Your Favorite Movies"
            metric_label = "Similarity Score"

        return render_template(
            "results.html",
            title=title,
            results=results,
            metric_label=metric_label
        )

    except Exception as e:
        return render_template(
            "results.html",
            title="Error",
            results=[],
            metric_label="",
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)
