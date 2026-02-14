import os
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy


class MovieRecommender:
    """
    MovieLens-compatible recommender.
    Expects:
      data/ratings.csv columns: userId,movieId,rating,timestamp
      data/movies.csv columns: movieId,title,genres
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "data")
        self.ratings_csv = os.path.join(self.data_dir, "ratings.csv")
        self.movies_csv = os.path.join(self.data_dir, "movies.csv")

        self.model = None
        self.movies = None
        self.ratings = None
        self.movie_id_by_title = {}
        self.title_by_movie_id = {}

    def load_data(self):
        if not os.path.exists(self.ratings_csv) or not os.path.exists(self.movies_csv):
            raise FileNotFoundError(
                "Missing data files. Put MovieLens ratings.csv and movies.csv in ./data/"
            )

        self.ratings = pd.read_csv(self.ratings_csv)
        self.movies = pd.read_csv(self.movies_csv)

        self.movie_id_by_title = dict(zip(self.movies["title"], self.movies["movieId"]))
        self.title_by_movie_id = dict(zip(self.movies["movieId"], self.movies["title"]))

    def train(self, test_size: float = 0.2, random_state: int = 42) -> float:
        if self.ratings is None or self.movies is None:
            self.load_data()

        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(self.ratings[["userId", "movieId", "rating"]], reader)

        trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)

        self.model = SVD(
            n_factors=100,
            n_epochs=20,
            lr_all=0.005,
            reg_all=0.02,
            random_state=random_state
        )
        self.model.fit(trainset)

        preds = self.model.test(testset)
        rmse = accuracy.rmse(preds, verbose=False)
        return rmse

    def recommend_for_existing_user(self, user_id: int, top_n: int = 10):
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        user_rated = set(self.ratings.loc[self.ratings["userId"] == user_id, "movieId"].tolist())
        all_movies = set(self.movies["movieId"].tolist())
        candidates = list(all_movies - user_rated)

        scored = []
        for mid in candidates:
            est = self.model.predict(user_id, mid).est
            scored.append((mid, est))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_n]
        return [(self.title_by_movie_id[mid], score) for mid, score in top]

    def recommend_from_favorites(self, favorite_titles: list[str], top_n: int = 10):
        """
        Favorites-based recommendations for a "new user":
        - Build centroid vector from favorite movie latent factors
        - Recommend movies with highest cosine similarity to centroid
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        fav_ids = []
        for t in favorite_titles:
            t = t.strip()
            if t and t in self.movie_id_by_title:
                fav_ids.append(self.movie_id_by_title[t])

        if not fav_ids:
            raise ValueError("No favorite titles matched movies.csv. Try exact titles.")

        trainset = self.model.trainset

        inner_ids = []
        for raw_mid in fav_ids:
            try:
                inner_ids.append(trainset.to_inner_iid(str(raw_mid)))
            except ValueError:
                pass

        if not inner_ids:
            raise ValueError("Favorites not found in trained model vocabulary.")

        import numpy as np
        fav_vecs = np.array([self.model.qi[i] for i in inner_ids])
        centroid = fav_vecs.mean(axis=0)

        scored = []
        for inner_iid in range(trainset.n_items):
            vec = self.model.qi[inner_iid]
            denom = (np.linalg.norm(vec) * np.linalg.norm(centroid)) + 1e-9
            sim = float(np.dot(vec, centroid) / denom)

            raw_mid = int(trainset.to_raw_iid(inner_iid))
            scored.append((raw_mid, sim))

        fav_set = set(fav_ids)
        scored = [(mid, s) for (mid, s) in scored if mid not in fav_set]

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_n]
        return [(self.title_by_movie_id.get(mid, f"movieId={mid}"), score) for mid, score in top]
