import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_data():
    model_path = os.path.join(os.path.dirname(__file__), 'recommender_model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

data_bundle = load_data()


# Filter options 

@st.cache_data
def get_filter_options():
    movie_df = data_bundle['movie_df']
    genres = sorted(set(
        g for gs in movie_df['genres'].dropna()
        for g in gs.split()
        if len(g) > 2
    ))
    years = movie_df['title'].str.extract(r'\((\d{4})\)')[0].dropna().astype(int)
    return genres, int(years.min()), int(years.max())


# Stage classification 

def get_user_stage(user_id, df):
    n = df[df['userId'] == user_id].shape[0]
    if n == 0:
        return "new"
    elif n < 10:
        return "near-cold"
    elif n < 50:
        return "early"
    else:
        return "resident"


# Recommender 
class HybridRecommender:

    def __init__(self, svd_model, tfidf_matrix, movie_df, train_df, trainset, data, alpha=0.7):
        self.svd = svd_model
        self.tfidf_matrix = tfidf_matrix
        self.movie_df = movie_df
        self.train_df = train_df
        self.trainset = trainset
        self.data = data
        self.alpha = alpha
        self.movie_id_to_index = dict(zip(movie_df['movieId'], movie_df.index))

    # pre-filter

    def filter_candidates(self, genres=None, year_range=None):
        """Return a set of movieIds matching the optional genre/year filters."""
        mask = pd.Series(True, index=self.movie_df.index)

        if genres:
            genre_mask = self.movie_df['genres'].apply(
                lambda g: any(genre in g.split() for genre in genres)
                if isinstance(g, str) else False
            )
            mask = mask & genre_mask

        if year_range:
            year_series = self.movie_df['title'].str.extract(r'\((\d{4})\)')[0].astype(float)
            mask = mask & year_series.between(year_range[0], year_range[1])

        return set(self.movie_df[mask]['movieId'])

    # popular 

    def recommend_popular(self, n=5, candidate_ids=None):
        popular = (
            self.train_df.groupby('movieId')
            .agg(avg_rating=('rating', 'mean'), n_ratings=('rating', 'count'))
            .query('n_ratings >= 20')
            .sort_values(['avg_rating', 'n_ratings'], ascending=False)
        )
        if candidate_ids is not None:
            popular = popular[popular.index.isin(candidate_ids)]
        popular = popular.head(n)
        rows = self.movie_df[self.movie_df['movieId'].isin(popular.index)]
        return [
            {'title': r['title'], 'explanation': 'Highly rated overall'}
            for _, r in rows.iterrows()
        ]

    # onboarding 

    def get_onboarding_movies(self, n=10):
        top_ids = self.train_df.groupby('movieId').size().nlargest(n).index
        return self.movie_df[self.movie_df['movieId'].isin(top_ids)][['movieId', 'title']].reset_index(drop=True)

    def recommend_content_from_ratings(self, ratings_dict, n=5, candidate_ids=None):
        movie_indices, ratings = [], []
        for movie_id, rating in ratings_dict.items():
            if movie_id in self.movie_id_to_index:
                movie_indices.append(self.movie_id_to_index[movie_id])
                ratings.append(rating)
        if not movie_indices:
            return None
        profile = np.average(self.tfidf_matrix[movie_indices].toarray(), axis=0, weights=ratings)
        profile = profile.reshape(1, -1)
        scores = cosine_similarity(profile, self.tfidf_matrix).flatten()
        for movie_id in ratings_dict:
            if movie_id in self.movie_id_to_index:
                scores[self.movie_id_to_index[movie_id]] = -1
        if candidate_ids is not None:
            allowed = {self.movie_id_to_index[m] for m in candidate_ids if m in self.movie_id_to_index}
            mask = np.array([i in allowed for i in range(len(scores))])
            scores[~mask] = -1
        top_idx = np.argsort(scores)[-n:][::-1]
        rows = self.movie_df.iloc[top_idx]
        return [
            {'title': r['title'], 'explanation': 'Matches what you rated highly'}
            for _, r in rows.iterrows()
        ]

    # content 

    def _build_user_profile(self, user_id):
        user_data = self.train_df[self.train_df['userId'] == user_id]
        movie_indices, ratings = [], []
        for _, row in user_data.iterrows():
            if row['movieId'] in self.movie_id_to_index:
                movie_indices.append(self.movie_id_to_index[row['movieId']])
                ratings.append(row['rating'])
        if not movie_indices:
            return None
        profile = np.average(self.tfidf_matrix[movie_indices].toarray(), axis=0, weights=ratings)
        return profile.reshape(1, -1)

    def recommend_content(self, user_id, n=5, candidate_ids=None):
        profile = self._build_user_profile(user_id)
        if profile is None:
            return None
        scores = cosine_similarity(profile, self.tfidf_matrix).flatten()
        for movie_id in self.train_df[self.train_df['userId'] == user_id]['movieId']:
            if movie_id in self.movie_id_to_index:
                scores[self.movie_id_to_index[movie_id]] = -1
        if candidate_ids is not None:
            allowed = {self.movie_id_to_index[m] for m in candidate_ids if m in self.movie_id_to_index}
            mask = np.array([i in allowed for i in range(len(scores))])
            scores[~mask] = -1
        top_idx = np.argsort(scores)[-n:][::-1]
        rows = self.movie_df.iloc[top_idx]
        return [
            {'title': r['title'], 'explanation': 'Matches your watch history'}
            for _, r in rows.iterrows()
        ]

    # hybrid 

    def _reason(self, svd_norm, content_norm):
        svd_c = self.alpha * svd_norm
        cont_c = (1 - self.alpha) * content_norm
        if svd_c + cont_c < 0.2:
            return "A hidden gem you might enjoy"
        if cont_c == 0:
            return "Viewers with similar taste loved this"
        ratio = svd_c / (cont_c + 1e-9)
        if ratio > 3:
            return "Viewers with similar taste loved this"
        elif ratio < 0.5:
            return "Closely matches your watching history"
        return "Matches your history and loved by similar viewers"

    def recommend_hybrid(self, user_id, n=10, candidate_ids=None):
        profile = self._build_user_profile(user_id)
        if profile is None:
            return self.recommend_svd(user_id, n, candidate_ids=candidate_ids)
        scores = cosine_similarity(profile, self.tfidf_matrix).flatten()
        watched = set(self.train_df[self.train_df['userId'] == user_id]['movieId'])
        for m in watched:
            if m in self.movie_id_to_index:
                scores[self.movie_id_to_index[m]] = -1

        candidates = candidate_ids if candidate_ids is not None else set(self.movie_df['movieId'].unique())
        scored = []
        for m_id in candidates:
            if m_id in watched:
                continue
            svd_norm = (self.svd.predict(user_id, m_id).est - 0.5) / 4.5
            content_norm = 0.0
            if m_id in self.movie_id_to_index:
                content_norm = (scores[self.movie_id_to_index[m_id]] + 1) / 2
            hybrid = (self.alpha * svd_norm) + ((1 - self.alpha) * content_norm)
            scored.append((m_id, hybrid, svd_norm, content_norm))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for m_id, _, svd_norm, content_norm in scored[:n]:
            row = self.movie_df[self.movie_df['movieId'] == m_id]
            if not row.empty:
                results.append({
                    'title': row.iloc[0]['title'],
                    'explanation': self._reason(svd_norm, content_norm),
                })
        return results

    # SVD 
    def recommend_svd(self, user_id, n=10, candidate_ids=None):
        all_ids = set(self.data.df['movieId'].unique())
        if candidate_ids is not None:
            all_ids = all_ids & candidate_ids
        try:
            rated = set(self.trainset.ur[self.trainset.to_inner_uid(user_id)])
        except ValueError:
            rated = set()
        preds = [(m, self.svd.predict(user_id, m).est) for m in all_ids if m not in rated]
        top_ids = [m for m, _ in sorted(preds, key=lambda x: x[1], reverse=True)[:n]]
        results = []
        for m_id in top_ids:
            row = self.movie_df[self.movie_df['movieId'] == m_id]
            if not row.empty:
                results.append({
                    'title': row.iloc[0]['title'],
                    'explanation': 'Predicted to match your taste',
                })
        return results


# Router 

def router(user_id, n, recommender, candidate_ids=None):
    stage = get_user_stage(user_id, recommender.train_df)

    if stage in ("new", "near-cold"):
        return stage, {
            "popular": recommender.recommend_popular(n=n, candidate_ids=candidate_ids),
            "content": recommender.recommend_content(user_id, n=n, candidate_ids=candidate_ids),
        }
    elif stage == "early":
        return stage, recommender.recommend_hybrid(user_id, n=n, candidate_ids=candidate_ids)
    else:
        return stage, recommender.recommend_svd(user_id, n=n, candidate_ids=candidate_ids)


def make_recommender():
    return HybridRecommender(
        data_bundle['svd_model'],
        data_bundle['tfidf_matrix'],
        data_bundle['movie_df'],
        data_bundle['train_df'],
        data_bundle['trainset'],
        data_bundle['raw_data'],
    )


# UI 

STAGE_LABELS = {
    "new":       "New user",
    "near-cold": "Near-cold",
    "early":     "Early",
    "resident":  "Resident",
}

def render_rec_list(recs, key_prefix):
    for i, rec in enumerate(recs, 1):
        col_info, col_yes, col_no = st.columns([6, 1, 1])
        with col_info:
            st.write(f"**{i}. {rec['title']}**")
            st.caption(rec['explanation'])
        with col_yes:
            if st.button("👍", key=f"{key_prefix}_yes_{i}", help="Interested"):
                st.session_state.feedback[rec['title']] = 'interested'
        with col_no:
            if st.button("👎", key=f"{key_prefix}_no_{i}", help="Not interested"):
                st.session_state.feedback[rec['title']] = 'not_interested'
        status = st.session_state.feedback.get(rec['title'])
        if status == 'interested':
            st.success("Marked as Interested")
        elif status == 'not_interested':
            st.warning("Marked as Not Interested")

def filter_summary(genres, year_range, year_min, year_max):
    parts = []
    if genres:
        parts.append(', '.join(genres))
    if year_range and (year_range[0] != year_min or year_range[1] != year_max):
        parts.append(f"{year_range[0]}–{year_range[1]}")
    return ' · '.join(parts) if parts else None


# App 

st.title("Movie Recommendation System")
st.markdown("Get personalized recommendations based on your viewing history.")

if 'stage' not in st.session_state:
    st.session_state.stage = 'input'

all_genres, year_min, year_max = get_filter_options()

# Stage: input 
if st.session_state.stage == 'input':
    user_id = st.number_input("Enter your User ID:", min_value=1, value=1, step=1)
    n_recs = st.slider("Number of recommendations:", 5, 20, 10)

    with st.expander("🎯 Filter by genre or era (optional)"):
        selected_genres = st.multiselect("Genres", options=all_genres)
        year_range = st.slider("Release year", year_min, year_max, (year_min, year_max))

    if st.button("Generate Recommendations", type="primary"):
        uid = int(user_id)
        stage = get_user_stage(uid, data_bundle['train_df'])
        st.session_state.user_id = uid
        st.session_state.n_recs = n_recs
        st.session_state.user_stage = stage
        st.session_state.selected_genres = selected_genres
        st.session_state.year_range = year_range

        rec = make_recommender()
        candidate_ids = rec.filter_candidates(
            genres=selected_genres or None,
            year_range=year_range if (year_range[0] != year_min or year_range[1] != year_max) else None,
        )
        st.session_state.candidate_ids = candidate_ids if candidate_ids is not None else None

        if stage == "new":
            st.session_state.stage = 'onboarding'
        else:
            with st.spinner("Finding your recommendations..."):
                _, results = router(uid, n_recs, rec, candidate_ids=candidate_ids or None)
            st.session_state.results = results
            st.session_state.feedback = {}
            st.session_state.stage = 'results'
        st.rerun()

# Stage: onboarding 
elif st.session_state.stage == 'onboarding':
    st.info("👋 You're new here! Rate a few movies so we can personalize your picks.")

    rec = make_recommender()
    popular = rec.get_onboarding_movies(10)

    st.markdown("### Rate movies you've seen (0 = haven't seen it)")
    ratings = {}
    cols = st.columns(2)
    for i, row in popular.iterrows():
        with cols[i % 2]:
            val = st.slider(row['title'], min_value=0, max_value=5, value=0, step=1,
                            key=f"ob_{row['movieId']}")
            if val > 0:
                ratings[row['movieId']] = val

    st.caption(f"{len(ratings)} movie(s) rated")
    st.divider()

    candidate_ids = st.session_state.get('candidate_ids')

    col_submit, col_skip = st.columns(2)
    with col_submit:
        if st.button("Get my recommendations", type="primary", disabled=len(ratings) == 0):
            with st.spinner("Building your profile..."):
                content = rec.recommend_content_from_ratings(
                    ratings, n=st.session_state.n_recs, candidate_ids=candidate_ids)
                popular_recs = rec.recommend_popular(
                    n=st.session_state.n_recs, candidate_ids=candidate_ids)
            st.session_state.results = {"popular": popular_recs, "content": content}
            st.session_state.feedback = {}
            st.session_state.stage = 'results'
            st.rerun()
    with col_skip:
        if st.button("Skip — show popular picks"):
            with st.spinner("Loading..."):
                popular_recs = rec.recommend_popular(
                    n=st.session_state.n_recs, candidate_ids=candidate_ids)
            st.session_state.results = {"popular": popular_recs, "content": None}
            st.session_state.feedback = {}
            st.session_state.stage = 'results'
            st.rerun()

# Stage: results 
elif st.session_state.stage == 'results':
    stage = st.session_state.user_stage
    uid = st.session_state.user_id

    st.markdown(f"**User {uid}** — {STAGE_LABELS[stage]}")

    summary = filter_summary(
        st.session_state.get('selected_genres', []),
        st.session_state.get('year_range'),
        year_min, year_max,
    )
    if summary:
        st.caption(f"Filtered: {summary}")

    st.divider()
    results = st.session_state.results

    if stage in ("new", "near-cold"):
        col_pop, col_content = st.columns(2)
        with col_pop:
            st.subheader("Popular Picks")
            render_rec_list(results["popular"], key_prefix="pop")
        with col_content:
            st.subheader("Based on Your History")
            if results["content"] is None:
                st.info("No watch history yet — rate some movies to unlock personalized picks.")
            else:
                render_rec_list(results["content"], key_prefix="cont")
    else:
        label = "Hybrid Recommendations" if stage == "early" else "Your Recommendations"
        st.subheader(label)
        render_rec_list(results, key_prefix="rec")

    if st.session_state.feedback:
        st.divider()
        st.markdown(f"**Feedback recorded:** {len(st.session_state.feedback)} movie(s) rated")

    st.divider()
    if st.button("Start over"):
        for key in ('stage', 'results', 'feedback', 'user_id', 'user_stage',
                    'selected_genres', 'year_range', 'candidate_ids'):
            st.session_state.pop(key, None)
        st.rerun()
