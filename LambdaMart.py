from LogisticRegression import *
import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import euclidean

def compute_similarity_features(query_embedding, passage_embedding): 

    #ensuring correct shape for sklearn functions
    query_embedding = np.array(query_embedding).reshape(1, -1)
    passage_embedding = np.array(passage_embedding).reshape(1, -1)

    # features (can always add more but will use these for now)
    cosine_sim = cosine_similarity(query_embedding, passage_embedding).item()
    euclidean_dist = euclidean_distances(query_embedding, passage_embedding).item()
    dot_product = np.dot(query_embedding, passage_embedding.T).item()
    absolute_diff = np.abs(query_embedding - passage_embedding).flatten()


    feature_vector = np.concatenate(([cosine_sim, euclidean_dist, dot_product], absolute_diff))
        
    return feature_vector

def prep_xgboost_data(df): 

    """
    Convert dataset into XGBoost Learning-to-Rank format.
    
    Args:
        df (DataFrame): DataFrame containing 'query_embedding', 'passage_embedding', 'relevancy'.
    
    Returns:
        X, y, group (Feature matrix, labels, query groups).
    """

    features = []
    labels = []
    q_groups = []

    for qid, group in tqdm(df.groupby("qid") , desc = "Processing features using LambdaMart"):
        group_features = []
        group_labels = []

        for i, row in group.iterrows():

            query_row = row["query_embedding"]
            passage_row = row["passage_embedding"]
            label = row["relevancy"]

            # compute similarity features
            sim_features = compute_similarity_features(query_row, passage_row)

            group_features.append(sim_features)
            group_labels.append(label)
        
        features.extend(group_features)
        labels.extend(group_labels)
        q_groups.append(len(group_features))

    return np.array(features, dtype = object), np.array(labels, dtype = object), np.array(q_groups, dtype = object)

def compute_lm_scores(model, queries_df, candidates_df, include_relevancy=False):
    """
    Compute and rank passage relevance scores using the trained Logistic Regression model.
    
    Parameters:
        model: Trained logistic regression model.
        queries_df: DataFrame with query embeddings.
        candidates_df: DataFrame with passage embeddings.
    
    Returns:
        DataFrame: Ranked passages with columns [qid, pid, rank, score, algoname]
    """
        
    rankings = []
    device = 'cpu'

    for _, query_row in tqdm(queries_df.iterrows(), total=len(queries_df), desc=f"Scoring Passages with LambdaMART"):
        qid = query_row["qid"]

        query_embedding = query_row["query_embedding"]

        query_passages = candidates_df[candidates_df["qid"] == qid].copy()

        # Prepare input features (concatenation of query & passage embeddings)

        X_test = np.array([compute_similarity_features(query_embedding, passage_embedding) for passage_embedding in query_passages["passage_embedding"]])
        X_test_dmatrix = xgb.DMatrix(X_test)
        scores = model.predict(X_test_dmatrix)

        query_passages["score"] = scores
        query_passages = query_passages.sort_values(by="score", ascending=False).reset_index(drop=True)

        query_passages["rank"] = query_passages.index + 1
        query_passages["algoname"] = "LM"
        query_passages["qid"] = query_passages["qid"].astype(str)
        query_passages["pid"] = query_passages["pid"].astype(str)

        # Select appropriate columns
        columns_to_include = ["qid", "pid", "rank", "score", "algoname"]
        if include_relevancy and "relevancy" in query_passages.columns:
            columns_to_include.append("relevancy")

        rankings.extend(query_passages[columns_to_include].values.tolist())

    return pd.DataFrame(rankings, columns=columns_to_include)

def compute_mAP_and_mNDCG_LM(model, queries_df, passages_df, include_relevancy=True):
    """
    Compute Mean Average Precision (mAP) and Mean Normalized Discounted Cumulative Gain (mNDCG).

    Parameters:
        model: Trained logistic regression model.
        queries_df: DataFrame with query embeddings.
        passages_df: DataFrame with passage embeddings.
        include_relevancy (bool): Whether to include the relevancy column in the rankings.

    Returns:
        dict: Dictionary containing 'mAP' and 'mNDCG' scores.
    """

    # Compute rankings with or without relevancy
    rankings = compute_lm_scores(model, queries_df, passages_df, include_relevancy=include_relevancy)

    # Ensure relevancy is present if required
    if include_relevancy and "relevancy" not in rankings.columns:
        raise ValueError("Error: 'relevancy' column is missing in computed rankings!")

    # Sort rankings correctly
    rankings = rankings.sort_values(by=['qid', 'score'], ascending=[True, False])

    query_metrics = []

    for qid, group in tqdm(rankings.groupby("qid"), desc="Computing mAP and mNDCG"):
        sorted_group = group.sort_values(by="score", ascending=False)
        
        if include_relevancy:
            relevancy_list = sorted_group["relevancy"].tolist()
        else:
            raise ValueError("Cannot compute mAP and mNDCG without relevancy information!")

        ap = average_precision(relevancy_list)
        ndcg = ndcg_at_k(relevancy_list, k=10)

        query_metrics.append({'AP': ap, 'NDCG': ndcg})

    results_df = pd.DataFrame(query_metrics)
    mAP = results_df['AP'].mean()
    mNDCG = results_df['NDCG'].mean()

    return {"mAP": mAP, "mNDCG": mNDCG}


if __name__ == "__main__": 

    glove_path = "glove.6B.300d.txt"
    glove_model = load_glove_embeddings(glove_path)

    dtype_mapping = {"qid": str, "pid": str, "query": str, "passage": str, "relevancy": int}

    train_file = "train_data.tsv"
    train_df = pd.read_csv(train_file, sep="\t", names=["qid", "pid", "query", "passage", "relevancy"], header=0, dtype=dtype_mapping)
    train_df = negative_sampling(train_df, neg_ratio = 10)   

    val_file = "validation_data.tsv"
    val_df = pd.read_csv(val_file, sep="\t", names=["qid", "pid", "query", "passage", "relevancy"], header=0, dtype=dtype_mapping)
    val_df = negative_sampling(val_df, neg_ratio = 10)

    tqdm.pandas(desc="Processing Queries")
    train_df["query_embedding"] = train_df["query"].progress_apply(lambda q: text_to_embedding_glove(q, glove_model))

    tqdm.pandas(desc="Processing Passages")
    train_df["passage_embedding"] = train_df["passage"].progress_apply(lambda p: text_to_embedding_glove(p, glove_model))

    tqdm.pandas(desc="Processing Validation Queries")
    val_df["query_embedding"] = val_df["query"].progress_apply(lambda q: text_to_embedding_glove(q, glove_model))

    tqdm.pandas(desc="Processing Validation Passages")
    val_df["passage_embedding"] = val_df["passage"].progress_apply(lambda p: text_to_embedding_glove(p, glove_model))

    # Converting data to necessary format for XGBoost
    X_train, y_train, group_train = prep_xgboost_data(train_df)
    X_val, y_val, group_val = prep_xgboost_data(val_df)

    # Training XGBoost model

    param_grid = {
        'objective': ['rank:ndcg'],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [4, 6, 8],
        "n_estimators": [100, 300, 500],
        "lambda": [0, 0.1, 1],  # L2 regularization (Ridge penalty)
        #"alpha": [0, 0.1, 1],  # L1 regularization (Lasso penalty) 
        'eval_metric': ['ndcg']
    }


    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    val_dmatrix = xgb.DMatrix(X_val, label=y_val)

    train_dmatrix.set_group(group_train)
    val_dmatrix.set_group(group_val)

    best_ndcg = 0.0
    best_hyperparams = {}

    for objective in param_grid['objective']:
        for learning_rate in param_grid['learning_rate']:
            for max_depth in param_grid['max_depth']:
                for n_estimators in param_grid['n_estimators']:
                    for lambd in param_grid['lambda']:
                        for eval_metric in param_grid['eval_metric']:
                            print(f"Objective: {objective}, Learning Rate: {learning_rate}, Max Depth: {max_depth}, N Estimators: {n_estimators},Lambda: {lambd} ,Eval Metric: {eval_metric}")
                            model_params = {
                                'objective': objective,
                                'learning_rate': learning_rate,
                                'max_depth': max_depth,
                                'lambda': lambd,
                                'eval_metric': eval_metric
                            }
                            
                            model = xgb.train(
                                model_params,
                                train_dmatrix,
                                num_boost_round= n_estimators,
                                evals=[(val_dmatrix, "validation")],
                                early_stopping_rounds=10,
                                verbose_eval=False  # Avoid excessive logging
                            )
   
                            ndcg_score = model.best_score
                            if ndcg_score > best_ndcg:
                                best_ndcg = ndcg_score
                                best_hyperparams = model_params
                                optimal_hyperparams = model_params.copy()
                                optimal_hyperparams["num_boost_round"] = n_estimators
                                optimal_hyperparams["best_iteration"] = model.best_iteration

    print(f"Best NDCG: {best_ndcg}")
    print(f"Best Hyperparameters: {optimal_hyperparams}")

    model_best = xgb.train(
        best_hyperparams,
        train_dmatrix,
        num_boost_round=100,
        evals = [(val_dmatrix, "validation")],
        early_stopping_rounds=10
    )

    # Now re-ranking the candidate_passages_top1000.tsv file with test_queries.tsv: 
    #

    test_queries_file = "test-queries.tsv"
    candidates_file = "candidate_passages_top1000.tsv"

    test_queries_df = pd.read_csv(test_queries_file, sep="\t", names=["qid", "query"])
    candidates_df = pd.read_csv(candidates_file, sep="\t", names=["qid", "pid", "query", "passage"])

    tqdm.pandas(desc="Processing Queries")
    test_queries_df["query_embedding"] = test_queries_df["query"].progress_apply(lambda q: text_to_embedding_glove(q, glove_model))

    tqdm.pandas(desc="Processing Passages")
    candidates_df["passage_embedding"] = candidates_df["passage"].progress_apply(lambda p: text_to_embedding_glove(p, glove_model))

    lm_rankings = compute_lm_scores(model_best, test_queries_df, candidates_df)
    save_rankings(lm_rankings, "LM.txt")

    val_queries_df = val_df[['qid', 'query']].drop_duplicates(subset = ["qid"])

    tqdm.pandas(desc="Processing Validation Queries")
    val_queries_df["query_embedding"] = val_queries_df["query"].progress_apply(lambda q: text_to_embedding_glove(q, glove_model))

    val_passages_df = val_df[['qid', 'pid', 'passage', 'relevancy']]
    val_passages_df["passage_embedding"] = val_passages_df["passage"].progress_apply(lambda q: text_to_embedding_glove(q, glove_model))

    # Compute mAP and mNDCG
    metrics = compute_mAP_and_mNDCG_LM(model_best, val_queries_df, val_passages_df, include_relevancy=True)
    print(metrics)













