# Improving Graph-Based Movie Recommender System Using Cinematic Experience
MDPI Applied Science 논문 게재
- (2022.01.28) [Improving Graph-Based Movie Recommender System Using Cinematic Experience](https://www.mdpi.com/2076-3417/12/3/1493) in Applied Science in MDPI
- (2022.01.20) [Multi-Relational Stacking Ensemble Recommender System Using Cinematic Experience](https://ieeexplore.ieee.org/document/9736528) in IEEE BigComp2022, [Code](https://github.com/cheonsol-lee/bigcomp_2022_multi_graph)
- Author's Paper link: [https://arxiv.org/abs/1904.12058](https://arxiv.org/abs/1904.12058)
- Author's code: [https://github.com/muhanzhang/IGMC](https://github.com/muhanzhang/IGMC)
- Reference code: [https://github.com/zhoujf620/Motif-based-inductive-GNN-training](https://github.com/zhoujf620/Motif-based-inductive-GNN-training)


## Dependencies

* PyTorch 1.2+
* DGL 0.5 (nightly version)

## Data

Movie data
- Rotten Tomatoes data : [https://www.kaggle.com/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset](https://www.kaggle.com/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)
- Amazon Movie data : [http://jmcauley.ucsd.edu/data/amazon/](http://jmcauley.ucsd.edu/data/amazon/)

Training set for BERT
- Sentiment Analysis : [https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)
- Emotion Analysis : [https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp](https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp)


## How to run

- explicit_train_rotten.ipynb


## Results

|Dataset|Our code <br> best of epochs|
|:-:|:-:|
|Rotten Tomatoes|0.8004|
|Amazon Movie|0.9621|
