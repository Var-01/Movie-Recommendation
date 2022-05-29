# Movie Recommender

I used Python to build item-based movie recommender systems using KNN a prediction based machine learning algorithm . I've created functions to run the recommendations based on user inputs.

## Data

The data set contained 10,000 movie ratings, produced by 600 users. The data is available at the GroupLens Research website.

## Recommender
- Item-Based Recommender.

    I built this recommender using Knn algorithm to give predictions of the movies and then fetching top 5 movies by calculating their cosine distance.


## Installation

To run the project following installations and imports are to be made

```bash
  download python set path to environment variable
  pip install pandas
  pip install sklearn
  pip install scipy
  pip install flask
```
  To run the project follow the given steps

  - open terminal for main.py 
  - run the python file
  - you will be provided with a server url 
  - copy the link in the browser and you will be able to see the output
  - in the search bar enter any movie name (which is present in the dataset) 
    and on pressing enter you will be given the top 5 movie recommendations.


## Next Steps

I will build an online dashboard where a person can select a movie and check out the recommendations.

#### Releases
No Releases published


#### published
No packages published
