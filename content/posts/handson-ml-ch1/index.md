---
title: "Hands on Machine Learning - Chapter 1 Notes"
author: "Will Barker"
date: 2020-08-02
tags: ["handson-ml", "ml"]
---
<!--more--> 

## Notes

Types of ML
- ML: algorithms learning from data and improving performance on a task
- advantage over rule based systems is that machine can update parameters/logic with new/more data (refreshing the model)
- applying ML to discover/understand patterns in the data: data mining

- Supervised learning: training a model with labelled examples. can be used for classification tasks (predict a discrete category) or regression (predict a continuous number)

- Unsupervised learning algos: clustering unlabelled data, visualizing and dimensionality reduction, association rule learning
- hierarchical clustering: sub-dividing clusters into smaller groups
- dimensionality reduction: simplify data without losing too much information. i.e. merging correlated features. can help performance, takes up less disk and memory space
- anomoly detection
- association rule learning: relations between attributes (similar to data mining)

- Semi-supervised learning. i.e. Google Photos, looks at unlabelled pictures to find common faces. once given a label for a person it can name everyone in the photo. mixed hierarchical models

- Reinforcement learning: agent learns its environment and selects a policy (actions, strategy) to optimize some reward

Batch vs. Online Learning
- Batch: training a model on all available data. train the system and then put it in prod. retrain new versions of the model with new data
- requires a lot of computing resources, can be expensive. not suitable for autonomous systems with limited space for data (i.e. Mars rover)

- Online: train system incrementally on mini-batches of data
- great for continuous flows of data or limited storage resources
- can be used to train on datasets that won't fit in memory (out-of-core learning)

- learning rate: how fast model adapts to new data
- too high, forgets old data rapidly, too low, system has inertia (also less sensitive to noisey data)
- need to monitor online systems if garbage data starts coming in

Instance-Based vs. Model-Based Learning
- how a system generalizes (i.e. answers examples its never seen before)
- Instance-based: generalizes to new examples by comparing similarity to training examples
- i.e. KNN
- Model-based: builds a model to predict new data
- select a model (i.e. linear model) to represent the data's pattern
- tune model on a utility or cost function

- if a model doesn't generalize well, you can try again with better quality training data, more features, or a stronger model (e.g. polynomial vs. basic linear)
- adding more data tends to get better results on all kinds of algos, to the point where performance can be identical with enough data
- data needs to be representitive of the problem space trying to model, more data can eliminate noise but procedure needs to be solid or it risks sampling bias

Feature Engineering
- feature selection, picking the most useful features
- feature extraction, combining existing features to produce more useful ones + dimensionality reduction
- creating new features with the intro of new data

Performance
- overfitting: performing well on the training data but not generalizing well
- machine learning can pick up on noise in the data and sometimes even irrelevant features/useless metadata (i.e. data ids/index), detecting false patterns
- overfitting happens when the model is too complex relative to the amount and noiseness of the data
- regularization is constraining a model to make it simpler can help overfitting
- controlled by model hyperparameters (knobs to tweak on the model itself, such as learning rate)
- underfitting is opposite problem, model is too simple
- fix it with a more complex model, more/better features, reducing regularization constraints

- test models to see if they generalize well
- split data into training and test sets to get error rate on new cases (out of sample/generalization error)
- if training error is now but oos error is high, overfitting
- use a third validation set to compare models/hyperparameters, then select the best one and use it one the test
- cross validation splits the training set into subsets which are used for validation

- need to make assumptions about the data to pick models reasonably


## Questions

How would you define ML?
- Algorithms that allow a computer to learn from data to improve on a task and generalize to new examples well

Four types of problems where it shines?
- Problems that traditionally require too many rules or hand-tuning
- Complex problems with no easy logic based model
- Problems where underlying strategy/solutions change over time, so new model can help adapt
- Data mining, learning underlying patterns in data

What is a labelled training set?
- Data that has the variable of interest classified (independent variable), and information about it in other attribures (dependent variables)

Two most common supervised tasks?
- Regression: predicting a continuous number for the target variable
- Classification: predicting a discrete category of target variable

Four kinds of unsupervised tasks?
- Clustering: creating groups in unstructured data
- Dimesionality Reduction: reducing the number of attributes in the training set while keeping most of the variance(underlying signal)
- Anomoly Dection: finding outliers
- Visualization
- Association Rule learning: data mining, learning patterns in the data

What type of ML algo would you use to allow a robot to walk in various unknown terrains?
- Reinforcement Learning

What type of algo would you use to segment your customers into multiple groups?
- Clustering, Unsupervised Learning

Would you frame the problem of spam detection as a supervised learning problem?
- Supervised; we can use examples of spam and ham (labelled training data) to create a model that identifies which is which

What is an online learning system?
- A system that can update with new data as it comes in, ingesting as it comes in through mini-batches

What is an out-of-core learning system?
- Using online learning to train the model on a dataset that wouldn't fit inside the computers memory if you tried to train it in one giant batch.

What type of learning algo relies on similarity measures to make predictions?
- Instanced-based models, i.e. KNN

Difference between a model parameter and a hyperparameter?
- A model parameter is the coefficient determined by the algorithm to apply to a attribute/feature in the data when making predictions, a hyperparameter is an aspect of the model that you can adjust to change how it is trained

What do model-based learning algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?
- These algos look to fit the model to the data (i.e. represent the problem with some simplified version of it). The strategy is to improve performance in respect to some cost/utility function. They make new predictions by applying the policy/parameters determined through training on the new data's features

Four main challenges in ML?
- Not enough data
- Non-representative data
- Poor quality data
- Overfitting/Underfitting

If a model performs well on the training data but generalizes poorly to new examples, what is happening? 3 possible solutions?
- Model is overfitting to the training data and can't generalize well to the new examples
- You can regularize the model (i.e. constrain it) to be less representative of the training data
- Train the model on more data/more representative data
- Tune the model on a validation set

What is a test set and why would you want to use it?
- Test the performance of the model on new examples to understand how well it generalizes to new data (i.e. the whole point)

What can go wrong if you tune hyperparameters on the test set?
- You fit the model to work well on the testing set, overfitting it and reducing the chance of generalizing well

What is cross validation, why is it better than a validation set?
- Cross validation takes different chunks of the training data and uses them iteratively to train and validate the model, creating a more robust model (training on different samples) and is a more economic use of data.
