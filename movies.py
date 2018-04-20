# coding: utf-8

from scipy import optimize
from numpy import *

# In[17]:

# define the number of movies in our 'database'

num_movies = 10

# In[18]:

# define the number of users in our 'database'

num_users = 5

# In[19]:

# randomly initialize some movie ratings
# a 10 X 5 matrix

ratings = random.randint(11, size=(num_movies, num_users))

# print(ratings)

# create a logical matrix (matrix that represents whether a rating was made, or not)
# != is the logical not operator
# print("Did Nikhil Rate?")
did_rate = (ratings != 0) * 1
# print(did_rate)

# Here's what happens if we don't multiply by 1
# print("Line 36")
# print(ratings != 0)

# print("line 40")
# print(ratings != 0 * 1)

# Get the dimensions of a matrix using the shape property
# print(ratings.shape)
# print(did_rate.shape)

# Let's make some ratings. A 10 X 1 column vector to store all the ratings I make

nikhil_ratings = zeros((num_movies, 1))
# print("Nikhil Ratings")
# print(nikhil_ratings)

# In[28]:

# Python data structures are 0 based
# print(nikhil_ratings[9])

# In[29]:

# I rate 3 movies
nikhil_ratings[0] = 8
nikhil_ratings[4] = 7
nikhil_ratings[7] = 3

# print(nikhil_ratings)

# In[30]:

# Update ratings and did_rate

ratings = append(nikhil_ratings, ratings, axis=1)
did_rate = append(((nikhil_ratings != 0) * 1), did_rate, axis=1)

# print("Ratings")
# print(ratings)
# print(ratings.shape)

# In[33]:

# print(did_rate)

# print(did_rate)

# print(did_rate.shape)

# Simple explanation of what it means to normalize a dataset

a = [10, 20, 30]
aSum = sum(a)

# In[37]:

# print(aSum)

# In[38]:

# aMean = aSum / len(a)

# print("The Mean rating is ")
# print(aMean)

# print("Input 40")
aMean = mean(a)
# print(aMean)

# In[41]:

# a = [10 - aMean, 20 - aMean, 30 - aMean]
# print(a)


# print(ratings)

# a function that normalizes a dataset

def normalize_ratings(ratings, did_rate):
	num_movies = ratings.shape[0]

	ratings_mean = zeros(shape=(num_movies, 1))
	ratings_norm = zeros(shape=ratings.shape)

	for i in range(num_movies):
		# Get all the indexes where there is a 1
		idx = where(did_rate[i] == 1)[0]
		#  Calculate mean rating of ith movie only from user's that gave a rating
		ratings_mean[i] = mean(ratings[i, idx])
		ratings_norm[i, idx] = ratings[i, idx] - ratings_mean[i]

	return ratings_norm, ratings_mean


# Normalize ratings
ratings, ratings_mean = normalize_ratings(ratings, did_rate)

# In[45]:

# Update some key variables now

num_users = ratings.shape[1]
num_features = 3

# Simple explanation of what it means to 'vectorised' a linear regression

X = array([[1, 2], [1, 5], [1, 9]])
Theta = array([[0.23], [0.34]])

# In[47]:

# print(X)

# In[48]:

# print(Theta)

# In[49]:

Y = X.dot(Theta)
# print(Y)

# In[50]:

# Initialize Parameters theta (user_prefs), X (movie_features)

movie_features = random.randn(num_movies, num_features)
user_prefs = random.randn(num_users, num_features)
initial_X_and_theta = r_[movie_features.T.flatten(), user_prefs.T.flatten()]

# In[51]:
# print("Movie Features")
# print(movie_features)

# print("User Preferences")
# print(user_prefs)

# print("Initial X and Theta")
# print(initial_X_and_theta)

# print(initial_X_and_theta.shape)

# print(movie_features.T.flatten().shape)

# print(user_prefs.T.flatten().shape)

# print("Input 57 at line 225")
# print(initial_X_and_theta)


# In[58]:

def unroll_params(x_and_theta, users, movies, features):
	# Retrieve the X and theta matrices from X_and_theta, based on their dimensions
	# (num_features, num_movies, num_movies)
	# --------------------------------------------------------------------------------------------------------------
	# Get the first 30 (10 * 3) rows in the 48 X 1 column vector
	first_30 = x_and_theta[:movies * features]
	# Reshape this column vector into a 10 X 3 matrix
	X = first_30.reshape((features, movies)).transpose()
	# Get the rest of the 18 the numbers, after the first 30
	last_18 = x_and_theta[movies * features:]
	# Reshape this column vector into a 6 X 3 matrix
	theta = last_18.reshape(features, users).transpose()
	return X, theta


# In[59]:

def calculate_gradient(x_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	X, theta = unroll_params(x_and_theta, num_users, num_movies, num_features)

	# we multiply by did_rate because we only want to consider observations for which a rating was given
	difference = X.dot(theta.T) * did_rate - ratings
	X_grad = difference.dot(theta) + reg_param * X
	theta_grad = difference.T.dot(X) + reg_param * theta

	# wrap the gradients back into a column vector
	return r_[X_grad.T.flatten(), theta_grad.T.flatten()]


# In[60]:

def calculate_cost(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)

	# we multiply (element-wise) by did_rate because we only want to consider observations for which a rating was given
	cost = sum((X.dot(theta.T) * did_rate - ratings) ** 2) / 2
	# '**' means an element-wise power
	regularization = (reg_param / 2) * (sum(theta ** 2) + sum(X ** 2))
	return cost + regularization


# import these for advanced optimizations (like gradient descent)

# regularization parameter

reg_param = 30

# perform gradient descent, find the minimum cost (sum of squared errors)
# and optimal values of X (movie_features) and Theta (user_preferences)

minimized_cost_and_optimal_params = optimize.fmin_cg(calculate_cost, fprime=calculate_gradient, x0=initial_X_and_theta,
                                                     args=(
                                                     ratings, did_rate, num_users, num_movies, num_features, reg_param),
                                                     maxiter=100, disp=True, full_output=True)

cost, optimal_movie_features_and_user_prefs = minimized_cost_and_optimal_params[1], minimized_cost_and_optimal_params[0]

# unroll once again

movie_features, user_prefs = unroll_params(optimal_movie_features_and_user_prefs, num_users, num_movies, num_features)

# print(movie_features)
# print(user_prefs)


# Make some predictions (movie recommendations). Dot product
all_predictions = movie_features.dot(user_prefs.T)

# In[ ]:
# print("All Predictions")
# print(all_predictions)

# add back the ratings_mean column vector to my (our) predictions
predictions_for_nikhil = all_predictions[:, 0:1] + ratings_mean

print("Predictions for nikHil")
print(predictions_for_nikhil)

# print("nikHil Ratings")
# for rating in nikhil_ratings:
# 	print(rating)
