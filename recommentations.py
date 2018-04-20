from math import sqrt

# A dictionary of movie critics and their ratings of a small set of movies
critics = {
	'Lisa Rose': {
		'Lady in the Water': 2.5,
		'Snakes on a Plane': 3.5,
		'Just My Luck': 3.0,
		'Superman Returns': 3.5,
		'You, Me and Dupree': 2.5,
		'The Night Listener': 3.0,
	},
	'Gene Seymour': {
		'Lady in the Water': 3.0,
		'Snakes on a Plane': 3.5,
		'Just My Luck': 1.5,
		'Superman Returns': 5.0,
		'The Night Listener': 3.0,
		'You, Me and Dupree': 3.5,
	},
	'Michael Phillips': {
		'Lady in the Water': 2.5,
		'Snakes on a Plane': 3.0,
		'Superman Returns': 3.5,
		'The Night Listener': 4.0,
	},
	'Claudia Puig': {
		'Snakes on a Plane': 3.5,
		'Just My Luck': 3.0,
		'The Night Listener': 4.5,
		'Superman Returns': 4.0,
		'You, Me and Dupree': 2.5,
	},
	'Mick LaSalle': {
		'Lady in the Water': 3.0,
		'Snakes on a Plane': 4.0,
		'Just My Luck': 2.0,
		'Superman Returns': 3.0,
		'The Night Listener': 3.0,
		'You, Me and Dupree': 2.0,
	},
	'Jack Matthews': {
		'Lady in the Water': 3.0,
		'Snakes on a Plane': 4.0,
		'The Night Listener': 3.0,
		'Superman Returns': 5.0,
		'You, Me and Dupree': 3.5,
	},
	'Toby': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 'Superman Returns': 4.0},
}


def test():
	print("Testing")
	print(similarItems(loadMovieLens()))
	return 0


def sim_distance(preferences, p1, p2):
	# Returns a distance-based similarity score for person1 and person2.

	sum_of_squares = 0
	# Get the list of shared_items
	si = {}
	for item in preferences[p1]:
		if item in preferences[p2]:
			si[item] = 1
	# If they have no ratings in common, return 0
	if len(si) == 0:
		return 0
	# Add up the squares of all the differences
	for item in preferences[p1]:
		if item in preferences[p2]:
			sum_of_squares = sum([pow(preferences[p1][item] - preferences[p2][item], 2)])
	return 1 / (1 + sqrt(sum_of_squares))


def sim_pearson(preferences, p1, p2):
	# Returns the Pearson correlation coefficient for p1 and p2.

	# Get the list of mutually rated items
	si = {}
	for item in preferences[p1]:
		if item in preferences[p2]:
			si[item] = 1
	# If they are no ratings in common, return 0
	if len(si) == 0:
		return 0
	# Sum calculations
	n = len(si)
	# Sums of all the preferences
	sum1 = sum([preferences[p1][it] for it in si])
	sum2 = sum([preferences[p2][it] for it in si])
	# Sums of the squares
	sum1root = sum([pow(preferences[p1][it], 2) for it in si])
	sum2root = sum([pow(preferences[p2][it], 2) for it in si])
	# Sum of the products
	prefsum = sum([preferences[p1][it] * preferences[p2][it] for it in si])
	# Calculate r (Pearson score)
	num = prefsum - sum1 * sum2 / n
	den = sqrt((sum1root - pow(sum1, 2) / n) * (sum2root - pow(sum2, 2) / n))
	if den == 0:
		return 0
	r = num / den
	return r


def topmatches(
		prefs,
		person,
		n=5,
		similarity=sim_pearson,
):
	# Returns the best matches for person from the prefs dictionary.
	# Number of results and similarity function are optional params.

	scores = [(similarity(prefs, person, other), other) for other in prefs if other != person]
	scores.sort()
	scores.reverse()
	return scores[0:n]


def recommendations(preferences, person, similarity=sim_pearson):
	# Gets recommendations for a person by using a weighted average
	# of every other user's rankings

	totals = {}
	simulationsums = {}
	for other in preferences:
		# Don't compare me to myself
		if other == person:
			continue
		sim = similarity(preferences, person, other)
		# Ignore scores of zero or lower
		if sim <= 0:
			continue
		for item in preferences[other]:
			# Only score movies I haven't seen yet
			if item not in preferences[person] or preferences[person][item] == 0:
				# Similarity * Score
				totals.setdefault(item, 0)
				# The final score is calculated by multiplying each item by the
				#   similarity and adding these products together
				totals[item] += preferences[other][item] * sim
				# Sum of similarities
				simulationsums.setdefault(item, 0)
				simulationsums[item] += sim
	# Create the normalized list
	rankings = [(total / simulationsums[item], item) for (item, total) in totals.items()]
	# Return the sorted list
	rankings.sort()
	rankings.reverse()
	return rankings


def transform(preferences):
	# Transform the recommendations into a mapping where persons are described
	# with interest scores for a given title e.g. {title: person} instead of
	# {person: title}.

	result = {}
	for person in preferences:
		for item in preferences[person]:
			result.setdefault(item, {})
			# Flip item and person
			result[item][person] = preferences[person][item]
	return result


def similarItems(preferences, n=10):
	# Create a dictionary of items showing which other items they are
	# most similar to.

	result = {}
	# Invert the preference matrix to be item-centric
	itemPreferences = transform(preferences)
	c = 0
	for item in itemPreferences:
		# Status updates for large data sets
		c += 1
		if c % 100 == 0:
			print('%d / %d' % (c, len(itemPreferences)))
		# Find the most similar items to this one
		scores = topmatches(itemPreferences, item, n=n, similarity=sim_distance)
		result[item] = scores
	return result


def getitems(prefs, itemMatch, user):
	ratings = prefs[user]
	scores = {}
	simulation = {}
	# Loop over items rated by this user
	for (item, rating) in ratings.items():
		# Loop over items similar to this one
		for (similarity, item2) in itemMatch[item]:
			# Ignore if this user has already rated this item
			if item2 in ratings:
				continue
			# Weighted sum of rating times similarity
			scores.setdefault(item2, 0)
			scores[item2] += similarity * rating
			# Sum of all the similarities
			simulation.setdefault(item2, 0)
			simulation[item2] += similarity
	# Divide each total score by total weighting to get an average
	rankings = [(score / simulation[item], item) for (item, score) in scores.items()]
	# Return the rankings from highest to lowest
	rankings.sort()
	rankings.reverse()
	return rankings


def loadMovieLens(path='/data/movielens'):
	# Get movie titles
	movies = {}
	for line in open(path + '/u.item'):
		(identifier, title) = line.split('|')[0:2]
		movies[identifier] = title
	# Load data
	preferences = {}
	for line in open(path + '/u.data'):
		(user, movieidentifier, rating, ts) = line.split('\t')
		preferences.setdefault(user, {})
		preferences[user][movies[movieidentifier]] = float(rating)
	return preferences


test()
