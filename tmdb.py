import json

for line in open("movie_ids_train.json").readlines():
	data = json.loads(line[:-1])
	title = str(data['id'])
	movie = str(data['original_title'])
	print("INSERT into movies.titles (title_id, title) values (" + title + ",'" + movie + "');")
