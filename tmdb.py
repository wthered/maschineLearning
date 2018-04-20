import json

for line in open("movie_ids_train.json").readlines():
	data = json.loads(line[:-1])
	title = str(data['id'])
	print("INSERT into movies.titles (title_id, title) values (" + title + ",'" + str(data['original_title']) + "');")
