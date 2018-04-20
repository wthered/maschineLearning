import json

for line in open("movie_ids_train.json").readlines():
    data = json.loads(line[:-1])
    print("INSERT into movies.titles (title_id, title) values (" + data['id'] + "," + data[''] + data['original_title'] + ");")
