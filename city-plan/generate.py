#!/bin/python

from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import sys
import os
import time
import subprocess

def get_arg(i):
	if len(sys.argv) <= i:
		return None
	return sys.argv[i]

idx_from = int(get_arg(1))
idx_amount = int(get_arg(2))
path = get_arg(3)
if path is None:
	path = "out"

def download_cities(idx_from, idx_amount, max_on_iteration=3000):
	def extract_coords(coord_string):
		coord_string = coord_string.replace('Point(', '')
		coord_string = coord_string.replace(')', '')
		coords = coord_string.split()
		return float(coords[0]), float(coords[1])
	
	data = None
	while idx_amount > 0:
		current = min(idx_amount, max_on_iteration)
		idx_amount = idx_amount - current
		limit_str = ""
		if idx_from is not None:
			limit_str += f' OFFSET {idx_from} '
			idx_from += current
		else:
			idx_from = current

		limit_str += f' LIMIT {current} '

		wikidata_api = 'https://query.wikidata.org/sparql'
		wikidata_query = ('SELECT ?location ?locationLabel ?coords ?area ?countryLabel '
			 'WHERE {'
			 '  ?location wdt:P31 ?settlement . '
			 '  ?settlement wdt:P279 wd:Q3266850 . '
			 '  ?location wdt:P625 ?coords . '
			 '  ?location wdt:P2046 ?area . '
			 '  ?location wdt:P17 ?country '
			 '  FILTER (?area > 20) .'
			 '  SERVICE wikibase:label {bd:serviceParam wikibase:language "en". }'
			 '  } '
			 + limit_str
			 )

		sparql = SPARQLWrapper(wikidata_api)
		sparql.setQuery(wikidata_query)
		sparql.setReturnFormat(JSON)
		results = sparql.query().convert()
		results_df = pd.json_normalize(results['results']['bindings'])
		if data is None:
			data = results_df
		else:
			data = pd.concat([data, results_df])
	
	res = pd.DataFrame()
	res['name'] = data['locationLabel.value']
	res['coords'] = data.apply(lambda row : extract_coords(row['coords.value']), axis=1)
	res['area'] = data['area.value']
	res['country'] = data['countryLabel.value']
	return res

if not os.path.exists(path):
	os.mkdir(path)
for name, (cx, cy), area, country in download_cities(idx_from, idx_amount).itertuples(index=False):
	subprocess.run(["./generate_images.py", f"{path}/{country}-{name}.png", str(cy-0.1), str(cx-0.1), "0.2", "0.2", "256", "17.1"])
	time.sleep(30)

