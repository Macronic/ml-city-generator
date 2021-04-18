#!/bin/python

from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import sys

def get_arg(i):
	if len(sys.argv) <= i:
		return None
	return sys.argv[i]

idx_from = get_arg(1)
idx_amount = get_arg(2)

limit_str = ""
if idx_from is not None:
	limit_str += f' OFFSET {idx_from} '

if idx_amount is not None:
	limit_str += f' LIMIT {idx_amount} '

wikidata_api = 'https://query.wikidata.org/sparql'
wikidata_query = ('SELECT ?location ?locationLabel ?coords ?area '
	 'WHERE {'
	 '  ?location wdt:P17 wd:Q142. '
	 '  ?location wdt:P31 ?settlement . '
	 '  ?settlement wdt:P279 wd:Q3266850 . '
	 '  ?location rdfs:label ?de_label . '
	 '  ?location wdt:P625 ?coords . '
	 '  ?location wdt:P2046 ?area '
	 '  FILTER (lang(?de_label) = "en") . '
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


def extract_coords(coord_string):
	coord_string = coord_string.replace('Point(', '')
	coord_string = coord_string.replace(')', '')
	coords = coord_string.split()
	return float(coords[0]), float(coords[1])


for index, row in results_df.iterrows():
	coords = extract_coords(row['coords.value'])
	print(f"City name: {row['locationLabel.value']}, Coords: {coords[0]}, {coords[1]}, Area: {row['area.value']}")

 
