# Metadata should be in current directory with the name "dataset.json"
import csv, re
file = open('metadata.json','r')
sample = eval(file.readline().rstrip())
del sample['related']
del sample['salesRank']
fieldnames=list(sample.keys())  
out = csv.DictWriter(open('metadata.csv','w'),fieldnames)
file = open('metadata.json','r')
out.writeheader()
pat =  re.compile('["\',`]')
for line in file: 
	obj = eval(line.rstrip()) 
	try:
		del_keys = ['related','salesRank','description']
		for del_key in del_keys:
			if del_key in obj:
				del obj[del_key]
		title = obj.get('title','')
		title = title.replace("&lt;", "<")
		title = title.replace("&gt;", ">")
		title = title.replace("&amp;", "&")
		obj['title'] = re.sub(pat, '', title)
		categories = set()
		[[categories.add(x) for x in l] for l in sample['categories']]
		categories.remove('Clothing, Shoes & Jewelry')
		obj['categories'] = ';'.join(categories)
		out.writerow(obj) 
	except Exception as e:
		raise
