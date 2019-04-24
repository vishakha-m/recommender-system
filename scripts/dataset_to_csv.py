# Dataset should be in current directory with the name "dataset.json"
import csv,json
import  re
file = open('dataset.json','r')
sample = json.loads(file.readline().rstrip())
del sample['reviewTime']
fieldnames=list(sample.keys())  
out = csv.DictWriter(open('reviews.csv','w'),fieldnames)
file = open('dataele.json','r')
out.writeheader()
pat = re.compile('[^A-Za-z0-9\s]+')
for line in file: 
	obj = json.loads(line.rstrip()) 
	try:
		del obj['reviewTime']
		obj['summary'] = re.sub(pat, '', obj['summary'])
		obj['reviewText'] = re.sub(pat, '', obj['reviewText'])
		obj['helpful'] = obj['helpful'][0]/(obj['helpful'][0]+1)
		out.writerow(obj) 
	except KeyError:
		print(obj,line)
