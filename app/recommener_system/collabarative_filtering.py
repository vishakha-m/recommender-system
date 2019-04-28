from neo4j import GraphDatabase
from collections import defaultdict
uri = "bolt://neo4j:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "viski123$"))

W_n = 0.2  # Weight for normal similarity
W_b = 0.8  # Weight for brand similarity
W_c = 0.8  # Weight for category similarity
TOP_THRESHOLD = 10
SIM_THRESHOLD = 0.7
BRAND_THRESHOLD = 0.3
CATEGORY_THRESHOLD = 0.3
LIMIT_BRAND = 10
LIMIT_CATEGORY = 10


class User:
    def __init__(self, id):
        self.id = id
        self.categories = set()
        self.brands = set()
        self.neighbours = defaultdict(dict)
        self.recommendations = {}
        with driver.session() as session:
            session.read_transaction(self.__get_neighbours)

    def __get_neighbours(self, tx):
        for record in tx.run("""match(u1:User{id:$id})-[x:REVIEWS]->(p:Product) with count(*) as c1,u1
				match(u2:User)-[x:REVIEWS]->(p:Product)<-[y:REVIEWS]-(u1) 
				where abs(x.score-y.score)<=1 with count(distinct p) as com_prod,c1,u2
				return collect(u2.id) as similar_users,collect(com_prod*1.0/c1) as similarity""",
                             id=self.id):
            for user_id, similarity in zip(record['similar_users'], record['similarity']):
                self.neighbours[user_id]['similarity'] = similarity
                self.neighbours[user_id]['brand_similarity'] = {}
                self.neighbours[user_id]['category_similarity'] = {}
        query = """ match(u1:User{id:$id})-[x:REVIEWS]->(p:Product)-[:BELONGS_TO]->(b:Category)
					with b,sum(x.score)/count(*) as avg_brand_rating
					match (u2:User)-[y:REVIEWS]->(:Product)-[:BELONGS_TO]->(b)
					where u2.id in $similar_users
					with u2.id as user_id,b.name as category_name,1-1.0*(abs(avg_brand_rating-sum(y.score)/count(*)))/5 as category_sim order by u2.id
					return user_id, collect({bn:category_name,cs:category_sim}) as category_similarity"""
        for record in tx.run(query, id=self.id,	similar_users=list(self.neighbours.keys())):
            neighbour = self.neighbours[record['user_id']]
            categories = record['category_similarity']
            for category in categories:
                neighbour['category_similarity'][category['bn']
                                                 ] = category['cs']*W_c+neighbour['similarity']*W_n
                self.categories.add(category['bn'])
        query = """ match(u1:User{id:$id})-[x:REVIEWS]->(p:Product)-[:BELONGS_TO]->(b:Brand)
					with b,sum(x.score)/count(*) as avg_brand_rating
					match (u2:User)-[y:REVIEWS]->(:Product)-[:BELONGS_TO]->(b)
					where u2.id in $similar_users 
					with u2.id as user_id,b.name as brand_name,1-1.0*(abs(avg_brand_rating-sum(y.score)/count(*)))/5 as brand_sim order by u2.id
					return user_id, {bn:brand_name,cs:brand_sim} as brand_similarity"""
        for record in tx.run(query, id=self.id,	similar_users=list(self.neighbours.keys())):
            neighbour = self.neighbours[record['user_id']]
            categories = record['brand_similarity']
            categories = [categories] if type(
                categories) != "list" else categories
            for category in categories:
                neighbour['brand_similarity'][category['bn']
                                              ] = category['cs']*W_b+neighbour['similarity']*W_n
                self.brands.add(category['bn'])

    def get_recommendations(self):
        # Branch 1 : By brands
        def get_sort_key(similarity_type, name):
            def __sort_key(kv):
                return kv[1][similarity_type].get(name, 0)
            return __sort_key

        def get_filter_function(similarity_type, name):
            def __filter_function(kv):
                return kv[1][similarity_type].get(name, 0) >= SIM_THRESHOLD
            return __filter_function
        for brand in self.brands:
            neighbours_sorted = sorted(
                self.neighbours.items(),
                key=get_sort_key('brand_similarity', brand),
                reverse=True
            )[:TOP_THRESHOLD]
            neighbours_sorted = filter(get_filter_function(
                'brand_similarity', brand), neighbours_sorted)
            neighbours_sorted = {x[0]: x[1] for x in neighbours_sorted}
            # print(brand,"Has",neighbours_sorted)

            def __get_brand_items(tx):
                query = """match (u:User{id:$id})-[:REVIEWS]->(p:Product)-[:BELONGS_TO]->(b:Brand{name:$brand}) 
						with collect(p) as prods,b
						match(u:User)-[r:REVIEWS]->(q:Product)-[:BELONGS_TO]->(b)
						where u.id in $users and not q in prods
						return u.id as user_id,q.id as prod_id,r.score as score 
						order by prod_id"""
                product_dict = defaultdict(float)
                for result in tx.run(query, id=self.id, users=list(neighbours_sorted.keys()), brand=brand):
                    product_dict[result['prod_id']] += neighbours_sorted[result['user_id']
                                                                         ]['brand_similarity'][brand] * result['score']
                #print(brand,"Products are",product_dict)
                sorted_dict = sorted(product_dict.items(),
                                     key=lambda x: x[1], reverse=True)
                sorted_dict = filter(
                    lambda x: x[1] > BRAND_THRESHOLD, sorted_dict)
                for product, score in sorted_dict:
                    self.recommendations[product] = score
            with driver.session() as s:
                s.read_transaction(__get_brand_items)
        # Branch 2: By categories
        for category in self.categories:
            neighbours_sorted = sorted(
                self.neighbours.items(),
                key=get_sort_key('category_similarity', category),
                reverse=True
            )[:TOP_THRESHOLD]
            neighbours_sorted = filter(get_filter_function(
                'category_similarity', category), neighbours_sorted)
            neighbours_sorted = {x[0]: x[1] for x in neighbours_sorted}
            # print(category,"Has",neighbours_sorted)

            def __get_category_items(tx):
                query = """match (u:User{id:$id})-[:REVIEWS]->(p:Product)-[:BELONGS_TO]->(b:Category{name:$category}) 
						with collect(p) as prods,b
						match(u:User)-[r:REVIEWS]->(q:Product)-[:BELONGS_TO]->(b)
						where u.id in $users and not q in prods
						return u.id as user_id,q.id as prod_id,r.score as score 
						order by prod_id"""
                product_dict = defaultdict(float)
                for result in tx.run(query, id=self.id, users=list(neighbours_sorted.keys()), category=category):
                    # print("DEBUG:",neighbours_sorted[result['user_id']],result['score'] )
                    product_dict[result['prod_id']] += neighbours_sorted[result['user_id']
                                                                         ]['category_similarity'][category] * result['score']
                #print(category,"Products are",product_dict)
                sorted_dict = sorted(product_dict.items(),
                                     key=lambda x: x[1], reverse=True)
                sorted_dict = filter(
                    lambda x: x[1] > CATEGORY_THRESHOLD, sorted_dict)
                for product, score in sorted_dict:
                    self.recommendations[product] = score
            with driver.session() as s:
                s.read_transaction(__get_category_items)
        # Check if set size < 10
        # If true, get user's top brands and categories
        # Recommend the top items that user has not bought
        if len(self.recommendations.keys()) < 10:
            def __get_similar_items(tx):
                query = """MATCH (u:User{id:$id})-[r:REVIEWS]->(p:Product)-[:BELONGS_TO]->(c:Category)
						with c.name as category,sum(r.score)/count(p) as avg  order by avg desc limit 1 where avg>2 with category,avg 
						MATCH (u:User{id:$id})-[r:REVIEWS]->(p:Product)
						with p,category,avg
						match(q:Product)-[:BELONGS_TO]->(c:Category{name:category}) 
						where p<>q
						return q.id limit  5"""
                for result in tx.run(query, id=self.id):
                    if result['q.id'] not in self.recommendations and result['score'] > 2:
                        self.recommendations[result['q.id']] = result['score']
            with driver.session() as s:
                s.read_transaction(__get_similar_items)
        # Check if set size < 10
        # If true, recommend top 4 items from each brand and category
        if len(self.recommendations.keys()) < 10:
            def __get_top_items(tx):
                query = """MATCH (u:User)-[r:REVIEWS]->(p:Product)-[:BELONGS_TO]->(c:Category)
						   with  c.name as category,p.id as prod ,sum(r.score)/count(p) as avg
						   order by avg desc
						   return category,collect({product:prod,score:avg})[0..2] as top_products"""
                for result in tx.run(query):
                    data = result['top_products']
                    for d in data:
                        self.recommendations[d['product']] = d['score']
                query = """MATCH (u:User)-[r:REVIEWS]->(p:Product)-[:BELONGS_TO]->(c:Brand)
						   with  c.name as category,p.id as prod ,sum(r.score)/count(p) as avg
						   order by avg desc
						   return category,collect({product:prod,score:avg})[0..2] as top_products
						   limit 10"""
                for result in tx.run(query):
                    data = result['top_products']
                    for d in data:
                        self.recommendations[d['product']] = d['score']
            with driver.session() as s:
                s.read_transaction(__get_top_items)
        return sorted(self.recommendations.items(),
                      key=lambda kv: kv[1],
                      reverse=True)

if __name__ == '__main__':
	u = User("A27UF1MSF3DB2")
	res = u.get_recommendations()
	print(res)
