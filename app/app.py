from flask import Flask, jsonify, request,render_template
from neo4j import GraphDatabase
import json


app = Flask(__name__,static_url_path='/static')
uri = "bolt://neo4j:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "viski123$"))


"""
{
    "limit": 20
}
"""
@app.route("/getUsers", methods=['POST'])
def get_users():
    data = json.loads(request.data)
    if "limit" not in data:
        return jsonify({
            "success": False,
            "message": "Invalid request, parameter limit either not found or invalid"
        })
    results = []

    def _query(tx):
        for record in tx.run("match (n:User) return (n) limit $l", l=data["limit"]):
            results.append({
                'id': record['n']['id'],
                'name': record['n']['name']
            })
    with driver.session() as session:
        session.read_transaction(_query)
    return jsonify({
        "success": True,
        "data": results
    })


"""
{
    "limit": 20
}
"""
@app.route("/getProducts", methods=['POST'])
def get_products():
    data = json.loads(request.data)
    if "limit" not in data:
        return jsonify({
            "success": False,
            "message": "Invalid request, parameter limit either not found or invalid"
        })
    results = []

    def _query(tx):
        for record in tx.run("match (n:Product) return (n) limit $l", l=data["limit"]):
            results.append({
                'id': record['n']['id']
            })
    with driver.session() as session:
        session.read_transaction(_query)
    return jsonify({
        "success": True,
        "data": results
    })


"""
{
    "limit": 20
}
"""
@app.route("/getUserRatings", methods=['POST',"GET"])
def get_user_ratings():
   # data = json.loads(request.data)
    data={
       "id":"A1UUFPECPCLDXL",
       "limit":10
       }
    #if "id" not in data:
        #return jsonify({
            #"success": False,
            #"message": "Invalid request, parameter id either not found"
       # })
    if "limit" not in data:
        data["limit"] = 10
    results = []

    def _query(tx):
        import sys
        for record in tx.run("match (n:User{id:$id})-[r:REVIEWS]->(p:Product) return r,p limit $l", id=data["id"], l=data["limit"]):
            results.append({
                'product_id': record['p']['id'],
                'rating': record['r']['score']
            })
    with driver.session() as session:
        session.read_transaction(_query)
    return jsonify({
        "success": True,
        "data": results
    })


@app.route("/getProductStats", methods=['POST'])
def get_product_stats():
    data = json.loads(request.data)
    if "id" not in data:
        return jsonify({
            "success": False,
            "message": "Invalid request, parameter id either not found"
        })
    if "limit" not in data:
        data["limit"] = 10
    results = []

    def _query(tx):
        query = "match (n:User)-[r:REVIEWS]->(p:Product{id:$id})  return n.id,r.score,r.text limit 5"
        for record in tx.run(query, id=data["id"]):
            results.append({
                'id':record['n.id'],
                'score':record['r.score'],
                'text':record['r.text']
            })
           
    with driver.session() as session:
        session.read_transaction(_query)
    if results=={}:
        return jsonify({
            "success":False,
            "message":"Invalid Product_id"
        })
    return jsonify({
        "success": True,
        "data": results
    })
 
@app.route("/Recommend", methods=['POST',"GET"])
def get_Recommends():
    data = json.loads(request.data)
    #data={
        #"id":"AMO214LNFCEI4",
        #"limit":10
   # }
    if "limit" not in data:
        return jsonify({
            "success": False,
            "message": "Invalid request, parameter limit either not found or invalid"
        })
    results = []
    
    def _query(tx):
        query1 = """MATCH (m1:Product)<-[:REVIEWS]-(u1:User{id:$id}) WITH count(m1) as countm MATCH (u2:User)-[r2:REVIEWS]->(m1:Product)<-[r1:REVIEWS]-(u1:User{id:$id}) WHERE (NOT u2=u1) AND (abs(r2.score - r1.score) <= 1) WITH u1, u2, tofloat(count(DISTINCT m1))/countm as sim WHERE sim>0.1 MATCH (m:Product)<-[r:REVIEWS]-(u2) WHERE (NOT (m)<-[:REVIEWS]-(u1)) WITH DISTINCT m, count(r) as n_u, tofloat(sum(r.score)) as sum_r WHERE n_u > 1 RETURN m, (sum_r/n_u)/6 as score limit $l"""
        for record in tx.run(query1,l=data['limit'],id=data["id"]):
            results.append({
                'product_id':record["m"]["id"],
                'predection_score':record["score"]
            })
    with driver.session() as session:
        session.read_transaction(_query)
    return jsonify({
        "success": True,
        "data": results
    })

@app.route("/getSimilaruser", methods=['POST'])
def get_similar_user():
    data = json.loads(request.data)
    if "limit" not in data:
        return jsonify({
            "success": False,
            "message": "Invalid request, parameter limit either not found or invalid"
        })
    results = []

    def _query(tx):
        querry="""match (u1:User{id:$id})-[x:REVIEWS]->(p:Product),(u2:User)-[y:REVIEWS]->(p) with u2.id as neighbour,count(*) as com,u1,u2  match(u2)-[:REVIEWS]-() with count(*) as c2,neighbour,u1,com match(:User{id:$id})-[:REVIEWS]-() with count(*) as c1,com,u1,c2,neighbour  return u1.id,neighbour,com*1.0/(c1+c2-com) as sim order by sim desc limit $l"""
        for record in tx.run(querry,id=data['id'] ,l=data["limit"]):
            results.append({
                'id': record['neighbour'],
                'sim':record['sim']
            })
    with driver.session() as session:
        session.read_transaction(_query)
    return jsonify({
        "success": True,
        "data": results
    })

@app.route("/getranduser", methods=['POST'])
def get_rand():
    data = json.loads(request.data)
    if "limit" not in data:
        return jsonify({
            "success": False,
            "message": "Invalid request, parameter limit either not found or invalid"
        })
    results = []

    def _query(tx):
        for record in tx.run("match (n:User) return (n) limit $l", l=data["limit"]):
            results.append({
                'id': record['n']['id']
                
            })
    with driver.session() as session:
        session.read_transaction(_query)
    return jsonify({
        "success": True,
        "data": results
    })

@app.route("/Ratingstat", methods=['POST',"GET"])
def get_Rating():
    data = json.loads(request.data)
    #data={
        #"id":"AMO214LNFCEI4",
        #"limit":10
   # }
  
    results = []
    
    def _query(tx):
        query1 = """match (n:User)-[r:REVIEWS]->(p:Product{id:$id}) return r.score,collect(n.id),count(*)"""
        for record in tx.run(query1,id=data["id"]):
           results.append({
               'score':record['r.score'],
               'count':record['count(*)']
                        })
    with driver.session() as session:
        session.read_transaction(_query)
    return jsonify({
        "success": True,
        "data": results
    })

'''@app.route('/senti')
def senti_analysis(review):
    results = []
    if tf.test.is_gpu_available():
        BATCH_SIZE = 128
        EPOCHS = 2
        VOCAB_SIZE = 30000
        MAX_LEN = 500
        EMBEDDING_DIM = 40 # Dimension of word embedding vector
    
    # Hyperparams for CPU training
    else:
        BATCH_SIZE = 32
        EPOCHS = 2
        VOCAB_SIZE = 20000
        MAX_LEN = 90
        EMBEDDING_DIM = 40
    
    DS_PATH = '/home/vishakha/Documents/final_project/final_project'  # ADD path/to/dataset
    LABELS = ['negative', 'positive']

    # Load data
    train = pd.read_csv(os.path.join(DS_PATH, "train.tsv"), sep='\t')  # EDIT WITH YOUR TRAIN FILE NAME
    val = pd.read_csv(os.path.join(DS_PATH, "val.tsv"), sep='\t')
    imdb_tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    imdb_tokenizer.fit_on_texts(train['text'].values)

    x_train_seq = imdb_tokenizer.texts_to_sequences(train['text'].values)
    x_val_seq = imdb_tokenizer.texts_to_sequences(val['text'].values)

    x_train = sequence.pad_sequences(x_train_seq, maxlen=MAX_LEN, padding="post", value=0)
    x_val = sequence.pad_sequences(x_val_seq, maxlen=MAX_LEN, padding="post", value=0)

    y_train, y_val = train['label'].values, val['label'].values

    NUM_FILTERS = 250
    KERNEL_SIZE = 3
    HIDDEN_DIMS = 250

    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN))
    model.add(Dropout(0.2))

    model.add(Conv1D(NUM_FILTERS,
                 KERNEL_SIZE,
                 padding='valid',
                 activation='relu',
                 strides=1))
    
    model.add(GlobalMaxPooling1D())
    model.add(Dense(HIDDEN_DIMS))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.load_weights('/home/vishakha/Documents/final_project/final_project/cnn_sentiment_weights.h5')

    review_np_array = imdb_tokenizer.texts_to_sequences([review])
    review_np_array = sequence.pad_sequences(review_np_array, maxlen=self.MAX_LEN, padding="post", value=0)
       
    score = self.model.predict(review_np_array)[0][0]
    prediction = self.LABELS[self.model.predict_classes(review_np_array)[0][0]]

    results.append({
        'Prediction':prediction,
        'Score':score
    })
    #print('REVIEW:', review, '\nPREDICTION:', prediction, '\nSCORE: ', score)
    return jsonify({
        "success": True,
        "data": results
    })'''

@app.route("/")
def hello():
    return render_template("main.html")

@app.route("/forward/", methods=['POST'])
def nextPage():
    return render_template("User.html")
@app.route("/forward1/", methods=['POST'])
def nextPage1():
    return render_template("Product.html")

@app.route("/forward2/", methods=['POST'])
def nextPage2():
    return render_template("example.html")


if __name__ == '__main__':
    app.run(port=8080, host="0.0.0.0", debug=True)
