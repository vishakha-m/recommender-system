# loading cloth.csv dataset to neo4j database
using periodic commit
load csv with headers from "file:///cloth.csv" as review
merge(u:User{id:review.reviewerID}) on create set u.name=review.reviewerName
merge(p:Product{id:review.asin})
merge (u)-[r:REVIEWS{time:review.unixReviewTime}]->(p)
on create set r.summary=review.summary,
        r.score=review.overall,
        r.helpful=review.helpful,
        r.text=review.reviewText

#converting string atrribute to integer
match ()-[r:REVIEWS]->() set r.score=tointeger(r.score)
