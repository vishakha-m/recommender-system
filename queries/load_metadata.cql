# Add constraints
create constraint on (b:Brand) assert b.name is unique
create constraint on (c:Category) assert c.name is unique

# Add data to products
using periodic commit
load csv with headers from "file:///metadata.csv" as metadata
match (p:Product{id:metadata.asin})
set p.name=metadata.title,
    p.price=metadata.price,
    p.image=metadata.imUrl

# Add categories
using periodic commit
load csv with headers from "file:///metadata.csv" as metadata
unwind split(metadata.categories,";") as category
with distinct category
merge (:Category{name: category});

# Add brand
using periodic commit
load csv with headers from "file:///metadata.csv" as metadata
where not metadata.brand is null
merge (:Brand{name: metadata.brand});

# Add relationships
using periodic commit
load csv with headers from "file:///metadata.csv" as metadata
match (p:Product{id:metadata.asin})
match (b:Brand{name:metadata.brand})
create (p)-[:BELNOGS_TO]->(b)
with p,metadata
unwind split(metadata.categories,";") as category
with p,category
match (c:Category{name:category})
create (p)-[:BELONGS_TO]->(c)
