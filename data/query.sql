-- Questions and Tags from 2010 to 2020
-- https://data.stackexchange.com/stackoverflow/query/new

SELECT 
       Posts.Tags as Tags,
       Posts.Title as Title,
       Posts.Body as Body

FROM Posts

WHERE Posts.CreationDate < '2020-01-01' and Posts.CreationDate > '2010-01-01'
      and Posts.Score > 100 and LEN(Posts.Tags) > 0
