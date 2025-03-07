-- SQLite
SELECT type AS tipo, COUNT(*) as quantidade
FROM anon_pairs
GROUP BY type;
