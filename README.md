# chromaprint2vec

The project, for the moment, contains 3 modules:
- `chromaprint_crawler.py`: crawls musicbrainz.org and acoustid.org to generate an index of chromaprints for the selected artists
- `chromaprint2vec.py`: converts the original encoded chromaprints into vectors that can be used to get distances between recordings or to be visualized via embedding-projector
- `chromaprint_crawl_anomalies.py`: crawls musicbrainz and acoustid to detect anomalies, by calculating distances between chromaprints associated to same recording

[Visualize with the embedding-projector](https://muoten.github.io/embedding-projector-standalone/). *Not optimized for mobile devices or portrait mode. A minimum resolution of 1024x768 is required* 

[![Projector example](images/projector_example.png)](https://muoten.github.io/embedding-projector-standalone/)

