# AnimalsPCA
Data Transformation of Animal Dataset for CPSC 340- Machine Learning and Data Mining

Loading a <a href="https://github.com/jfwk22/AnimalsPCA/blob/673a3d0c2553f3513f32c42a5f124d0646b90312/a5/animals.csv"> dataset </a>  containing 50 examples, each representing an animal. The
85 features are traits of these animals. The script standardizes these features and gives two unsatisfying
visualizations of it. First it shows a plot of the matrix entries, which has too much information and thus
gives little insight into the relationships between the animals. Next it shows a scatterplot based on two
random features and displays the name of 10 randomly-chosen animals.

<img src="https://user-images.githubusercontent.com/16784008/212438781-776b3bc3-e5de-43e9-bd47-055be1936900.png" width="500" height="300">

The points that are labeled could vary, through standarization or not. Roughly it looks like as we move from the left to the
right we go from terrestrial to aquatic animals, and we also see an steadily increase in size. However, there are definitely exceptions to these rules that the higher-order principal components might capture, and the "crowding" effect places some very disimilar animals next
to each other.) 

Hence we will then apply gradient descent to minimize the following multi-dimensional scaling (MDS) objective:

<img src="https://user-images.githubusercontent.com/16784008/212440084-cb9c50d3-27ea-44e3-9c93-4e70b96c9e5b.png" width="500" height="300">

Although this visualization isn’t perfect (with “gorilla” being placed close to the dogs and “otter” being
placed close to two types of bears), this visualization does organize the animals in a mostly-logical way.
Euclidean distances between very different animals are unlikely to be particularly meaningful. However,
since related animals tend to share similar traits we might expect the animals to live on a low-dimensional
manifold. This suggests that ISOMAP may give a better visualization:


<img src="https://user-images.githubusercontent.com/16784008/212440300-a11ce022-ea5f-4f12-a062-b23b4b8a7511.png" width="500" height="300">

An issue with measuring distances on graphs is that the graph may not be connected. . One heuristic
to address this is to set these infinite distances to the maximum distance in the graph (i.e., the maximum
geodesic distance between any two points that are connected), which will encourage non-connected points
to be far apart:

<img src="https://user-images.githubusercontent.com/16784008/212440422-cd993c22-40f6-41b5-9ec0-b0de81f3d1ec.png" width="500" height="300">

