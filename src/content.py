# content.py

NETWORK_CAPTION = """
Ingredient co-occurrence network across the whole recipes for the 100 most-used ingredients in the dataset. 
Each node/circles represents an ingredient (click "Labels Off" button on top-right to toggle the node labels on),
with node size proportional to the number of recipes containing it. Edges/lines connect
ingredients that frequently appear together, with width/thickness that depends on the number of the corresponding pairs occuring accross recipes. Node colour
encodes the log-scaled weighted degree. See "Insight" for explanation.
"""

NETWORK_INSIGHT_TITLE = "Ingredient Co-occurrence Network"
NETWORK_INSIGHT_SUBTITLE = "Visualizing the Co-Occurrence of the 100 Most-Used Ingredients with Graph Network"

NETWORK_INSIGHT_READ = """
<p>
Each circle or "node" represents an ingredient extracted from the recipe dataset, and the naming of each ingredient (most of them) have been standardized to make meaningful analysis.
The position of nodes is determined by the Fruchterman-Reingold force-directed algorithm, which places ingredients that frequently appear together closer in the graph.
As a result, <strong>ingredients that are commonly used in similar recipes tend to cluster spatially.</strong>
The node size tells the number of recipes containing the ingredient. Larger node means more recipes that contain the corresponding ingredient.
The node color shows the log-scaled weighted degree of the node which defines the number of edges (see next) connected to the nodes. 
Darker nodes represent more edges connected to the corresponding nodes,
meaning ingredients that are more strongly connected to many other ingredients.
Finally, lines or "edges" connect ingredients that appear together in recipes, and the width of each edge corresponds to how frequent the two ingredients co-occur.
</p>
"""

NETWORK_INSIGHT_FINDINGS = """
<p>
Here, we plot 100 ingredients or nodes that contribute to more than 4900 recipes. At the center lies a dense cluster of highly connected ingredients such as flour, sugar, butter, milk, eggs, salt, and vanilla. 
These ingredients act as structural hubs because they appear in many recipes and co-occur with a large number of other ingredients. 
Their strong connectivity results in darker colors (higher log weighted degree) and larger node sizes in the graph. 
Look at salt for example, if you hover its node, you will see that salt is connected to 99 other ingredients, which basically is the entire ingredients plotted!
That is why <strong>salt is ingredient/node with the highest degree in this plot</strong>, signifying its extreme importance among the world of foods.
If we moving outward, the network transitions from these common ingredients to less common ingredients, such as garlic, onion, parsley, 
and paprika, that connect different culinary contexts, and finally to specialized ingredients at the periphery that appear in fewer or more niche recipes.
If more ingredients are plotted, we will see around the periphery even less common recipes like brandy, asparagus, and even rum with smaller nodes and fewer edges than most.
Hence, we can see that this structure highlights how a relatively small set of foundational ingredients supports a much broader and diverse set of recipe combinations.
<strong>So, make sure to at least have sugar, salt, eggs, and flour in your fridge, because with just those basics, an entire world of recipes is already at your fingertips :)</strong>
</p>
"""

NETWORK_INSIGHT_METHOD = """
<p>
For each ingredient node, we first compute its weighted degree, defined as the total strength of its connections to other ingredients in the network. 
Each edge between two ingredients carries a weight equal to the number of recipes in which the pair co-occurs. 
Thus, the weighted degree of an ingredient is calculated by summing the weights of all edges connected to that node, 
representing the total frequency with which that ingredient appears together with other ingredients across the dataset. 
Because these values can vary widely (for example, some ingredients co-occurring thousands of times while others only by a few) the weighted degree is transformed using a logarithmic scaling, 
specifically \( \log(1 + k_w) \), where \(k_w\). 
This transformation compresses the range of values so that highly connected ingredients do not dominate the color scale,
 allowing all ingredients' nodes to remain visually distinguishable in the visualization.
</p>
"""

NETWORK_HL_1_VALUE = "100"
NETWORK_HL_1_LABEL = "Nodes (ingredients)"
NETWORK_HL_2_VALUE = "4914"
NETWORK_HL_2_LABEL = "Edges (co-occur. pairs)"
NETWORK_HL_3_VALUE = "Salt"
NETWORK_HL_3_LABEL = "Highest-degree node"


LEIDEN_CAPTION = """
Ingredient community graph detected via the Leiden algorithm.
Each colour represents a distinct community of ingredients that tend to co-occur together.
All other features are the same as the co-occurrence graph. See "Insight" for explanation.
"""

LEIDEN_INSIGHT_TITLE = "Ingredient Community Graph Using Leiden Algorithm"
LEIDEN_INSIGHT_SUBTITLE = "Community Detection · Leiden Algorithm"

LEIDEN_INSIGHT_WHAT = """
<p>
It builds directly on the other ingredient co-occurrence graph. 
The co-occurrence network already reveals which ingredients frequently appear together and highlights the most connected ingredients in the dataset. 
However, while that graph provides an overview of relationships, it can be difficult to clearly identify distinct ingredient groups simply by visual inspection since
the network is dense, with many overlapping connections, and clusters that appear visually may not be clearly separable. 
Therefore, we use Leiden algorithm, a kind of network clustering algorithm, to provide a systematic way to extract this hidden structure and to  <strong>allow us from simply observing connections to formally identifying ingredient communities </strong>.
</p>
"""

LEIDEN_INSIGHT_READ = """
<p>
While all the other features such as nodes and edges define the same thing as in the co-occurrence graph, here, the color of each node indicates its community membership, meaning that ingredients sharing the same color belong to the same group/community.
These communities are not manually defined categories; instead, they emerge automatically from the structure of the ingredient co-occurrence network. 
The Leiden algorithm identifies these communities by grouping together ingredients that are more strongly connected to each other than to the rest of the network.
</p>
"""

LEIDEN_INSIGHT_FINDINGS = """
<p>
<strong>The resulting communities roughly represents some culinary patterns</strong>, at least for some of the most common ingredients. 
One of the most prominent clusters (darkest color) contains typical baking ingredients such as flour, sugar, eggs, vanilla, and baking powder, and baking soda, along with ingredients like nuts, chocolate chips, and fruit additions. 
These ingredients frequently appear together in cakes, cookies, and other desserts, forming a baking community. 
Another example is a different community (blue) that centers around, what seems to be, Mediterranean-style cooking ingredients such as olive oil, basil, oregano, tomatoes, and mozzarella, emphasize herbs and fresh vegetable-based recipes.
</p>
<p>
The chart below provides a simple summary of the detected community structure by showing how many ingredients belong to each Leiden community. 
A few communities contain a large number of ingredients, while several others are much smaller and even consist of only one ingredient (community C7 and C8: salt and, somehow, heavy cream). 
This uneven distribution reflects how ingredient ecosystems naturally form in the network: a small number of dominant culinary systems 
(such as baking or herb-based cooking) group many ingredients together, while highly versatile or unusual ingredients may end up forming 
very small communities. Note that the number of ingredients that seems to be progressively going down from C0 to C8 is the artefact of community ordering within the code, not the natural structure of the network.
</p>
"""

LEIDEN_INSIGHT_INTERPRET = """
<p>
So, we know now how ingredients cluster and forming  culinary categories... right? Well...
</p>
<p>
While the detected communities reveal meaningful patterns, the clustering is not perfect and should be interpreted cautiously. 
For example, within the baking community there are ingredients that are not typically associated with baking, such as pineapple, 
and some clusters are extremely small; community C7 and C8. These imperfections occur because 
the Leiden algorithm groups ingredients purely based on network structure rather than culinary knowledge. Ingredients that appear 
across many different recipe types, such as salt, can behave differently in the network and sometimes form their own cluster.  <strong>Therefore, 
the detected communities should be interpreted as approximate structural groupings in the ingredient network rather than strict 
culinary categories, which is not necessarily true.</strong> Despite these limitations, the clustering still highlights major ingredient ecosystems such as baking or 
herb based cookin that reflect how recipes tend to organize ingredients in practice.
</p>
"""

LEIDEN_INSIGHT_METHOD = """
<p>
The Leiden algorithm is a community detection method designed to identify groups of nodes in a network that are more densely connected internally than with the rest of the graph. 
It works iteratively through three main stages: moving nodes between communities to improve a quality score, refining the partition to ensure that communities remain internally well connected, 
and then aggregating each community into a new node to repeat the process on a simplified network.
The algorithm typically optimizes a measure called <i>modularity</i>, which evaluates how strong the community structure is compared to a random network. In simplified form, modularity is defined as:
</p>

<p style="text-align:center;">
<i>Q = (1 / 2m) Σ<sub>ij</sub> (A<sub>ij</sub> − (k<sub>i</sub>k<sub>j</sub> / 2m)) δ(c<sub>i</sub>, c<sub>j</sub>)</i>
</p>

<p>
Here, A<sub>ij</sub> represents the connection strength between nodes i and j, k<sub>i</sub> and k<sub>j</sub> are the total connection strengths of those nodes, 
m is the total edge weight of the network, and δ(c<sub>i</sub>, c<sub>j</sub>) equals 1 when the two nodes belong to the same community. 
Maximizing this quantity means that the algorithm finds partitions where connections within communities are stronger than expected by chance.
</p>
"""

# NETWORK_HL_1_VALUE = "100"
# NETWORK_HL_1_LABEL = "Nodes (ingredients)"
# NETWORK_HL_2_VALUE = "4914"
# NETWORK_HL_2_LABEL = "Edges (co-occur. pairs)"
# NETWORK_HL_3_VALUE = "Salt"
# NETWORK_HL_3_LABEL = "Highest-degree node"