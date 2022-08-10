# Bipartite Perfect Matching
This algorithm finds a minimum weight perfect matching in a bipartite graph using linear programmings and BFS.\
If there is a perfect matching, returns its weight, the used edges and a certificate that it is perfect, which is a solution to the dual linear programming, with equal objective function value.\
If there is not a perfect matching in the graph, returns a certificate consisting of a set of edges in A that has fewer nenighbors in B than members.

## Example 1:
Graph with 4 nodes (1 and 2 in A, 3 and 4 in B) and 3 edges: 
  - 1 - 3 (weight 2)
  - 1 - 4 (weight 3)
  - 2 - 3 (weight 5)
### Input: 
    2 3
    1 1 0
    0 0 1
    1 0 1
    0 1 0
    2 3 5

### Output
    8
    0 1 1
    3 6 -1 0
    
## Example 2:
Graph with 4 nodes (1 and 2 in A, 3 and 4 in B) and 2 edges: 
  - 1 - 3 (weight 2)
  - 2 - 3 (weight 3)
### Input: 
    2 2
    1 0
    0 1
    1 1
    0 0
    2 3

### Output
    -1
    1 1 
    1 0 

