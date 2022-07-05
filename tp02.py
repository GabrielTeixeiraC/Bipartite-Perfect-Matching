import numpy as np

def getAdjacencyMatrix(incidenceMatrix):
    """
    Returns the adjacency matrix of the graph.
    """
    N = incidenceMatrix.shape[1]
    M = incidenceMatrix.shape[0]
    adjacencyMatrix = [[] for _ in range(N)]
    for i in range(M):
        edge = incidenceMatrix[i]
        originNode = -1
        destinationNode = -1
        for j in range(N):
            if edge[j] == 1 and originNode == -1:
                originNode = j
            elif edge[j] == 1 and originNode != -1:
                destinationNode = j
                break
            else:
                continue
        adjacencyMatrix[originNode].append(destinationNode)
    return adjacencyMatrix
                


def bfs(adjacencyMatrix, initialNode, A, B):
    N = len(adjacencyMatrix)

    visited = np.zeros(N)
    queue = []
    queue.append(initialNode)
    visited[initialNode] = 1

    edgesAtoB = []

    leafNodes = []

    nodesInA = [initialNode]
    nodesInB = []
    while len(queue) > 0:
        node = queue.pop()
        if len(adjacencyMatrix[node]) == 0:
            leafNodes.append(node)
        for i in range(len(adjacencyMatrix[node])):
            adjacentNode = adjacencyMatrix[node][i]
            if visited[adjacentNode] == 0:
                visited[adjacentNode] = 1
                queue.append(adjacentNode)
                if adjacentNode in A:
                    nodesInA.append(adjacentNode)
                elif adjacentNode in B:
                    nodesInB.append(adjacentNode)
                    edgesAtoB.append((node, adjacentNode))
    return edgesAtoB, leafNodes, nodesInA, nodesInB


def checkForPerfectMatching(incidenceMatrix, A, B):
    """
    Checks if the graph has a perfect matching.
    """
    N = incidenceMatrix.shape[1]
    adjacencyMatrix = getAdjacencyMatrix(incidenceMatrix)
    matchings, matched = getMatchings(A, N, adjacencyMatrix)
    if(len(matchings) == N):
        return 
    setBackwardEdges(adjacencyMatrix, matchings)
    unmatchedNode = getUnmatchedNode(matched)

    edgesAtoB, leafNodes, nodesInA, nodesInB = bfs(adjacencyMatrix, unmatchedNode, A, B)
    if(allLeavesInA(A, leafNodes)):
        return (nodesInA, nodesInB, {}, False, False) 
    else:
        return ([], [], edgesAtoB, True, False)

def findPathToLeaf(parents, unmatchedNode):
    path = []
    while parents[unmatchedNode] != -1:
        path.append(unmatchedNode)
        unmatchedNode = parents[unmatchedNode]
    path.append(unmatchedNode)
    return path

def allLeavesInA(A, leafNodes):
    for i in range(len(leafNodes)):
        if leafNodes[i] not in A:
            return False
    return True

def setBackwardEdges(adjacencyMatrix, matchings):
    for i in range(len(matchings)):
        originNode, destinationNode = matchings[i]
        adjacencyMatrix[destinationNode].append(originNode)

def getUnmatchedNode(matched):
    unmatchedNode = -1
    for i in range(len(matched)):
        if matched[i] == 0:
            unmatchedNode = i
            break
    return unmatchedNode

def getMatchings(A, N, adjacencyMatrix):
    matchings = []
    matched = np.zeros(N)
    for i in range(len(A)):
        originNode = i
        matching = (originNode, )
        if matched[i] == 0:
            for j in range(len(adjacencyMatrix[i])):
                destinationNode = adjacencyMatrix[i][j]
                if matched[destinationNode] == 0:
                    matched[originNode] = 1
                    matched[destinationNode] = 1
                    matching += (destinationNode,)
                    matchings.append(matching)
                    break
    return matchings, matched
    
def updateYVector(incidenceMatrix, secondaryGraphEdges, edgeWeights, yVector, S, NHS):
    M = incidenceMatrix.shape[0]

    epsilon = np.inf
    for i in range(M):
        if i in secondaryGraphEdges:
            continue
        edge = incidenceMatrix[i]
        for j in range(len(S)):
            if edge[S[j]] == 1:
                epsilon = min(epsilon, edgeWeights[i] - np.dot(edge, yVector))
    yVector[S] += epsilon
    yVector[NHS] -= epsilon

N, M = input().split()
N = int(N)
M = int(M)

incidenceMatrix = []

for i in range(2 * N):
    incidenceMatrix.append(input().split())

incidenceMatrix = np.array(incidenceMatrix, dtype=float)

bipartiteA = np.array(list(range(N)))
bipartiteB = np.array([list(range(N, 2 * N))])

incidenceMatrix = incidenceMatrix.T

edgeWeights = []
edgeWeights = input().split()
edgeWeights = np.array(edgeWeights, dtype=float)

yVector = np.zeros(2 * N)

viableWeight = np.min(edgeWeights)/2
yVector[:] = viableWeight

while True:
    secondaryGraphEdges = []

    for i in range(M):
        if np.dot(incidenceMatrix[i], yVector) == edgeWeights[i]:
            secondaryGraphEdges.append(i)

    secondaryGraph = incidenceMatrix[secondaryGraphEdges]

    S, NHS, matching, perfectMatchingExists, unfeasible = checkForPerfectMatching(secondaryGraph, bipartiteA, bipartiteB)

    if(perfectMatchingExists):
        print("YES")
        print(matching)
        break
    elif(unfeasible):
        print("NO")
        break
    else:
        updateYVector(incidenceMatrix, secondaryGraphEdges, edgeWeights, yVector, S, NHS)
