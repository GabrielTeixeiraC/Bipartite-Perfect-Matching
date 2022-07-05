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

def bfs(adjacencyMatrixOfH, adjacencyMatrixOfG, initialNode, A, B):
    N = len(adjacencyMatrixOfH)

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
        if len(adjacencyMatrixOfH[node]) == 0:
            leafNodes.append(node)
        for i in range(len(adjacencyMatrixOfH[node])):
            adjacentNode = adjacencyMatrixOfH[node][i]
            if visited[adjacentNode] == 0:
                visited[adjacentNode] = 1
                queue.append(adjacentNode)
                if adjacentNode in A:
                    nodesInA.append(adjacentNode)
                elif adjacentNode in B:
                    nodesInB.append(adjacentNode)
                    edgesAtoB.append((node, adjacentNode))
    neighborsInG = getNeighborsInG(adjacencyMatrixOfG, nodesInA)
    if (len(nodesInA) > len(neighborsInG)):
        return ([], [], nodesInA, neighborsInG, True)
    return (edgesAtoB, leafNodes, nodesInA, nodesInB, False)

def getNeighborsInG(adjacencyMatrix, nodes):
    neighbors = []
    visited = np.zeros(len(adjacencyMatrix))
    for i in range(len(nodes)):
        for j in range(len(adjacencyMatrix[nodes[i]])):
            if visited[adjacencyMatrix[nodes[i]][j]] == 0:
                neighbors.append(adjacencyMatrix[nodes[i]][j])
                visited[adjacencyMatrix[nodes[i]][j]] = 1
    return neighbors

def checkForPerfectMatching(incidenceMatrix, A, B, adjacencyMatrixOfG):
    """
    Checks if the graph has a perfect matching.
    """
    N = incidenceMatrix.shape[1]
    adjacencyMatrix = getAdjacencyMatrix(incidenceMatrix)
    matchings, matched = getMatchings(A, N, adjacencyMatrix)
    if(len(matchings) == N//2):
        return ([], [], matchings, True, False)
    setBackwardEdges(adjacencyMatrix, matchings)
    unmatchedNode = getUnmatchedNode(matched)

    edgesAtoB, leafNodes, nodesInA, nodesInB, unfeasible = bfs(adjacencyMatrix, adjacencyMatrixOfG, unmatchedNode, A, B)

    if (unfeasible):
        return (nodesInA, nodesInB, [], False, True)
    elif (allLeavesInA(A, leafNodes)):
        return (nodesInA, nodesInB, [], False, False) 
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

adjacencyMatrixOfG = getAdjacencyMatrix(incidenceMatrix)
def printVector(vector):
    for i in range(len(vector)):
        print(int(vector[i]), end=" ")
    print()

def getEdgesFromIncidenceMatrix(N, M, incidenceMatrix, matching):
    edges = []
    for i in range(len(matching)):
        edge = np.zeros(2 * N)
        originNode, destinationNode = matching[i]
        edge[originNode] = 1
        edge[destinationNode] = 1
        edges.append(edge)
    edgesIncidenceMatrix = np.zeros(M)
    for i in range(M):
        for j in range(len(edges)):
            if (incidenceMatrix[i] == edges[j]).all():
                edgesIncidenceMatrix[i] = 1
    return edgesIncidenceMatrix

def getCertificate(N, setS, neighborsOfS):
    certificate = np.zeros(2 * N)
    for i in range(len(setS)):
        certificate[setS[i]] = 1
    for i in range(len(neighborsOfS)):
        certificate[neighborsOfS[i]] = 1
    return certificate

while True:
    secondaryGraphEdges = []

    for i in range(M):
        if np.dot(incidenceMatrix[i], yVector) == edgeWeights[i]:
            secondaryGraphEdges.append(i)

    secondaryGraph = incidenceMatrix[secondaryGraphEdges]

    setS, neighborsOfS, matching, perfectMatchingExists, unfeasible = checkForPerfectMatching(secondaryGraph, bipartiteA, bipartiteB, adjacencyMatrixOfG)

    if(perfectMatchingExists):
        edgesIncidenceMatrix = getEdgesFromIncidenceMatrix(N, M, incidenceMatrix, matching)
        optimalValue = np.dot(np.ones(2 * N), yVector)
        print(int(optimalValue))
        printVector(edgesIncidenceMatrix)
        printVector(yVector)
        break
    elif(unfeasible):
        print("-1")
        certificate = getCertificate(N, setS, neighborsOfS)
        printVector(certificate[:N])
        printVector(certificate[N:])
        break
    else:
        updateYVector(incidenceMatrix, secondaryGraphEdges, edgeWeights, yVector, setS, neighborsOfS)
