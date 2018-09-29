import schema as sc

# Get the input category and target catgory file name
print("Enter Source Category: ")
source_category = input()

print("\nEnter File for Target Categories: ")
target_path = input()

# Store all the categories in a list
target_file = open(target_path, 'r')
target_category = []
for eachLine in target_file:
    target_category.append(eachLine)
target_file.close()

# Find extended split term set for all the nodes in source category
extendedSet = sc.ExtendedSplitTermSet()
source_extendedSet = extendedSet.getExtendedSplitSet(source_category)

print("\nExtended Split term set: \n" + str(source_extendedSet))

# Select candidates target categories
semantic_matcher = sc.SemanticMatcher()
candidates = semantic_matcher.getCandidate(source_extendedSet, target_category)

# Convert candidates list into a list of category node list
candidate_nodes=[]
for category in candidates:
    c = sc._split_category(category)
    candidate_nodes.append(c)
print("\nCandidates: \n" + str(candidate_nodes))

# Find the best match
pathKey = sc.PathKey(source_category, candidate_nodes, source_extendedSet)
bestCandidate = pathKey.getBestCandidate()
bestScore = pathKey.getBestScore()

print("\nBest Candidates: \n" + str(bestCandidate))
print("\nScores: \n" + str(bestScore))
