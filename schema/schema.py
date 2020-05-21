from __future__ import division

import itertools, re, Levenshtein
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from nltk.corpus import wordnet as wn

def _split_composite(w):
    ''' Splits composite category name w into a set of individual classes:
        a split term set W. '''
    w = w.replace('\n', '')
    m = re.split(', | & | and |/', w)
    return set([s.lower() for s in m])

def _split_category(w):
    ''' Splits the category nodes seperated by a '/' '''
    w = w.replace('\n', '')
    m = w.split('/')
    return [s.lower() for s in m]

def _longest_common_substring(wa, wb):
    ''' Computes the length of the longest common sequence of consecutive
        characters between two strings, corrected for length of the longest
        string, resulting in an index in the range [0, 1] '''
    x = len(wa)
    y = len(wb)

    LCStuff = [[0] * (1 + y) for i in range(1 + x)]
    length, row, col = 0, 0, 0
    for i in range(1, x+1):
        for j in range(1, y+1):
            if(wa[i-1] == wb[j-1]):
                LCStuff[i][j] = LCStuff[i-1][j-1] + 1
                if(length < LCStuff[i][j]):
                    length = LCStuff[i][j]
                    row = i
                    col = j
            else:
                LCStuff[i][j] = 0

    s=''
    while(LCStuff[row][col] != 0):
        s += wa[row-1]
        row -= 1
        col -= 1

    return length/(x if x>y else y)

def _contains_as_separate_component(wa, wb):
    ''' Indicates whether string wa contains string wb as separate part (middle
        of another word is not suffcient) '''
    if wb in wa:
        return True
    return False


# -----------------------------------------------------------------------------
# Find Source Category's Extended Split Term Set
# -----------------------------------------------------------------------------

class ExtendedSplitTermSet(object):
    ''' Generates a split term set for the given category, using parent
        and children as context. '''
        
    def getExtendedSplitSet(self, category):
        ''' Breaks the source category into nodes and find extended split term 
            set for each node in source category
            
            category: (String) Source category
            
            Output: (Set) of disambiguated synonyms of wcategory '''
        source_nodes = _split_category(category)
        extendedSet=[]
        for node in source_nodes:
            context = ''
            for context_node in source_nodes:
                if(context_node == node):
                    continue
                context += ' & ' + context_node
            e = self.split_terms(node, context)
            extendedSet.append(e)
        return extendedSet

    def split_terms(self, wcategory, wcontext):
        ''' 
            wcategory: (String) A single node of the source category
            wcontext: (String) All the nodoes of source category except wcategory
        '''
        Wcontext = _split_composite(wcontext)
        Wcategory = _split_composite(wcategory)

        # Find the extended split term set
        extendedSplitTermSet = set()
        for wsrcSplit in Wcategory:
            extendedTermSet = self.disambiguate(wsrcSplit, Wcontext)
            ext=[]
            if extendedTermSet:
                ext=[l.name() for l in extendedTermSet.lemmas()]
            ext = set(ext)
            ext = ext | set([wsrcSplit])       # Extended split term should always contain the split word
            extendedSplitTermSet = extendedSplitTermSet | ext

        # If disambiguate() doesnt find a synonym, it adds None in the set
        extendedSplitTermSet = set([x for x in extendedSplitTermSet if x is not None])
        return extendedSplitTermSet

    def disambiguate(self, w, Wcontext):
        ''' Disambiguates a word using a set of context words, resulting in
            a set of correct synonyms. '''
        z=self.get_synsets(w)
        bestscore=0
        bestsynset=None
        for s in z:
            sensescore=0
            r=set(self.get_related(s))
            p=itertools.product(r, Wcontext)
            for (sr, w) in p:
                gloss=self.get_gloss(sr)
                sensescore+=_longest_common_substring(gloss, w)
            if sensescore>bestscore:
                bestscore=sensescore
                bestsynset=s
        return bestsynset

    def get_synsets(self, w):
        ''' Gives all synonym sets (representing one sense in WordNet), of
            which word w is a member. '''
        return wn.synsets(w)

    def get_related(self, S):
        ''' Gives synonym sets directly related to synset S in WordNet, based
            on hypernymy, hyponymy, meronymy and holonymy. Result includes
            synset S as well. '''
        related=[S]
        related.extend(S.hypernyms())
        related.extend(S.hyponyms())
        related.extend(S.part_meronyms())
        related.extend(S.part_holonyms())
        return related

    def get_gloss(self, S):
        ''' Returns the gloss associated to a synset S in WordNet. '''
        return S.definition()


# -----------------------------------------------------------------------------
# Semantic Match
# -----------------------------------------------------------------------------

class SemanticMatcher(object):
    ''' Semantic matcher class.'''
    def getCandidate(self, extendedSet, target_category):
        candidates = []
        for category in target_category:
            if self.match(extendedSet, category):
                candidates.append(category)
        return candidates

    def match(self, E, wtarget, tnode=0.7):
        ''' Returns true if a semantic match exists between the
            ExtendedSplitTermSet (E) and wtarget, with a node matching
            threshold specified by tnode.

            E: (Set) extendedSplitTermSet of source category
            wtarget: (String) Full path of target category
            tnode: (Int) Threshold value to select a category

            Output: (Boolean) true if wtarget is an appropriate candidate '''
        Wtarget = _split_composite(wtarget)

        subSetOf = True
        if not E:
            return False
        for SsrcSplit in E:
            matchFound = False
            p=itertools.product(SsrcSplit, Wtarget)
            for (wsrcSplitSyn, wtargetSplit) in p:
                edit_dist = Levenshtein.distance(str(wsrcSplitSyn), str(wtargetSplit))
                similarity = 1 - edit_dist / max(len(wsrcSplitSyn), len(wtargetSplit))
                if _contains_as_separate_component(wtargetSplit, wsrcSplitSyn):
                    matchFound = True
                elif similarity >= tnode:
                    matchFound = True
            if matchFound is False:
                subSetOf = False
        return subSetOf


# -----------------------------------------------------------------------------
# Candidate Target Path Key Comparison
# -----------------------------------------------------------------------------

class PathKey(object):
    ''' Matches the nodes of all source category and the candidate category.
        Similiar nodes are given same key. '''
    def __init__(self, wcategory, candidates, extendedSplitTermSet):
        self.wcategory = _split_category(wcategory)
        self.candidates = candidates
        self.extendedSplitTermSet = extendedSplitTermSet
        self.matchNodes()

    def matchNodes(self):
        source_len = len(self.wcategory)

        # Assign alphabetical a key to each node in source category
        key = ''
        for i in range(97, 97+source_len):
            key += chr(i)
        #print("\nKey: " + str(key))

        # Assign key to similar nodes.
        # Creates a string of keys for each target category
        ex = self.extendedSplitTermSet
        semMatcher = SemanticMatcher()
        i = source_len + 97
        target_pathKey = []
        for category in self.candidates:
            candidateKey=''
            for node in category:
                node_match = False
                for j in range(0, source_len):
                    if semMatcher.match([ex[j]], node, 0.6):
                        node_match = True
                        break
                    #print("ex[j]: " + str(ex[j]) + ",  node: " + node)
                if node_match:
                    candidateKey += chr(j + 97)
                else:
                    candidateKey += chr(i)
                    i += 1
            target_pathKey.append(candidateKey)
        #print("\nPath Key: \n" + str(target_pathKey))

        # Get best 3 matches with their score
        i = 0
        bestScore = [0 for i in range(0, 3)]
        bestCandidate = [0 for i in range(0, 3)]
        for i in range(0, len(target_pathKey)):
            score = self.rank(key, target_pathKey[i])
            for j in range(0, len(bestScore)):
                if score > bestScore[j]:
                    # Shift current values to enter new value
                    bestScore[j+1:3] = bestScore[j:2]
                    bestScore[j] = score

                    bestCandidate[j+1:3] = bestCandidate[j:2]
                    bestCandidate[j] = self.candidates[i]
                    break

        self.bestCandidate = bestCandidate
        self.bestScore = bestScore

    def rank(self, src, tgt):
        ''' Returns the rank of the source and target paths. '''
        p = len(set(tgt) - set(src))
        a = normalized_damerau_levenshtein_distance(str(src), str(tgt)) + p
        b = max(len(src), len(tgt)) + p
        candidateScore = 1 - (a/b)
        return candidateScore

    def getBestCandidate(self):
        return self.bestCandidate

    def getBestScore(self):
        return self.bestScore
