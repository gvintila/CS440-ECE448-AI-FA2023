"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    
    word_tags = {}
    tags = {}
    # Iterate through sentences 
    for sen in train:
        # Iterate through pairs of words and tags
        for pair in sen:
            if pair[0] not in word_tags:
                # Make each word have a dictionary containing the count for each tag
                word_tags[pair[0]] = {}

            if pair[1] not in tags:
                tags[pair[1]] = 0
                word_tags[pair[0]][pair[1]] = 0
            elif pair[1] not in word_tags[pair[0]]:
                word_tags[pair[0]][pair[1]] = 0

            tags[pair[1]] += 1
            word_tags[pair[0]][pair[1]] += 1

    # Find the most commonly used tag and make it the default tag for unseen words
    high_tag = ("", 0) # (Tag, Count)
    for tag, count in tags.items():
        if count > high_tag[1]:
            high_tag = (tag, count)
    unseen_word_tag = high_tag[0]

    # Find the most commonly used tag for each word
    for word in word_tags:
        high_tag = ("", 0)
        for tag, count in word_tags[word].items():
            if count > high_tag[1]:
                high_tag = (tag, count)
        # Replace word_tag dict with the most commonly used word
        word_tags[word] = high_tag[0]     

    # Parse through the test data and return word tag pairs
    output = []
    for sen in test:
        output.append([])
        for word in sen:
            if word not in word_tags:
                output[-1].append((word, unseen_word_tag))
            else:
                output[-1].append((word, word_tags[word])) # (Word, Tag)
            
    return output