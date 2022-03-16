import json
import numpy as np
f = open("most_common_patterns16-36-43.json")
x = json.load(f)
epsilon_to_size = dict()
epsilon_to_jaccard = dict()

for epsilon in list(x.keys()):#assume keys are epsilon
    for class1 in x[epsilon].keys():#assume keys are class labels
        for class2 in x[epsilon].keys():
            if class1 != class2:
                
                s1 = set(x[epsilon][class1][0]) #get stable indicies 
                s2 = set(x[epsilon][class2][0])
                
                combined = s1.intersection(s2) #calculare jaccard distance j(A,B) = (A^B)/(A U B)
                jaccard = len(combined)/len(s1.union(s2))
                
                if not epsilon in epsilon_to_size.keys():
                    epsilon_to_size[epsilon] = []
                epsilon_to_size[epsilon] += [len(combined)]
                
                if not epsilon in epsilon_to_jaccard.keys():
                    epsilon_to_jaccard[epsilon] = []
                epsilon_to_jaccard[epsilon] += [jaccard]
                
                #print("with epsilon = {} : {} and {} intersction is {} of size {} comapred to sizes of {} and {} for s1 and s2".format(key, k1, k2, combined, len(combined), len(s1), len(s2)))
                print("with epsilon = {} : {} and {} intersction is of size {} comapred to sizes of {} and {} for s1 and s2".format(epsilon, class1, class2,  len(combined), len(s1), len(s2)))
assert(epsilon_to_size.keys() == epsilon_to_jaccard.keys())#Make sure that both dicts are the same

for eps in epsilon_to_size.keys():
    print("for eps {} average size  is {}".format(eps, np.mean(np.array(epsilon_to_size[eps]))))
for eps in epsilon_to_jaccard.keys():
    print("for eps {} average jaccard is {}".format(eps, np.mean(np.array(epsilon_to_jaccard[eps]))))    
