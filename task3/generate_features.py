""" All feature matrix creation functions are stored in this file. """

import pandas as pd


def generate_features_onehot(df, same_aminos, path_to_save):
    """
    Creates a feature matrix from the provided dataframe df and stores it as 
    .csv file in ./data/features/
    ----------
    df: Pandas data frame of the read in training data. 
    same_aminos: Boolean, specifies whether pairs, triplets and quadtriplets should be used as features. 
    path_to_save: String, path where the created .csv file is stored. 
    """
    # split sequence column into 4 separate columns for each letter
    df_split = df['Sequence'].str.split('', expand=True, n=4).loc[:,1:]
    df_split.columns = [i for i in range(4)]    
    
    # One hot encoded columns (4x20 = 80+2x20**2 if pairwise)
    df_dummies = pd.get_dummies(df_split)

    df = pd.concat([df_dummies, df], axis=1)
    
    # encode pairs (same amino acids as direct neigbors)
    # encode triplets (three amino acids at consecutive positions)
    # encode quadruplets (same amino acid at all positions)
    # encode double pairs
    if same_aminos:
        df['pairs'] = df.Sequence.apply(lambda x: 1 if (1 == sum([a==b for a, b in zip(x[:-1], x[1:])])) else 0)
        df['triplets'] = df.Sequence.apply(lambda x: 1 if (1 == sum([a==b==c for a, b, c in zip(x[:-1], x[1:], x[2:])])) else 0)
        df['quads'] = df.Sequence.apply(lambda x: 1 if (3 == sum([a==b for a, b in zip(x[:-1], x[1:])])) else 0)
        df['two_pairs'] = df.Sequence.apply(lambda x: 1 if \
                                                (2 == sum([a==b for a, b in zip(x[:-1], x[1:])]) and \
                                                0 == sum([a==b==c for a, b, c in zip(x[:-1], x[1:], x[2:])]))\
                                                else 0)


    # if structure:
    #    df_structure = pd.read_cs(structure_path, sep=';')

    df.to_csv(path_to_save, index=False)
    print(f'Generated feature matrix .csf file at: {path_to_save}')


####################################################################################
# Notes about results. 

# clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced')
# pairwise=True gives worse results for RF
# pairwise 1&3 0.786
# pairwise 1,2&3 0.74
# no pairwise 0.8
# pairwise only 0.559, no identity
# identity (onehot encoding w/o pairs) only is best 0.82!!
# identity plus same aminos 0.825, helps a little bit



# manual features give bad results for RF 0.62
# best features for RF: one-hot w/o pairwise indicators
# 0.812 for pairwise True and xgboost
# 0.816 for pairwise False and xgboost
# adaboost and RUS perform really bad (0.37 validation score)

####################################################################################


# incorporate 3D structure
# https://www.genome.jp/dbget-bin/www_bget?aaindex:TANS760101
# https://www.genome.jp/dbget-bin/www_bget?pmid:1004017
# Medium- and long-range interaction parameters between amino acids for predicting 
# three-dimensional structures of proteins.
#   -2.6
#   -3.4 -4.3
#   -3.1 -4.1 -3.2
#   -2.8 -3.9 -3.1 -2.7
#   -4.2 -5.3 -4.9 -4.2 -7.1
#   -3.5 -4.5 -3.8 -3.2 -5.0 -3.4
#   -3.0 -4.2 -3.4 -3.3 -4.4 -3.6 -2.8
#   -3.8 -4.5 -4.0 -3.7 -5.4 -4.4 -3.8 -3.9
#   -4.0 -4.9 -4.4 -4.3 -5.6 -4.7 -4.5 -4.7 -4.9
#   -5.9 -6.2 -5.8 -5.4 -7.3 -5.9 -5.7 -6.3 -6.6 -8.2
#   -4.8 -5.1 -4.6 -4.3 -6.2 -5.0 -4.6 -5.2 -5.6 -7.5 -6.0
#   -3.1 -3.6 -3.3 -3.2 -4.4 -3.7 -3.8 -3.8 -4.1 -5.6 -4.6 -2.7
#   -4.6 -5.0 -4.2 -4.3 -6.2 -3.5 -4.6 -5.1 -5.4 -7.4 -6.3 -4.7 -5.8
#   -5.1 -5.8 -5.0 -4.9 -6.8 -5.3 -5.0 -5.6 -6.4 -8.0 -7.0 -4.9 -6.6 -7.1
#   -3.4 -4.2 -3.6 -3.3 -5.3 -4.0 -3.5 -4.2 -4.5 -6.0 -4.8 -3.6 -5.1 -5.2 -3.5
#   -2.9 -3.8 -3.1 -2.7 -4.6 -3.6 -3.2 -3.8 -4.3 -5.5 -4.4 -3.0 -4.1 -4.7 -3.4 -2.5
#   -3.3 -4.0 -3.5 -3.1 -4.8 -3.7 -3.3 -4.1 -4.5 -5.9 -4.8 -3.3 -4.6 -5.1 -3.6 -3.3 -3.1
#   -5.2 -5.8 -5.3 -5.1 -6.9 -5.8 -5.2 -5.8 -6.5 -7.8 -6.8 -5.0 -6.9 -7.4 -5.6 -5.0 -5.1 -6.8
#   -4.7 -5.6 -5.0 -4.7 -6.6 -5.2 -4.9 -5.4 -6.1 -7.4 -6.2 -4.9 -6.1 -6.6 -5.2 -4.7 -4.9 -6.8 -6.0
#   -4.3 -4.9 -4.3 -4.0 -6.0 -4.7 -4.2 -5.1 -5.3 -7.3 -6.2 -4.2 -6.0 -6.5 -4.7 -4.2 -4.4 -6.5 -5.9 -5.5
