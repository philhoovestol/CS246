Implementation of user-user and item-item collaborative filtering on a sample data set of watch patterns for different users to different television shows.

Contains the following files:

1. user-shows.txt This is the ratings matrix R, where each row corresponds to a user
and each column corresponds to a TV show. Rij = 1 if user i watched the show j over
a period of three months. The columns are separated by a space.
2. shows.txt This is a file containing the titles of the TV shows, in the same order as
the columns of R. 
3. The implementation file (p4.py) uses collaborative filtering methods to output the top 5 shows recommended for the 500th user of the dataset, whose first 100 entries in the ratings matrix you will notice are left blank.
