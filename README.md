# tweet-portmanteau-classifier
Processes plain text tweets and a reference dictionary to identify likely unknown portmanteaus and their possible source words.

The program takes information from the data folder
- dictionary - a plaintext list of dictionary words
- optional_answers - a plaintext list of unknown portmanteaus, and their source words - one set per line eg Brexit Britain exit. This is used to measure the performance of the algorithm.
- tweets - a plaintext list of tweets to analyse

And outputs to the results folder
- candidates_w_sources - a CSV list (and associated pickle data) of the likely blend words, associated candidates for the source word, and the similarity value of the source word to the blend.
	eg	Brexit prefix 0.2 Britain
			Brexit suffix 0.8 exit
- candidates_w_sources_best_only - the same as above but including only the highest scoring source word
- rank_performance_against_true_answers - requires "optional answers" - a CSV list (and associated pickle data) of words that were successfully identified as possible blends (list of true positives), and the average Reciprocal Mean Rank of their source words. For example "Brexit, 0.25" means that the program correctly identified Brexit as a possible blend word, and the words Britain and exit were on average, in the 4th position in the program's list of likely best candidates for source word. A score of 1 means that the true source words were at the top of the list. A score of 0 meanns the true source words were not in the list.

For a full discussion of program methodology, please consult the associated paper, "An exclusion-based method for the detection of lexical blends"