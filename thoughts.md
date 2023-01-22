## Resources:
    https://github.com/microsoft/recommenders
    https://github.com/Trinitui/RecEngineTesting
    https://help.figma.com/hc/en-us/articles/6376798776343-AWS-Amplify-Studio-and-Figma

[--- We're using Collaborative/Content Based Filtering ---]

## Flow:

    1) Determine Features/Dims/Predictors to use (Talk to Kyle...)

    2) Turn each property into a vector

    3) Maybe remove properties with many many transactions

    4) Do PCA on data, TSNE to visualize

    5) Test for similarities using cosine similarities or Jaccard - "These are you n-closest properties (Sanity Check / Explanability)"

    6) Maybe this becomes our baseline, and we can test for better results with other models

    7) Return a list of most-similar properties


## User Story:
"I bought 2 2-bed [You have a similarity score of: n]
[We search dataset for similar n's] -> Here are your top 5 properties to look at!"



## Next:


    0) Andrew will get data from Sagemaker (share how to do this)
    1) Work in Sagemeaker to get steps 1 - 4 done. 
    2) Mike to start working on math
