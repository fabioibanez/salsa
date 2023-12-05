# Suite of Models for Enhanced Metadata Extraction

We are developing a suite of models aimed at acquiring more detailed metadata. The key strategies for metadata extraction involve the following steps:

1. **Collecting Statistics**

   This approach can guide you towards the first significant areas to examine. The collection of data and its proper analysis may indicate specific patterns or trends, which serve as a good starting point for in-depth exploration.

2. **Temporal Clustering**

   Instead of clustering single frames, a more comprehensive approach would be to cluster batches of frames. This strategy focuses on the temporal aspect of the data and can be broken down into several subcategories:
   - a. **Action Queries Batching**

     This approach involves grouping similar action queries over a specific time frame.

   - b. **Contextual Queries**

     Examples of such queries include requests like, "Show me car crashes after the shooting". These requests ask for more events of a particular type after a specific event, illustrating the uniqueness of the temporal aspect.

   - c. **Sequences of Event Querying**

     This strategy takes into account sequences of events while maintaining the "show me more" aspect. An example would be to understand the trend before and after a specific event, such as "crashes after a shooting".

   - d. **Temporal "Show me More" Queries**

     This approach involves the execution of temporal "show me more" queries. These queries seek to identify more instances of a specific event after a certain point in time, or where a particular event happened after another.


''' 
suite of models to get more metadata 

1. Collect statistics - may lead to hey here is the first place you should be looking for stuff
2. temporal - not just a clustering of one frame, you need to cluster batches of frames
  a. action queries batching
  b. show me car crashes after the shooting; show me more of this after this, temporal aspect is unique
  c. Sequences of event querying, but there is still the show me more aspect
    i. the trend before and after - crashes after a shooting
  d. temporal aspect of show me more of this, show me more of this after x event, or where this happened after it
'''
  