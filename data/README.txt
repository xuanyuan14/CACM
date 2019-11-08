1. Due to the difference of the session size, the batch_size can only be set to one.
2. You should generate your embedding files through the node2vec tool and put it under the ./data/graph directory.
3. The format of sample session files is as follows:
|
|-- each line: [<query sequence>]<tab>[<previous interaction>]<tab>[<document info>]<tab><clicked>
|-- query sequence: qids 
|-- interaction sequence: uid, rank, vid, clicked
|-- document info: uid, rank, vid
|-- clicked: 0 or 1