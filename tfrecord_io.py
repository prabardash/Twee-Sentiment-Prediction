from tensorflow.train import SequenceExample
from tensorflow.io import TFRecordWriter, FixedLenFeature, FixedLenSequenceFeature, parse_single_sequence_example
from tensorflow.data import TFRecordDataset
import tensorflow as tf
import os

def funcOne():
    print('Inside funcOne')

    
def sequence_to_tfexample(sequence, sentiment):
    '''
    Converts each row of tokens to sequence feature in feature list,
    while the label-id is stored as a context feature
    '''
    ex = SequenceExample()
    seq_len = len(sequence)
    ex.context.feature["length"].int64_list.value.append(seq_len)
    ex.context.feature["sentiment"].int64_list.value.append(sentiment)

    fl_tokens = ex.feature_lists.feature_list["tokens"]
    for token in sequence:
              fl_tokens.feature.add().int64_list.value.append(token)

    return ex

def write_tfr_batches(data, label,batch_size, num_batches, savepath, dataset_type):
    start =0 
    next_start = 0

    for batch in range(num_batches):
        #print(batch)
        start = batch*batch_size
        filename = '{}_0{}.tfrecord'.format(dataset_type,batch)
        filepath = os.path.join(savepath,filename)
        with open(filepath,'w') as f:
            writer = TFRecordWriter(f.name)

        if(batch != num_batches-1):
            next_start = (batch+1)*batch_size
        else:
            next_start = len(data)

        for i in range(start,next_start):
            #write_tfrecord(data[star:next_start], out_path, )
            record = sequence_to_tfexample(sequence = data[i], sentiment = label[i])
            writer.write(record.SerializeToString())
            
def parse(ex):
    '''
    Explain to TF how to go froma  serialized example back to tensors
    :param ex:
    :return:
    '''
    context_features = {
        "length": FixedLenFeature([], dtype=tf.int64),
        "sentiment": FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "tokens": FixedLenSequenceFeature([], dtype=tf.int64),
    }

    # Parse the example (returns a dictionary of tensors)
    context_parsed, sequence_parsed = parse_single_sequence_example(
        serialized=ex,
        context_features=context_features,
        sequence_features=sequence_features
    )
    return sequence_parsed["tokens"], [context_parsed['sentiment']]

def dataset_from_batch(filepaths,n_readers = 4,batch_size=128):
    '''
    Makes  a Tensorflow dataset that is shuffled, batched and parsed according to the same Tokenizer that was used earlier.
    :return: a Dataset that is shuffled
    '''
    dataset = tf.data.Dataset.list_files(filepaths).repeat(None)
    
    # Read multiple tf record files in interleaved manner. This makes a dataset of raw TFRecords
    dataset = dataset.interleave(lambda filepath: TFRecordDataset([filepath]).repeat(None),
                                cycle_length = n_readers, 
                                num_parallel_calls = tf.data.experimental.AUTOTUNE)
    
    # Apply/map the parse function to every record. Now the dataset is a bunch of dictionaries of Tensors
    dataset =  dataset.map(parse,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=3000)
    #Batch the dataset so that we get batch_size examples in each batch.
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)

def make_dataset(path,batch_size=128):
    '''
    Makes  a Tensorflow dataset that is shuffled, batched and parsed according to Tokenizer.
    You can chain all the lines here, I split them into seperate calls so I could comment easily
    :param path: The path to a tf record file
    :param path: The size of our batch
    :return: a Dataset that shuffles and is padded
    '''
    # Read a tf record file. This makes a dataset of raw TFRecords
    dataset = tf.data.TFRecordDataset([path]).repeat(None)
    # Apply/map the parse function to every record. Now the dataset is a bunch of dictionaries of Tensors
    dataset =  dataset.map(parse,num_parallel_calls=5)
    #Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=3000)
    #Batch the dataset so that we get batch_size examples in each batch.
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)