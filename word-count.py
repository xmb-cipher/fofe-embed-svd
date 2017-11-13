from pyspark import SparkContext, SparkConf
import codecs, argparse, logging
logger = logging.getLogger( __name__ )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( 'data', type = str, 
                          help = 'training data in a single file, one sentence per line' )
    parser.add_argument( 'output', type = str, 
                          help = 'word counts in decending order' )
    parser.add_argument( '--minCnt', type = int, default = 3,
                          help = 'words whose frequency less than minCnt are pruned' )
    parser.add_argument( '--topK', type = int, default = 100000,
                          help = 'how many words to keep' )
    args = parser.parse_args()

    logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
                        level = logging.INFO )
    logger.info( args )

    conf = SparkConf().setAppName( 'word-count' )
    sc = SparkContext( conf = conf )

    tokenized = sc.textFile( args.data ).flatMap(lambda line: line.split() + [u'</s>'])
    wordCount = tokenized.map(lambda w: (w, 1)).reduceByKey(lambda v1, v2: v1 + v2).collect()
    logger.info( 'Counting done' )

    wordCount.sort( key = lambda x: (x[1], x[0]), reverse = True )
    logger.info( 'Sorting done' )

    with codecs.open( args.output, 'wb', 'utf8' ) as fp:
        outCnt, unkCnt, eosCnt = 0, 0, 1
        for w, c in wordCount:
            if w == u'</s>':
                eosCnt = c
            elif c < args.minCnt or \
                    outCnt + 3 == args.topK or \
                    w == u'<unk>':
                unkCnt += c
            else:
                print >> fp, w, c
                outCnt += 1
        print >> fp, '<s>', 1
        print >> fp, '</s>', eosCnt
        print >> fp, '<unk>', max(1, unkCnt)
    logger.info( 'Writing done' )
