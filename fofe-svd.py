import numpy
import logging
import argparse
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.basicConfig( 
        format='%(asctime)s : %(levelname)s : %(message)s', 
        level=logging.INFO 
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('row', type=str, help='ppmi coo row indices' )
    parser.add_argument('col', type=str, help='ppmi coo column indices' )
    parser.add_argument('val', type=str, help='ppmi coo values')
    parser.add_argument('output', type=str, help='output embedding')
    parser.add_argument('--embed_dim', type=int, default=256, help='dimension of the desired word embedding')
    parser.add_argument('--n_iter', type=int, default=32, help='#iterations of TruncatedSVD')
    args = parser.parse_args()
    logger.info(args)

    row = numpy.fromfile(args.row, dtype=numpy.int32)
    col = numpy.fromfile(args.col, dtype=numpy.int32)
    val = numpy.fromfile(args.val, dtype=numpy.float32)

    fofe = coo_matrix((val, (row, col))).astype(numpy.float32)
    fofe = fofe.tocsr()
    logger.info('fofe matrix after ppmi has %d non-zero elements' % fofe.nnz)

    svd = TruncatedSVD(n_components=args.embed_dim, n_iter=args.n_iter)
    embed = svd.fit_transform(fofe)
    numpy.save(args.output, embed)
