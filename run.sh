#!/bin/bash

export KNRM="\x1B[0m"
export KRED="\x1B[31m"
export KGRN="\x1B[32m"
export KYEL="\x1B[33m"
export KBLU="\x1B[34m"
export KMAG="\x1B[35m"
export KCYN="\x1B[36m"
export KWHT="\x1B[37m"


function INFO() {
    msg="$@"
    printf "${KGRN}"
    printf "`date +"%Y-%m-%d %H-%M-%S"` [INFO]: ${msg}"
    printf "\n${KNRM}"
}
export -f INFO


function CRITICAL() {
    msg="$@"
    printf "${KRED}"
    printf "`date +"%Y-%m-%d %H-%M-%S"` [CRITICAL]: ${msg}"
    printf "\n${KNRM}"
}
export -f CRITICAL


function USAGE() {
	printf "${KRED}"
	printf "Usage: $(basename $0) [option] <text-data>\n"
	printf "Options: -f               force run\n"
	printf "         -b               bidirectional fofe\n"
	printf "         -n <num-of-word> number of most frequent words to keep\n"
	printf "         -c <min-cnt>     words whose frequency is less will be pruned\n"
	printf "         -o <output-dir>  output directory\n"
	printf "         -a <alpah>  	  forgetting factor\n"
	printf "         -e <embed-dim>   desired embeding dimension\n"
	printf "${KNRM}"
	exit 1
}


function txt2bin() {
	if [ ! -e ${OUT_DIR}/row-idx ] || [ ! -z ${FORCE_RUN}]; then
		perl -pe '$_=pack"l",$_' \
			<(for f in $(ls ${PPMI}); do sed s"/[\(\),]//g" ${PPMI}/${f}; done | cut -d' ' -f1 ) \
			> ${OUT_DIR}/row-idx
		[ ! -e ${OUT_DIR}/row-idx ] && CRITICAL "fail to pack row indices" && exit 1
	fi
	INFO "binary coo row indices packed"

	if [ ! -e ${OUT_DIR}/col-idx ] || [ ! -z ${FORCE_RUN} ]; then
		perl -pe '$_=pack"l",$_' \
			<(for f in $(ls ${PPMI}); do sed s"/[\(\),]//g" ${PPMI}/${f}; done | cut -d' ' -f2 ) \
			> ${OUT_DIR}/col-idx
		[ ! -e ${OUT_DIR}/col-idx ] && CRITICAL "fail to pack column indices" && exit 1
	fi
	INFO "binary coo column indices packed"

	if [ ! -e ${OUT_DIR}/value ] || [ -z ${FORCE_RUN} ]; then
		perl -pe '$_=pack"f",$_' \
			<(for f in $(ls ${PPMI}); do sed s"/[\(\),]//g" ${PPMI}/${f}; done | cut -d' ' -f3 ) \
			> ${OUT_DIR}/value
		[ ! -e ${OUT_DIR}/value ] && CRITICAL "fail to pack values" && exit 1
	fi 
	INFO "binary coo values packed"
}


export THIS_DIR=$(cd $(dirname $0); pwd)
export FORCE_RUN=""
export DATA=""
export MIN_CNT=1
export NUM_WORD=100000
export OUT_DIR=${THIS_DIR}
export EMBED=128
export ALPHA=0.7
export BI_OPT=""


while getopts "fbn:c:o:a:e:" opt; do
    case "${opt}" in
    	f)
			FORCE_RUN=true
			;;
		b)
			BI_OPT="--bidirectoinal"
			;;
		n)
			NUM_WORD="${OPTARG}"
			;;
		c)
			MIN_CNT="${OPTARG}"
			;;
		o)
			OUT_DIR="${OPTARG}"
			;;
		a)
			ALPHA="${OPTARG}"
			;;
		e)
			EMBED="${OPTARG}"
			;;
        *)
            USAGE
            ;;
    esac
done
shift $((OPTIND - 1))


if [ $# -ne 1 ] || [ ! -f $1 ]; then
    USAGE
fi


export DATA=$1
if [ ! -f ${DATA} ] || [ ! -r ${DATA} ]; then
	CRITICAL "unable to read ${DATA}"
	exit 1
fi


[ ! -d ${OUT_DIR}/logs ] && mkdir -p ${OUT_DIR}/logs


export VOCAB=${OUT_DIR}/$(basename ${DATA}).vocab
if [ ! -f ${VOCAB} ] || [ ! -z ${FORCE_RUN} ]; then
	rm -rf ${VOCAB} &> /dev/null
	spark-submit ${THIS_DIR}/word-count.py \
		--minCnt ${MIN_CNT} \
		--topK ${NUM_WORD} \
		${DATA} \
		${VOCAB} |& tee -a ${OUT_DIR}/logs/vocab.log
fi
[ ! -f ${VOCAB} ] && CRITICAL "fail to generate vocab" && exit 1
INFO "vocab generated"


export PPMI=${OUT_DIR}/$(basename ${DATA}).ppmi
if [ ! -d ${PPMI} ] || [ ! -z ${FORCE_RUN} ]; then
	rm -rf ${PPMI} &> /dev/null
	spark-submit ${THIS_DIR}/fofe-ppmi.py \
		--alpha ${ALPHA} \
		${BI_OPT} \
		${DATA} \
		${VOCAB} \
		${PPMI} |& tee -a ${OUT_DIR}/logs/ppmi.log
fi
if [ ! -d ${PPMI} ] || [ ! -f ${PPMI}/_SUCCESS ]; then
	CRITICAL "fail to compute ppmi"
	exit 1
fi
INFO "ppmi computed"


txt2bin |& tee -a ${OUT_DIR}/logs/txt2bin.log


export NPY=${OUT_DIR}/$(basename ${DATA}).npy
if [ ! -e ${NPY} ] || [ ! -z ${FORCE_RUN} ]; then
	python ${THIS_DIR}/fofe-svd.py \
		--embed_dim ${EMBED} \
		${OUT_DIR}/row-idx \
		${OUT_DIR}/col-idx \
		${OUT_DIR}/value \
		${NPY} |& tee -a ${OUT_DIR}/logs/svd.log
fi
[ ! -e ${NPY} ] && CRITICAL "fail to run SVD" && exit 1
INFO "ppmi decomposed into embedding"
