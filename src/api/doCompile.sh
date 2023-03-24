
ROOT=/dockerdata/ambyera
# BOOST_ROOT=/share_1399748/users/danielkxwu/envs/boost_1_66_0
BOOST_ROOT=/data/envs/boost_1_66_0/
BOOST_LIB=$BOOST_ROOT/stage/lib
BOOST_INC=$BOOST_ROOT

SEG_INC=/dockerdata/ambyera/Tentrans-2.0/solib/src/seg
RE2_INC=/dockerdata/ambyera/Re2
GLIB=/dockerdata/ambyera/glib
SOLIBS=$ROOT/Tentrans-2.0/solib/lib

SOLIBS2=$ROOT/Tentrans-2.0/nmt_only/v0_decode/TenTrans-Decoding/lib

<<:PREPROCESS
g++ -std=c++11 -Dlinux -I ./inc \
	-L$BOOST_LIB -lboost_regex -lboost_system -lboost_timer -lboost_iostreams -lboost_filesystem -lboost_chrono -lboost_program_options -lboost_thread \
	-L ${GLIB}/lib \
	-L ${SOLIBS} -lopencc -lglib-2.0 -lgomp -ltorch -lnvToolsExt -lnvrtc -lnvrtc-builtins -lmkl_gnu_thread -lmkl_core -lcurand -lcudart -lcublas -lc10 -lc10_cuda -lgcc_s -lcaffe2_nvrtc -lstdc++ \
	-L ./solib -lTentrans -lm -lpthread \
	${SOLIBS}/libscws.a ${SOLIBS}/libre2.a ${SOLIBS}/liburheen.a ${SOLIBS}/libsentencepiece.a ${SOLIBS}/libsentencepiece_train.a ${SOLIBS}/libverify.a \
	test_preprocess.cpp \
	-o ./z_preprocess
:PREPROCESS

#<<:PRE_TRANS
g++ -std=c++11 -Dlinux -I ./inc -g \
	-L$BOOST_LIB -lboost_regex -lboost_system -lboost_timer -lboost_iostreams -lboost_filesystem -lboost_chrono -lboost_program_options -lboost_thread \
	-L ${GLIB}/lib \
	-L ${SOLIBS} -lopencc -lglib-2.0 -lgomp -ltorch -lnvToolsExt -lnvrtc -lnvrtc-builtins -lmkl_gnu_thread -lmkl_core -lcurand -lcudart -lcublas -lc10 -lc10_cuda -lgcc_s -lcaffe2_nvrtc -lstdc++ \
	-L ./solib -lTentrans_api -lTenTrans -lTenTrans_cuda -lm -lpthread \
	${SOLIBS2}/libyaml-cpp.a \
	${SOLIBS}/libscws.a ${SOLIBS}/libre2.a ${SOLIBS}/liburheen.a ${SOLIBS}/libsentencepiece.a ${SOLIBS}/libsentencepiece_train.a ${SOLIBS}/libverify.a \
	test_trans.cpp \
	-o ./z_translator
#:PRE_TRANS

