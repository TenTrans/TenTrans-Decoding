BOOST_ROOT=/data/envs/boost_1_66_0
BOOST_LIB=$BOOST_ROOT/stage/lib
SOLIBS=/usr/local/cuda-9.1/lib64

/usr/bin/gcc -std=c++11 -I . \
    -L$BOOST_LIB -lboost_regex -lboost_system -lboost_timer -lboost_iostreams -lboost_filesystem -lboost_chrono -lboost_program_options -lboost_thread \
    -L ${SOLIBS} -lgomp -lnvToolsExt -lnvrtc -lnvrtc-builtins -lcurand -lcudart -lcublas -lgcc_s -lstdc++ \
    -L ./solib -lTenTrans -lTenTrans_cuda -lm -lpthread \
    ./solib/libyaml-cpp.a \
    main.cpp \
    -o ./z_translator
