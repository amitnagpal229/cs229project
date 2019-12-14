/Users/anagpal/libsvm-3.24/svm-train -s 0 -t 1 -e 0.0000001 -w1 21 -w-1 1 jump_svm_train.txt jump.svm
/Users/anagpal/libsvm-3.24/svm-predict jump_svm_test.txt jump.svm oo_test.jump
/Users/anagpal/libsvm-3.24/svm-predict jump_svm_train.txt jump.svm oo_train.jump

/Users/anagpal/libsvm-3.24/svm-train -s 1 -t 1 -e 0.0000001 -n 0.155 mj_svm_train.txt mj.svm
/Users/anagpal/libsvm-3.24/svm-predict mj_svm_test.txt mj.svm oo_test.mj
/Users/anagpal/libsvm-3.24/svm-predict mj_svm_train.txt mj.svm oo_train.mj

/Users/anagpal/libsvm-3.24/svm-train -s 1 -t 1 -e 0.0000001 move_svm_train.txt mm.svm
/Users/anagpal/libsvm-3.24/svm-predict move_svm_test.txt mm.svm oo_test.move
/Users/anagpal/libsvm-3.24/svm-predict move_svm_tran.txt mm.svm oo_train.move

