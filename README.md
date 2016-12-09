This implementation is based on the [cnn library](https://github.com/clab/cnn-v1) for this software to function.

#### Building

First you need to download the cnn library and related libraries, and make them compiled, and then download the directory into the cnn 
    
    copy -r constituent_hierarchy cnn/

Add the directory into CMakeList.txt

    cd cnn
    vi CMakeList.txt
	add_subdirectory(constituent_hierarchy)

Follow the cnn `README` to compile them

#### Preprocessing

You can get the constituent hierarchy by scripts/conhier_s.py for s-type constituent hierarchy and scripts/conhier_e.py for e-type constituent hierachy

    scripts/conhier_s.py train.con > train.s
    scripts/conhier_e.py train.con > train.e

### Training

    ./s-hierarchy-trainer train.s dev.s 

### Decoding

    ./s-hierarchy-trainer train.s test.s

It will automatically generate the output file test.sOUT, which then can be used as extra features on the lookahead implementation of [ZPar](https://github.com/SUTDNLP/ZPar)
