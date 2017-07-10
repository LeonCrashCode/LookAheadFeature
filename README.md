This implementation is based on the [cnn library](https://github.com/clab/cnn-v1) for this software to function. The paper is "Shift-Reduce Constituent Parsing with Neural Lookahead Features". 

#### Building


    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make    


#### Preprocessing

You can get the constituent hierarchy by scripts/conhier_s.py for s-type constituent hierarchy and scripts/conhier_e.py for e-type constituent hierachy

    ./scripts/conhier_s.py [training data in bracketed format] > [s-type training data]
    ./scripts/conhier_e.py [training data in bracketed format] > [e-type training data]

    ./scripts/conhier_s.py [development data in bracketed format] > [s-type development data]
    ./scripts/conhier_e.py [development data in bracketed format] > [e-type development data]

    ./scripts/conhier_s.py [test data in bracketed format] > [s-type test data]
    ./scripts/conhier_e.py [test data in bracketed format] > [e-type test data]

The directory data contains the related data.

### Training

    ./build/impl/s-hierarchy-trainer train.s dev.s 
    ./build/impl/e-hierarchy-trainer train.e dev.e

### Decoding

    ./build/impl/s-hierarchy-trainer train.s test.s [s_model]
    ./build/impl/e-hierarchy-decoder train.e test.e [s_model]

It will automatically generate the output file test.sOUT and test.eOUT, respectively, which then can be used as extra features on the lookahead implementation of [ZPar](https://github.com/SUTDNLP/ZPar)

### Usage of ZPar
    
    ./scripts/combine.py [.sOUT] [eOUT] > [extra feature]
 
You can follow the ZPar instruction to complie constituent parser with implementation of "jiangming".
    
    ./conparser [in] [out] zpar_model [extra feature]

### Citation

    @article{TACL927,
	    author = {Liu, Jiangming  and Zhang, Yue },
	    title = {Shift-Reduce Constituent Parsing with Neural Lookahead Features},
	    journal = {Transactions of the Association for Computational Linguistics},
	    volume = {5},
	    year = {2017},
	    keywords = {},
        issn = {2307-387X},
        url = {https://transacl.org/ojs/index.php/tacl/article/view/927},
        pages = {45--58}
    }
