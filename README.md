This implementation is based on the [cnn library](https://github.com/clab/cnn-v1) for this software to function. The paper is "Shift-Reduce Constituent Parsing with Neural Lookahead Features". 

#### Building


    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make    


#### Preprocessing

You can get the constituent hierarchy by scripts/conhier_s.py for s-type constituent hierarchy and scripts/conhier_e.py for e-type constituent hierachy

    scripts/conhier_s.py train.con > train.s
    scripts/conhier_e.py train.con > train.e

### Training

    ./s-hierarchy-trainer train.s dev.s 

### Decoding

    ./s-hierarchy-trainer train.s test.s

It will automatically generate the output file test.sOUT, which then can be used as extra features on the lookahead implementation of [ZPar](https://github.com/SUTDNLP/ZPar)

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
