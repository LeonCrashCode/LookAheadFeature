#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/fast-lstm.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
# include "cnn/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace cnn;

float pdrop = 0.5;
unsigned LAYERS = 1;

//word
unsigned WORD_DIM = 50;

unsigned CHAR_DIM = 30;
unsigned CHAR_HIDDEN_DIM = 60;
unsigned CHAR_WINDOWS = 2;

unsigned INPUT_DIM = CHAR_HIDDEN_DIM + WORD_DIM;
unsigned INPUT_WINDOWS = 2;
unsigned INPUT_WINDOWS_DIM = INPUT_DIM * (INPUT_WINDOWS*2+1);

unsigned HIDDEN_DIM = 100;

unsigned LABEL_HIDDEN_DIM = 50; 

unsigned CHAR_SIZE = 0;
unsigned LABEL_SIZE = 0;
unsigned VOCAB_SIZE = 0;

bool eval = false;
cnn::Dict wd;
cnn::Dict cd;
cnn::Dict ld;
int kUNK;
int endlabel;

template <class Builder>
struct RNNLanguageModel {
  LookupParameters* p_w;
  LookupParameters* p_c;

  Parameters* p_wordpadding_2;
  Parameters* p_wordpadding_1;
  Parameters* p_wordpadding1;
  Parameters* p_wordpadding2;

  Parameters* p_charpadding_2;
  Parameters* p_charpadding_1;
  Parameters* p_charpadding1;
  Parameters* p_charpadding2;

  Parameters* p_charw2charh;
  Parameters* p_charhbias;
  //attention pooling
  Parameters* p_charh2xMExp;
  Parameters* p_word2xMExp;
  Parameters* p_xMExpbias;
  Parameters* p_xMExp2xExp;
  //

  // Parameters* p_l2oh;
  // Parameters* p_r2oh;
  // Parameters* p_ohbias;

  Parameters* p_oh2ohxMExp; 
  Parameters* p_osh2ohxMExp;
  Parameters* p_u;

  Parameters* p_oh2lh;
  Parameters* p_lhbias;

  Parameters* p_lh2l;

  Builder l2rbuilder;
  Builder r2lbuilder;
  Builder orderbuilder;
  explicit RNNLanguageModel(Model& model) :
      l2rbuilder(LAYERS+1, INPUT_WINDOWS_DIM, HIDDEN_DIM, &model),
      r2lbuilder(LAYERS+1, INPUT_WINDOWS_DIM, HIDDEN_DIM, &model),
      orderbuilder(LAYERS, HIDDEN_DIM*4, HIDDEN_DIM, &model){

    p_w = model.add_lookup_parameters(VOCAB_SIZE, {WORD_DIM}); 
    p_c = model.add_lookup_parameters(CHAR_SIZE, {CHAR_DIM});
    
    p_wordpadding_2 = model.add_parameters({INPUT_DIM});
    p_wordpadding_1 = model.add_parameters({INPUT_DIM});
    p_wordpadding2 = model.add_parameters({INPUT_DIM});
    p_wordpadding1 = model.add_parameters({INPUT_DIM});

    p_charpadding_2 = model.add_parameters({CHAR_DIM});
    p_charpadding_1 = model.add_parameters({CHAR_DIM});
    p_charpadding2 = model.add_parameters({CHAR_DIM});
    p_charpadding1 = model.add_parameters({CHAR_DIM});

    p_charw2charh = model.add_parameters({CHAR_HIDDEN_DIM, CHAR_DIM*(CHAR_WINDOWS*2+1)});
    p_charhbias = model.add_parameters({CHAR_HIDDEN_DIM});

    //attention pooling
    p_charh2xMExp = model.add_parameters({CHAR_HIDDEN_DIM, CHAR_HIDDEN_DIM});
    p_word2xMExp = model.add_parameters({CHAR_HIDDEN_DIM, WORD_DIM});
    p_xMExpbias = model.add_parameters({CHAR_HIDDEN_DIM});
    p_xMExp2xExp = model.add_parameters({CHAR_HIDDEN_DIM, CHAR_HIDDEN_DIM});
    //

    // p_l2oh = model.add_parameters({HIDDEN_DIM, HIDDEN_DIM});
    // p_r2oh = model.add_parameters({HIDDEN_DIM, HIDDEN_DIM});
    // p_ohbias = model.add_parameters({HIDDEN_DIM});

    //attention labelling
    p_oh2ohxMExp = model.add_parameters({HIDDEN_DIM, HIDDEN_DIM*2});
    p_osh2ohxMExp = model.add_parameters({HIDDEN_DIM, HIDDEN_DIM});
    p_u = model.add_parameters({1,HIDDEN_DIM});

    p_oh2lh = model.add_parameters({LABEL_HIDDEN_DIM,HIDDEN_DIM});
    p_lhbias = model.add_parameters({LABEL_HIDDEN_DIM});

    p_lh2l = model.add_parameters({LABEL_SIZE, LABEL_HIDDEN_DIM});
  }

  // return Expression of total loss
  Expression BuildTaggingGraph(const vector<int>& sent, const vector< vector<int> >& chars, const vector< vector<int> >& labels, ComputationGraph& cg, double* cor = 0, unsigned* predict = 0, unsigned* overall = 0, bool train=true) {
    const unsigned slen = sent.size();
    l2rbuilder.new_graph(cg);  // reset RNN builder for new graph
    l2rbuilder.start_new_sequence();
    r2lbuilder.new_graph(cg);  // reset RNN builder for new graph
    r2lbuilder.start_new_sequence();
    
    Expression i_wordpadding_2 = parameter(cg, p_wordpadding_2);
    Expression i_wordpadding_1 = parameter(cg, p_wordpadding_1);
    Expression i_wordpadding2 = parameter(cg, p_wordpadding2);
    Expression i_wordpadding1 = parameter(cg, p_wordpadding1);

    Expression i_charpadding_2 = parameter(cg, p_charpadding_2);
    Expression i_charpadding_1 = parameter(cg, p_charpadding_1);
    Expression i_charpadding2 = parameter(cg, p_charpadding2);
    Expression i_charpadding1 = parameter(cg, p_charpadding1);
  
    Expression i_charw2charh = parameter(cg, p_charw2charh);
    Expression i_charhbias = parameter(cg, p_charhbias);
    //attention pooling
    Expression i_charh2xMExp = parameter(cg, p_charh2xMExp);
    Expression i_word2xMExp = parameter(cg, p_word2xMExp);
    Expression i_xMExpbias = parameter(cg, p_xMExpbias);
    Expression i_xMExp2xExp = parameter(cg, p_xMExp2xExp);
    //
    
    // Expression i_l2oh = parameter(cg, p_l2oh);
    // Expression i_r2oh = parameter(cg, p_r2oh);
    // Expression i_ohbias = parameter(cg, p_ohbias);

    //attention labeling
    Expression i_oh2ohxMExp = parameter(cg, p_oh2ohxMExp);
    Expression oh2ohxMExp = parameter(cg, p_osh2ohxMExp);
    Expression i_u = parameter(cg, p_u);

    Expression i_oh2lh = parameter(cg, p_oh2lh);
    Expression i_lhbias = parameter(cg, p_lhbias);

    Expression i_lh2l = parameter(cg, p_lh2l);

    vector<Expression> i_word(slen);

    vector< vector<Expression> > i_char(slen);
    vector< vector<Expression> > i_charw(slen);
    vector< vector<Expression> > i_charh(slen);
    //attention pooling
    vector< vector<Expression> > i_charxMExp(slen);
    vector< vector<Expression> > i_charxExp(slen);
    vector< Expression > i_charSum(slen);
    vector< vector<Expression> > i_charPool(slen);
    vector< Expression > i_charinput(slen);
    //

    vector< Expression > i_input(slen);
    vector< Expression > i_inputw(slen);

    vector<Expression> fwds(slen);
    vector<Expression> revs(slen);
    vector<Expression> i_oh(slen);

    vector<Expression> errs; 
    for (unsigned i = 0; i < slen; ++i) {
      i_word[i] = lookup(cg, p_w, sent[i]);
      if (!eval) { i_word[i] = dropout(i_word[i], pdrop); }
      i_char[i].resize(chars[i].size());
      for (unsigned j = 0; j < chars[i].size(); ++j) {
        i_char[i][j] = lookup(cg, p_c, chars[i][j]);
        if (!eval) { i_char[i][j] = dropout(i_char[i][j], pdrop); }
      }
//      cerr<<"lookup done"<<endl;
      i_charw[i].resize(i_char[i].size());
      i_charh[i].resize(i_char[i].size());
      i_charxMExp[i].resize(i_char[i].size());
      i_charxExp[i].resize(i_char[i].size());
      for (int j = 0; j < i_char[i].size(); ++j) {
        if ( j-1 < 0){
	  if ( j+1 >= i_char[i].size())
	    i_charw[i][j] = concatenate({i_charpadding_2, i_charpadding_1, i_char[i][j], i_charpadding1, i_charpadding2});
	  else if ( j+2 >= i_char[i].size())
	    i_charw[i][j] = concatenate({i_charpadding_2, i_charpadding_1, i_char[i][j], i_char[i][j+1], i_charpadding1});
	  else
	    i_charw[i][j] = concatenate({i_charpadding_2, i_charpadding_1, i_char[i][j], i_char[i][j+1], i_char[i][j+2]});
  	}
	else if ( j-2 < 0){
	  if ( j+1 >= i_char[i].size())
	    i_charw[i][j] = concatenate({i_charpadding_1, i_char[i][j-1], i_char[i][j], i_charpadding1, i_charpadding2});
    	  else if ( j+2 >= i_char[i].size())
	    i_charw[i][j] = concatenate({i_charpadding_1, i_char[i][j-1], i_char[i][j], i_char[i][j+1], i_charpadding1});
	  else
	    i_charw[i][j] = concatenate({i_charpadding_1, i_char[i][j-1], i_char[i][j], i_char[i][j+1], i_char[i][j+2]});
        }
  	else{
	  if ( j+1 >= i_char[i].size())
            i_charw[i][j] = concatenate({i_char[i][j-2], i_char[i][j-1], i_char[i][j], i_charpadding1, i_charpadding2});
          else if ( j+2 >= i_char[i].size())
            i_charw[i][j] = concatenate({i_char[i][j-2], i_char[i][j-1], i_char[i][j], i_char[i][j+1], i_charpadding1});
          else 
            i_charw[i][j] = concatenate({i_char[i][j-2], i_char[i][j-1], i_char[i][j], i_char[i][j+1], i_char[i][j+2]});
	}
	i_charh[i][j] = tanh(affine_transform({i_charhbias, i_charw2charh ,i_charw[i][j]}));
	i_charxMExp[i][j] = tanh(affine_transform({i_xMExpbias, i_charh2xMExp, i_charh[i][j], i_word2xMExp, i_word[i]}));
	i_charxExp[i][j] = exp(i_xMExp2xExp * i_charxMExp[i][j]);
      }
      i_charSum[i] = sum(i_charxExp[i]);
      i_charPool[i].resize(i_char[i].size());
      for (unsigned j = 0; j < i_char[i].size(); ++j){
        i_charPool[i][j] = cwise_multiply(i_charh[i][j], cdiv(i_charxExp[i][j], i_charSum[i]));
      }
      i_charinput[i] = sum(i_charPool[i]);
    }
//    cerr<<"input done"<<endl;
    for (unsigned i = 0; i < slen; ++i) {
      i_input[i] = concatenate({i_charinput[i], i_word[i]});
    }
//    cerr<<"concatenate done"<<endl;
    for (int i = 0; i < slen; ++i) {
      if(i-1 < 0){
        if(i+1>= slen){
	  i_inputw[i] = concatenate({i_wordpadding_2, i_wordpadding_1, i_input[i], i_wordpadding1, i_wordpadding2});
	}
   	else if(i+2>=slen){
	  i_inputw[i] = concatenate({i_wordpadding_2, i_wordpadding_1, i_input[i], i_input[i+1], i_wordpadding1});
	}
 	else{
 	  i_inputw[i] = concatenate({i_wordpadding_2, i_wordpadding_1, i_input[i], i_input[i+1], i_input[i+2]});
	}
      }
      else if(i-2<0){
        if(i+1>= slen){
          i_inputw[i] = concatenate({i_wordpadding_1, i_input[i-1], i_input[i], i_wordpadding1, i_wordpadding2});
        }
        else if(i+2>=slen){
          i_inputw[i] = concatenate({i_wordpadding_1, i_input[i-1], i_input[i], i_input[i+1], i_wordpadding1});
        }
        else{
          i_inputw[i] = concatenate({i_wordpadding_1, i_input[i-1], i_input[i], i_input[i+1], i_input[i+2]});
        }
      }
      else{
        if(i+1>= slen){
          i_inputw[i] = concatenate({i_input[i-2], i_input[i-1], i_input[i], i_wordpadding1, i_wordpadding2});
        }
        else if(i+2>=slen){
          i_inputw[i] = concatenate({i_input[i-2], i_input[i-1], i_input[i], i_input[i+1], i_wordpadding1});
        }
        else{
          i_inputw[i] = concatenate({i_input[i-2], i_input[i-1], i_input[i], i_input[i+1], i_input[i+2]});
        }
      }
    }
//    cerr<<"intput windows done "<<INPUT_WINDOWS_DIM<<endl;
    for (unsigned i = 0; i < slen; ++i)
      fwds[i] = l2rbuilder.add_input(i_inputw[i]);
    for (unsigned i = 0; i < slen; ++i)
      revs[slen - i - 1] = r2lbuilder.add_input(i_inputw[slen - i - 1]);   
//    cerr<<"lstm done"<<endl;
    for (unsigned i = 0; i < slen; ++i)
      //i_oh[i] = tanh(affine_transform({i_ohbias, i_l2oh, fwds[i], i_r2oh, revs[i]}));
        i_oh[i] = concatenate({fwds[i], revs[i]});

    for (unsigned i = 0; i < slen; ++i) {
   //   cerr<<i<<endl;
      orderbuilder.new_graph(cg);
      orderbuilder.start_new_sequence();
      if(train) {
	int j = 0;

	Expression oh = zeroes(cg, {HIDDEN_DIM});

	while(j < labels[i].size()){
	  //attention
	  Expression i_att_oh;
      	  vector<Expression> alpha_i(slen);
	  for (unsigned att = 0; att < slen; ++att){
	    alpha_i[att] = i_u * tanh(i_oh2ohxMExp * i_oh[att] + oh2ohxMExp * oh);
   	  }

  	  Expression input_cols = concatenate_cols(i_oh);
	  Expression alpha = softmax(concatenate(alpha_i));
    	  i_att_oh = transpose( transpose(alpha) * transpose( input_cols ) );
	  
	  Expression combine_h = concatenate({i_oh[i], i_att_oh});
	  oh = orderbuilder.add_input(combine_h);

          Expression i_lh = tanh(affine_transform({i_lhbias, i_oh2lh, oh}));
          Expression i_l = i_lh2l * i_lh;
	  
	  if (cor) {
	    vector<float> dist = as_vector(cg.incremental_forward());
	    double best = -9e99;
	    int bestk = -1;
            for (int k = 0; k < dist.size(); ++k) {
              if(dist[k] > best) {best = dist[k]; bestk = k; }
            }
            if (labels[i][j] == bestk) (*cor)++;
	    errs.push_back(pickneglogsoftmax(i_l, labels[i][j]));
            j += 1;
	  } 
	}
        if(predict) (*predict) += j;
        if(overall) (*overall) += labels[i].size();
      }
      else{
	int j = 0;
	
	Expression oh = zeroes(cg, {HIDDEN_DIM});

        while(j < 7) {
	  //attention
	  Expression i_att_oh;
          vector<Expression> alpha_i(slen);
          for (unsigned att = 0; att < slen; ++att){
            alpha_i[att] = i_u * tanh(i_oh2ohxMExp * i_oh[att] + oh2ohxMExp * oh);
          }

          Expression input_cols = concatenate_cols(i_oh);
          Expression alpha = softmax(concatenate(alpha_i));
          i_att_oh = transpose( transpose(alpha) * transpose( input_cols ) );

          Expression combine_h = concatenate({i_oh[i], i_att_oh});
          oh = orderbuilder.add_input(combine_h);

          Expression i_lh = tanh(affine_transform({i_lhbias, i_oh2lh, oh}));
          Expression i_l = i_lh2l * i_lh;
	  if (cor) {
	    vector<float> dist = as_vector(cg.incremental_forward());
	    double best = -9e99;
	    int bestk = -1;
	    for (int k = 0; k < dist.size(); ++k) {
	      if(dist[k] > best) {best = dist[k]; bestk = k; }
   	    }
	    if (j < labels[i].size() && labels[i][j] == bestk) (*cor)++;
	  
	    if (j < labels[i].size()) errs.push_back(pickneglogsoftmax(i_l, labels[i][j]));
	    else errs.push_back(pickneglogsoftmax(i_l, labels[i][labels[i].size()-1]));
	    j += 1;
	    if(bestk == endlabel) break;
	  }
       }
       if(predict) (*predict) += j;
       if(overall) (*overall) += labels[i].size();
     }
    }
//    cerr<<"errs done"<<endl;
    return sum(errs);
  }
};

class Instance{
public:
        vector<int> words;
	vector< vector<int> > chars;
	vector< vector<int> > labels;
        Instance(const vector<int>& _words, const vector< vector<int> > _chars, const vector< vector<int> > _labels){
		words = _words;
		chars = _chars;
		labels = _labels;
	};
	~Instance(){};
	void show(){
		for(unsigned i = 0; i < words.size(); i ++){
			cerr<<wd.Convert(words[i])<<" ";
		}
		cerr<<"||| ";
		for(unsigned i = 0; i < chars.size(); i ++){
			for(unsigned j = 0; j < chars[i].size(); j ++){
				cerr<<cd.Convert(chars[i][j]);
			}
			cerr<<" ";
		}
		cerr<<"||| ";
		for(unsigned i = 0; i < labels.size(); i ++){
			for(unsigned j = 0; j < labels[i].size(); j ++){
				cerr<<ld.Convert(labels[i][j])<<" ";
			}
			cerr<<"||| ";
		}
		cerr<<endl;
		
	};
};

void normalize_digital_lower(string& line){
  for(unsigned i = 0; i < line.size(); i ++){
    if(line[i] >= '0' && line[i] <= '9'){
      line[i] = '0';
    }
    else if(line[i] >= 'A' && line[i] <= 'Z'){
      line[i] = line[i] - 'A' + 'a';
    }
  }
}

void ReadSentencePair(const string& line, vector<int>* s, cnn::Dict* sd, vector< vector<int> >* c, cnn::Dict* cd, vector< vector<int> >& l, cnn::Dict* ld, int& unk, int& total){
  istringstream in(line);
  string word;
  string sep = "|||";
  int f = 0;

  while(in) {
    in >> word;
    if (!in) break;

    if (word == sep) { f+=1; continue; }
    
    if(f == 0) {
      total += 1;
      vector<int> ctmp;
      for(unsigned i = 0; i < word.size(); i ++){
        ctmp.push_back(cd->Convert(string(1,word[i])));
      }
      c->push_back(ctmp);
      normalize_digital_lower(word);
      int k = sd->Convert(word);
      if(k == kUNK) unk++;
      s->push_back(k);
    }
    else {
      if(f == 1) l.resize(s->size());
      l[f-1].push_back(ld->Convert(word));
    }
  }
}

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv, 1989);
  if (argc != 3 && argc != 4) {
    cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [model.params]\n";
    return 1;
  }
  vector<Instance> training,dev;
  string line;

  int tlc = 0;
  int ttoks = 0;
  
  cerr << "Reading training data from " << argv[1] << "...\n";
  {
    int total = 0;
    int unk = 0;
    ifstream in(argv[1]);
    assert(in);
    while(getline(in, line)) {
      ++tlc;
      vector<int> x;
      vector< vector<int> > y;
      vector< vector<int> > c;
      ReadSentencePair(line, &x, &wd, &c, &cd, y, &ld, unk, total);
      assert(x.size() == y.size() && x.size() == c.size());
      if (x.size() == 0) { cerr << line << endl; abort(); }
      training.push_back(Instance(x,c,y));
      ttoks += x.size();
    }
    cerr << tlc << " lines, " << ttoks << " tokens, " << wd.size() << " types\n";
    cerr << "OOV ratio: " << float(unk)/total<<"\n";
    cerr << "Tags: " << ld.size() << endl;
  }
  wd.Freeze();
  wd.SetUnk("*UNK*");
  endlabel = ld.Convert("NULL");
  cd.Freeze();
  ld.Freeze(); // no new tag types allowed

  cd.SetUnk("*UNK*");
  ld.SetUnk("*UNK*");  
  VOCAB_SIZE = wd.size();
  CHAR_SIZE = cd.size();
  LABEL_SIZE = ld.size();

  int dlc = 0;
  int dtoks = 0;
  cerr << "Reading dev data from " << argv[2] << "...\n";
  {
    int total = 0;
    int unk = 0;
    ifstream in(argv[2]);
    assert(in);
    while(getline(in, line)) {
      ++dlc;
      vector<int> x;
      vector< vector<int> > y;
      vector< vector<int> > c;
      ReadSentencePair(line, &x, &wd, &c, &cd, y, &ld, unk, total);
      assert(x.size() == y.size());
      dev.push_back(Instance(x,c,y));
      dtoks += x.size();
    }
    cerr << dlc << " lines, " << dtoks << " tokens\n";
    cerr << "OOV ratio: " << float(unk)/total<<"\n";
  }
  ostringstream os;
  os << "ordered"
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;
  double best = -9e+99;

  Model model;
  bool use_momentum = true;
  Trainer* sgd = nullptr;
  if (use_momentum)
    sgd = new MomentumSGDTrainer(&model);
  else
    sgd = new SimpleSGDTrainer(&model);

  RNNLanguageModel<FastLSTMBuilder> lm(model);
  //RNNLanguageModel<SimpleRNNBuilder> lm(model);
  if (argc == 4) {
    string fname = argv[3];
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  unsigned report_every_i = 50;
  unsigned dev_every_i_reports = 25;
  unsigned si = training.size();
  vector<unsigned> order(training.size());
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
  bool first = true;
  int report = 0;
  unsigned lines = 0;
  while(1) {
    Timer iteration("completed in");
    double loss = 0;
    unsigned tpredict = 0;
    unsigned toverall = 0;
    double correct = 0;
    for (unsigned i = 0; i < report_every_i; ++i) {
      if (si == training.size()) {
        si = 0;
        if (first) { first = false; } else { sgd->update_epoch(); }
        cerr << "**SHUFFLE\n";
        shuffle(order.begin(), order.end(), *rndeng);
      }

      // build graph for this instance
      ComputationGraph cg;
      auto& sent = training[order[si]];
//      cerr<<si<<" "<<order[si]<<endl;
//      sent.show();
      ++si;
      lm.BuildTaggingGraph(sent.words, sent.chars, sent.labels, cg, &correct, &tpredict, &toverall, true);
//      cerr<<"done"<<endl;
      loss += as_scalar(cg.forward());
      cg.backward();
      sgd->update(1.0);
      ++lines;
    }
    sgd->status();
    cerr << " E = " << (loss / toverall) << " ppl=" << exp(loss / toverall) << " (acc=" << (correct / toverall) << ") ";

    // show score on dev data?
    report++;
    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      unsigned dpredict = 0;
      unsigned doverall = 0;
      double dcorr = 0;
      eval = true;
      //lm.p_th2t->scale_parameters(pdrop);
      for (auto& sent : dev) {
        ComputationGraph cg;
        lm.BuildTaggingGraph(sent.words, sent.chars, sent.labels, cg, &dcorr, &dpredict, &doverall, false);
        dloss += as_scalar(cg.forward());
      }
      //lm.p_th2t->scale_parameters(1/pdrop);
      eval = false;
      double P = (dcorr / dpredict);
      double R = (dcorr / doverall);
      double F = 2*P*R / (P+R);
      if (F > best) {
        best = F;
        ofstream out(fname);
        boost::archive::text_oarchive oa(out);
        oa << model;
      }
      cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] P = " << P << " R = " << R << " F=" << F << ' ';
    }
  }
  delete sgd;
}

