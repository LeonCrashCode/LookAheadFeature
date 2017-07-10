#include "cnn/forestlstm_var1.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/nodes.h"

using namespace std;
using namespace cnn::expr;

namespace cnn {

enum { X2I, C2I, BI, X2F, C2F, BF, X2O, C2O, BO, X2C, BC };
enum { H2I, H2O, H2C};
enum { H2F};
// See "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks"
// by Tai, Socher, and Manning (2015), section 3.2, for details on this model.
// http://arxiv.org/pdf/1503.00075v3.pdf
ForestLSTMBuilder_var1::ForestLSTMBuilder_var1(unsigned N,
                         unsigned layers,
                         unsigned input_dim,
                         unsigned hidden_dim,
                         Model* model) : layers(layers), N(N) {
  unsigned layer_input_dim = input_dim;
  for (unsigned i = 0; i < layers; ++i) {
    // i
    Parameters* p_x2i = model->add_parameters({hidden_dim, layer_input_dim});
    Parameters* p_c2i = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_bi = model->add_parameters({hidden_dim});
    vector<Parameters*> p_h2i(N);
    for (unsigned j = 0; j < N; ++j) {p_h2i[j] = model->add_parameters({hidden_dim, hidden_dim});}
    // f
    Parameters* p_x2f = model->add_parameters({hidden_dim, layer_input_dim});
    Parameters* p_c2f = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_bf = model->add_parameters({hidden_dim});
    vector< vector<Parameters*> > p_h2f(N);
    for (unsigned j = 0; j < N; ++j) {
        p_h2f[j].resize(N);
	for (unsigned k = 0; k < N; ++k) {
	    p_h2f[j][k] = model->add_parameters({hidden_dim, hidden_dim});
	}
    }
    // o
    Parameters* p_x2o = model->add_parameters({hidden_dim, layer_input_dim});
    Parameters* p_c2o = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_bo = model->add_parameters({hidden_dim});
    vector<Parameters*> p_h2o(N);
    for (unsigned j = 0; j < N; ++j) {p_h2o[j] = model->add_parameters({hidden_dim, hidden_dim});}
    // c (a.k.a. u)
    Parameters* p_x2c = model->add_parameters({hidden_dim, layer_input_dim});
    Parameters* p_bc = model->add_parameters({hidden_dim});
    vector<Parameters*> p_h2c(N);
    for (unsigned j = 0; j < N; ++j) {p_h2c[j] = model->add_parameters({hidden_dim, hidden_dim});}
    layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

    vector<Parameters*> ps = {p_x2i, p_c2i, p_bi, p_x2f, p_c2f, p_bf, p_x2o, p_c2o, p_bo, p_x2c, p_bc};
    vector< vector<Parameters*> > iocps = {p_h2i, p_h2o, p_h2c};
    params.push_back(ps);
    iocparams.push_back(iocps);
    fparams.push_back(p_h2f);

    Parameters* p_w = model->add_parameters({hidden_dim, hidden_dim});
    wparams.push_back(p_w);  
  }  // layers
  uparams = model->add_parameters({1,hidden_dim}); 
  
}

void ForestLSTMBuilder_var1::new_graph_impl(ComputationGraph& cg){
  param_vars.clear();
  for (unsigned i = 0; i < layers; ++i){
    auto& p = params[i];
    auto& iocp = iocparams[i];
    auto& fp = fparams[i];

    //i
    Expression i_x2i = parameter(cg, p[X2I]);
    Expression i_c2i = parameter(cg, p[C2I]);
    Expression i_bi = parameter(cg, p[BI]);
    vector<Expression> i_h2i(N);
    for (unsigned j = 0; j < N; ++j){
	i_h2i[j] = parameter(cg, iocp[H2I][j]);
    }   
    //f
    Expression i_x2f = parameter(cg, p[X2F]);
    Expression i_c2f = parameter(cg, p[C2F]);
    Expression i_bf = parameter(cg, p[BF]);
    vector<vector<Expression> > i_h2f(N);
    for (unsigned j = 0; j < N; ++j){
	i_h2f[j].resize(N);
	for (unsigned k = 0; k < N; ++k) {
	    i_h2f[j][k] = parameter(cg, fp[j][k]);
	}
    }
    
    //b
    Expression i_x2o = parameter(cg, p[X2O]);
    Expression i_c2o = parameter(cg, p[C2O]);
    Expression i_bo = parameter(cg, p[BO]);  
    vector<Expression> i_h2o(N);
    for (unsigned j = 0; j < N; ++j){
        i_h2o[j] = parameter(cg, iocp[H2O][j]);
    }
    //c
    Expression i_x2c = parameter(cg, p[X2C]);
    Expression i_bc = parameter(cg, p[BC]);
    vector<Expression> i_h2c(N);
    for (unsigned j = 0; j < N; ++j){
        i_h2c[j] = parameter(cg, iocp[H2C][j]);
    }

    vector<Expression> vars = {i_x2i, i_c2i, i_bi, i_x2f, i_c2f, i_bf, i_x2o, i_c2o, i_bo, i_x2c, i_bc};
    param_vars.push_back(vars);
   
    vector<vector<Expression> > iocvars = {i_h2i, i_h2o, i_h2c};
    iocparam_vars.push_back(iocvars);

    fparam_vars.push_back(i_h2f);

    auto& wp = wparams[i];
    i_w.push_back(parameter(cg, wp));
  }
  i_u = parameter(cg, uparams);
}

// layout: 0..layers = c
//         layers+1..2*layers = h
void ForestLSTMBuilder_var1::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
  c.clear();
  if (hinit.size() > 0) {
    assert(layers*2 == hinit.size());
    h0.resize(layers);
    c0.resize(layers);
    for (unsigned i = 0; i < layers; ++i) {
      c0[i] = hinit[i];
      h0[i] = hinit[i + layers];
    }
    has_initial_state = true;
  } else {
    has_initial_state = false;
  }
}

Expression ForestLSTMBuilder_var1::add_input(const vector< vector<int> >& childrens, const Expression& x) {
  ComputationGraph& cg = *(ComputationGraph*)x.pg;
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  vector<Expression>& ht = h.back();
  vector<Expression>& ct = c.back();

  vector< vector<Expression> > h_tmp(childrens.size());
  vector< vector<Expression> > c_tmp(childrens.size());

  for(unsigned chd = 0; chd < childrens.size(); ++chd){
    const vector<int>& children = childrens[chd];
    h_tmp[chd].resize(layers);
    c_tmp[chd].resize(layers);

    Expression in = x;
    for (unsigned i = 0; i < layers; ++i) {
      const vector<Expression>& vars = param_vars[i];
      const vector< vector<Expression> >& iocvars = iocparam_vars[i];
      const vector< vector<Expression> >& fvars = fparam_vars[i];
      vector<Expression> i_h_children, i_c_children;
      i_h_children.reserve(children.size() > 1 ? children.size() : 1);
      i_c_children.reserve(children.size() > 1 ? children.size() : 1);

      bool has_prev_state = (children.size() > 0 || has_initial_state);
      if (children.size() == 0) {
        i_h_children.push_back(Expression());
        i_c_children.push_back(Expression());
        if (has_initial_state) {
        // intial value for h and c at timestep 0 in layer i
        // defaults to zero matrix input if not set in add_parameter_edges
          i_h_children[0] = h0[i];
          i_c_children[0] = c0[i];
        }
      }
      else {  // t > 0
        for (int child : children) {
          i_h_children.push_back(h[child][i]);
          i_c_children.push_back(c[child][i]);
        }
      }
//	cerr<<"pre done"<<endl;
    // input
      Expression i_ait;
      if (has_prev_state) {
        vector<Expression> xs = {vars[BI], vars[X2I], in};
        xs.reserve(2 * children.size() + 3);

        for(int j = 0; j < children.size(); ++j){
          int ej = j < N ? j : N - 1;
	  xs.push_back(iocvars[H2I][ej]);
          xs.push_back(i_h_children[ej]);
        }
      
        assert (xs.size() == 2 * children.size() + 3);
        i_ait = affine_transform(xs);
      }
      else
        i_ait = affine_transform({vars[BI], vars[X2I], in});
      Expression i_it = logistic(i_ait);

//	cerr<<"input done" <<endl;
    // forget
      vector<Expression> i_ft;
      for (int j = 0; j < children.size(); ++j)
      {
        int ej = j < N ? j : N - 1;
        Expression i_aft;
        if (has_prev_state) {
          vector<Expression> xs = {vars[BF], vars[X2F], in};
          xs.reserve(2 * children.size() + 3);

	  for (int k = 0; k < children.size(); ++k) {
	    int ek = k < N ? k : N -1;
	    xs.push_back(fvars[ek][ej]);
            xs.push_back(i_h_children[k]);
	    
	  }
          assert (xs.size() == 2 * children.size() + 3);
          i_aft = affine_transform(xs);
        }
        else
          i_aft = affine_transform({vars[BF], vars[X2F], in});
        i_ft.push_back(logistic(i_aft));
      }
//	cerr<<"forget done" <<endl;
    // write memory cell
      Expression i_awt;
      if (has_prev_state) {
        vector<Expression> xs = {vars[BC], vars[X2C], in};
      // This is the one and only place that should *not* condition on i_c_children
      // This should condition only on x (a.k.a. in), the bias (vars[BC]) and i_h_children
        xs.reserve(2 * children.size() + 3);
      
        for(int j = 0; j < children.size(); ++j){
          int ej = j < N ? j : N - 1;
          xs.push_back(iocvars[H2C][ej]);
          xs.push_back(i_h_children[ej]);
        }
        assert (xs.size() == 2 * children.size() + 3);
        i_awt = affine_transform(xs);
      }
      else
        i_awt = affine_transform({vars[BC], vars[X2C], in});
      Expression i_wt = tanh(i_awt);

//	cerr<<"cell done"<<endl;
    // compute new cell value
      if (has_prev_state) {
        Expression i_nwt = cwise_multiply(i_it, i_wt);
        vector<Expression> i_crts(children.size());
        for (unsigned j = 0; j < children.size(); ++j) {
          i_crts[j] = cwise_multiply(i_ft[j], i_c_children[j]);
        }
        Expression i_crt = sum(i_crts);
        c_tmp[chd][i] = i_crt + i_nwt;
      }
      else {
        c_tmp[chd][i] = cwise_multiply(i_it, i_wt);
      }
//	cerr <<"new cell done" <<endl;

    // output
      Expression i_aot;
      if (has_prev_state) {
        vector<Expression> xs = {vars[BO], vars[X2O], in};
        xs.reserve(2 * children.size() + 3);

        for(int j = 0; j < children.size(); ++j){
          int ej = j < N ? j : N - 1;
          xs.push_back(iocvars[H2O][ej]);
          xs.push_back(i_h_children[ej]);
        }
        assert (xs.size() == 2 * children.size() + 3);
        i_aot = affine_transform(xs);
      }
      else
        i_aot = affine_transform({vars[BO], vars[X2O], in});
      Expression i_ot = logistic(i_aot);

//	cerr<<"output done"<<endl;
    // Compute new h value
      Expression ph_t = tanh(c_tmp[chd][i]);
      in = h_tmp[chd][i] = cwise_multiply(i_ot, ph_t);
    }
  }
  
  vector<Expression> alpha_i(childrens.size());
  for(unsigned att = 0; att < childrens.size(); ++att) {
    Expression tmps = i_w[0] * h_tmp[att][0];
    for(unsigned i = 1; i < layers; ++i){
        tmps = tmps + i_w[i] * h_tmp[att][i];
    }
    alpha_i[att] = i_u * tanh(tmps);
  }
  Expression alpha = softmax(concatenate(alpha_i));

  for(unsigned i = 0; i < layers; ++i){
    vector<Expression> possibles_h;
    vector<Expression> possibles_c;
    for(unsigned chd = 0; chd < childrens.size(); ++ chd){
      possibles_h.push_back(h_tmp[chd][i]);
      possibles_c.push_back(c_tmp[chd][i]);
    }
    Expression possibles_h_cols = concatenate_cols(possibles_h);
    ht[i] = transpose( transpose(alpha) * transpose(possibles_h_cols));

    Expression possibles_c_cols = concatenate_cols(possibles_c);
    ct[i] = transpose( transpose(alpha) * transpose(possibles_c_cols));
  }
  return ht.back();
}

Expression ForestLSTMBuilder_var1::add_input_impl(int prev, const Expression& x) {
  assert (false);
  return x;
}

void ForestLSTMBuilder_var1::copy(const RNNBuilder & rnn) {
  const ForestLSTMBuilder_var1 & rnn_treelstm = (const ForestLSTMBuilder_var1&)rnn;
  assert(params.size() == rnn_treelstm.params.size());
  assert(iocparams.size() == rnn_treelstm.iocparams.size());
  assert(fparams.size() == rnn_treelstm.fparams.size());
  for(size_t i = 0; i < params.size(); ++i){
      for(size_t j = 0; j < params[i].size(); ++j){
        params[i][j]->copy(*rnn_treelstm.params[i][j]);
      }
  }
  for(size_t i = 0; i < iocparams.size(); ++i){
	assert(iocparams[i].size() == rnn_treelstm.iocparams.size());
	for(size_t j = 0; j < iocparams[i].size(); ++j){
	    for(size_t k = 0; k < iocparams[i][j].size(); ++k){
	    	iocparams[i][j][k]->copy(*rnn_treelstm.iocparams[i][j][k]);
	    }
	}
  }
  for(size_t i = 0; i < fparams.size(); ++i) {
      assert(fparams[i].size() == rnn_treelstm.fparams.size());
      for(size_t j = 0; j < fparams[i].size(); ++j) {
      	assert(fparams[i][j].size() == rnn_treelstm.fparams[i][j].size());
	for(size_t k = 0; k < fparams[i][j].size(); ++k){
	  iocparams[i][j][k]->copy(*rnn_treelstm.iocparams[i][j][k]);
	}
      }
  }
	
  for(size_t i = 0; i < wparams.size(); ++i)
	wparams[i]->copy(*rnn_treelstm.wparams[i]);
  uparams->copy(*rnn_treelstm.uparams);
}

} // namespace cnn
