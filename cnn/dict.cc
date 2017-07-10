#include "dict.h"

#include <string>
#include <vector>
#include <sstream>

using namespace std;

namespace cnn {

std::vector<int> ReadSentence(const std::string& line, Dict* sd) {
  std::istringstream in(line);
  std::string word;
  std::vector<int> res;
  while(in) {
    in >> word;
    if (!in || word.empty()) break;
    res.push_back(sd->Convert(word));
  }
  return res;
}

void ReadSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td) {
  std::istringstream in(line);
  std::string word;
  std::string sep = "|||";
  Dict* d = sd;
  std::vector<int>* v = s;
  while(in) {
    in >> word;
    if (!in) break;
    if (word == sep) { d = td; v = t; continue; }
    v->push_back(d->Convert(word));
  }
}

/*void ReadSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector< std::vector<int> >* c, Dict* cd, std::vector<int>* l, Dict* ld){
  std::istringstream in(line);
  std::string word;
  std::string sep = "|||";
  bool f = false;
  while(in) {
    in >> word;
    if (!in) break;
    if (word == sep) { f = true; continue; }
    if(f == false) {
      std::vector<int> ctmp;
      for(unsigned i = 0; i < word.size(); i ++){
        ctmp.push_back(cd->Convert(std::string(1,word[i])));
      }
      c->push_back(ctmp);
      normalize_digital_lower(word);
      s->push_back(sd->Convert(word));
    }
    else {
      l->push_back(ld->Convert(word));
    }
  }
}*/
/*bool is_startwith(const std::string& word, const std::string& prefix) {
  if (word.size() < prefix.size())
    return false;
  for (unsigned int i = 0; i < prefix.size(); i++) {
    if (word[i] != prefix[i]) {
      return false;
    }
  }
  return true;
}
void normalize_digital_lower(std::string& line){
  for(unsigned i = 0; i < line.size(); i ++){
    if(line[i] >= '0' && line[i] <= '9'){
      line[i] = '0';
    }
    else if(line[i] >= 'A' && line[i] <= 'Z'){
      line[i] = line[i] - 'A' + 'a';
    }
  }
}
void ReadSentenceMultilabel(const std::string& line, 
			    Example_item* s, Dict* sd, Dict* cd, Dict* td, Dict* ld,
			    int* unks=NULL, int* unkc=NULL, int* unkt=NULL){
  std::istringstream in(line);
  std::string item;
  if(in) {
    in >> item;
    normalize_digital_lower(item);
    s->s = sd->Convert(item);
    if(s->s == 0 && unks != NULL){
        *unks += 1;
    }
  }
  while(in) {
    in >> item;
    if(is_startwith(item,"[C]")) {
   	(s->c).push_back(cd->Convert(item));
    	if((s->c).back() == 0 && unkc!=NULL) *unkc += 1;
    }
    else if(is_startwith(item,"[T]")) {
 	s->t = td->Convert(item);
        if(s->t == 0 && unkt!=NULL) *unkt += 1;
    }
    else if(is_startwith(item,"[L]")) (s->l).push_back(ld->Convert(item));
  }
}
void ReadWordEmbeding(const std::string& line,
			int* s, Dict* sd,
			std::vector<float> *e){
  std::istringstream in(line);
  std::string item;
  float one;
  if(in) {
    in >> item;
    normalize_digital_lower(item);
    *s = sd->Convert(item);
  }
  while(in) {
    in >> one;
    e->push_back(one);
  }
  assert(e->size() == 50);
}*/	
} // namespace cnn

